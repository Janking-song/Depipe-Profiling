"""Export a materialized stage checkpoint from a full HF checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from safetensors.torch import save_file
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from profiling.adapters import get_adapter
    from profiling.builder import default_stage_checkpoint_path, resolve_dtype, resolve_stage_checkpoint_path
    from profiling.checkpoint_loader import CheckpointSlicer
    from profiling.quantization import (
        STAGE_QUANTIZATION_CHOICES,
        STAGE_QUANTIZATION_NONE,
        parse_stage_quantization_spec,
        replace_linear_with_bnb_linear4bit,
    )
    from profiling.segment_stage import SegmentStage
else:
    from .adapters import get_adapter
    from .builder import default_stage_checkpoint_path, resolve_dtype, resolve_stage_checkpoint_path
    from .checkpoint_loader import CheckpointSlicer
    from .quantization import (
        STAGE_QUANTIZATION_CHOICES,
        STAGE_QUANTIZATION_NONE,
        parse_stage_quantization_spec,
        replace_linear_with_bnb_linear4bit,
    )
    from .segment_stage import SegmentStage


WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a small checkpoint for one local profiling stage.")
    parser.add_argument("--model-family", required=True, choices=["qwen2", "llama"])
    parser.add_argument("--model-path", required=True, help="Full Hugging Face checkpoint directory.")
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="Include and profile the final norm plus lm_head. Intended for full-model layer ranges.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory or .safetensors file. Defaults to profiling/stage_checkpoints/...",
    )
    parser.add_argument(
        "--torch-dtype",
        default="source",
        choices=["source", "auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"],
        help="Dtype stored in the small checkpoint. source/auto preserves original tensor dtype.",
    )
    parser.add_argument(
        "--quantization",
        default=STAGE_QUANTIZATION_NONE,
        choices=STAGE_QUANTIZATION_CHOICES,
        help="Stage checkpoint quantization mode.",
    )
    parser.add_argument(
        "--bnb-compute-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
        help="bitsandbytes 4-bit compute dtype used at inference time.",
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing stage checkpoint.")
    parser.add_argument(
        "--no-copy-assets",
        action="store_true",
        help="Do not copy config/tokenizer assets next to the stage checkpoint.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")

    export_dtype = resolve_export_dtype(args.torch_dtype)
    quantization_spec = parse_stage_quantization_spec(
        args.quantization,
        compute_dtype=resolve_dtype(args.bnb_compute_dtype, torch.device("cpu")),
    )
    adapter = get_adapter(args.model_family, args.model_path, attn_implementation=args.attn_implementation)
    layer_range = adapter.validate_layer_range(args.start_layer, args.end_layer)
    validate_lm_head_export(adapter, layer_range, include_lm_head=args.include_lm_head)
    local_config = adapter.make_local_config(layer_range)

    with torch.device("meta"):
        stage = SegmentStage(
            adapter=adapter,
            config=local_config,
            layer_range=layer_range,
            include_lm_head=args.include_lm_head,
        )
    expected_local_keys = set(stage.state_dict().keys())

    output_path = resolve_stage_checkpoint_path(args.output)
    if output_path is None:
        output_path = default_stage_checkpoint_path(
            model_family=adapter.family,
            model_path=args.model_path,
            start_layer=layer_range.start_layer,
            end_layer=layer_range.end_layer,
            torch_dtype=export_dtype,
            include_lm_head=args.include_lm_head,
        )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Stage checkpoint already exists: {output_path}. Pass --overwrite to replace it.")

    slicer = CheckpointSlicer(args.model_path)
    if quantization_spec.enabled:
        report = materialize_quantized_stage_checkpoint(
            slicer=slicer,
            adapter=adapter,
            local_config=local_config,
            layer_range=layer_range,
            expected_local_keys=expected_local_keys,
            output_path=output_path,
            export_dtype=export_dtype,
            quantization_spec=quantization_spec,
            include_lm_head=args.include_lm_head,
        )
    else:
        report = slicer.materialize_segment_checkpoint(
            adapter=adapter,
            layer_range=layer_range,
            expected_local_keys=expected_local_keys,
            torch_dtype=export_dtype,
            output_path=output_path,
            include_lm_head=args.include_lm_head,
        )
    write_stage_metadata(
        output_path=output_path,
        model_family=adapter.family,
        source_model_path=args.model_path,
        start_layer=layer_range.start_layer,
        end_layer=layer_range.end_layer,
        torch_dtype=export_dtype,
        loaded_key_count=report.loaded_key_count,
        quantization_method=quantization_spec.method,
        bnb_compute_dtype=quantization_spec.compute_dtype,
        include_lm_head=args.include_lm_head,
    )

    if not args.no_copy_assets:
        copy_model_assets(Path(args.model_path), output_path.parent, overwrite=args.overwrite)

    print(f"stage_checkpoint_path: {output_path}")
    print(f"loaded_key_count: {report.loaded_key_count}")


def resolve_export_dtype(value: str) -> torch.dtype | None:
    if value in {"source", "auto"}:
        return None
    return resolve_dtype(value, torch.device("cpu"))


def validate_lm_head_export(adapter, layer_range, *, include_lm_head: bool) -> None:
    if not include_lm_head:
        return
    total_layers = adapter.total_num_layers()
    if layer_range.start_layer != 0 or layer_range.end_layer != total_layers - 1:
        raise ValueError(
            "--include-lm-head is intended for full-model exports. "
            f"Use --start-layer 0 --end-layer {total_layers - 1} for this checkpoint."
        )


def materialize_quantized_stage_checkpoint(
    *,
    slicer: CheckpointSlicer,
    adapter,
    local_config,
    layer_range,
    expected_local_keys: set[str],
    output_path: Path,
    export_dtype: torch.dtype | None,
    quantization_spec,
    include_lm_head: bool,
):
    local_state_dict, report = slicer.load_local_stage_state_dict_from_source_checkpoint(
        adapter=adapter,
        layer_range=layer_range,
        expected_local_keys=expected_local_keys,
        torch_dtype=export_dtype,
        include_lm_head=include_lm_head,
    )
    quant_stage = SegmentStage(
        adapter=adapter,
        config=local_config,
        layer_range=layer_range,
        include_lm_head=include_lm_head,
    )
    model_dtype = infer_floating_dtype(local_state_dict)
    if model_dtype is not None:
        quant_stage = quant_stage.to(dtype=model_dtype)
    replace_linear_with_bnb_linear4bit(quant_stage, quantization_spec)
    incompatible = quant_stage.load_state_dict(local_state_dict, strict=False)
    del local_state_dict
    if incompatible.missing_keys or incompatible.unexpected_keys or report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            "Quantized local stage load failed before quantization. "
            f"missing={sorted(set(report.missing_keys) | set(incompatible.missing_keys))}, "
            f"unexpected={sorted(set(report.unexpected_keys) | set(incompatible.unexpected_keys))}"
        )
    quant_stage.reset_runtime_modules(device="cpu")
    quant_stage = quant_stage.to("cpu")

    quantized_state_dict = {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in quant_stage.state_dict().items()
    }
    metadata = {
        "format": "depipe_segment_stage",
        "model_family": adapter.family,
        "source_model_path": str(adapter.model_path),
        "start_layer": str(layer_range.start_layer),
        "end_layer": str(layer_range.end_layer),
        "num_local_layers": str(layer_range.num_layers),
        "torch_dtype": "source" if export_dtype is None else str(export_dtype),
        "quantization_method": quantization_spec.method,
        "bnb_compute_dtype": str(quantization_spec.compute_dtype),
        "checkpoint_key_space": "local_stage",
        "include_lm_head": str(include_lm_head),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(quantized_state_dict, output_path, metadata=metadata)
    del quantized_state_dict
    logging.getLogger(__name__).info("Wrote quantized stage checkpoint: %s", output_path)
    report.stage_checkpoint_path = str(output_path)
    return report


def infer_floating_dtype(state_dict: dict[str, torch.Tensor]) -> torch.dtype | None:
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            return tensor.dtype
    return None


def copy_model_assets(source_model_path: Path, output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in source_model_path.iterdir():
        if not source.is_file() or is_weight_file(source):
            continue
        destination = output_dir / source.name
        if destination.exists() and not overwrite:
            continue
        shutil.copy2(source, destination)


def is_weight_file(path: Path) -> bool:
    name = path.name
    if name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json"):
        return True
    return path.suffix in WEIGHT_SUFFIXES


def write_stage_metadata(
    *,
    output_path: Path,
    model_family: str,
    source_model_path: str,
    start_layer: int,
    end_layer: int,
    torch_dtype: torch.dtype | None,
    loaded_key_count: int,
    quantization_method: str,
    bnb_compute_dtype: torch.dtype | None,
    include_lm_head: bool,
) -> None:
    metadata = {
        "format": "depipe_segment_stage",
        "model_family": model_family,
        "source_model_path": str(source_model_path),
        "stage_checkpoint_path": str(output_path),
        "start_layer": start_layer,
        "end_layer": end_layer,
        "num_local_layers": end_layer - start_layer + 1,
        "torch_dtype": "source" if torch_dtype is None else str(torch_dtype),
        "loaded_key_count": loaded_key_count,
        "quantization_method": quantization_method,
        "bnb_compute_dtype": None if bnb_compute_dtype is None else str(bnb_compute_dtype),
        "checkpoint_key_space": "local_stage" if quantization_method != STAGE_QUANTIZATION_NONE else "source_checkpoint",
        "include_lm_head": include_lm_head,
    }
    metadata_path = output_path.parent / "stage_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
