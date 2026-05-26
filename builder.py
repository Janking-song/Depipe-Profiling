"""Construction entry point for local profiling stages."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .adapters import get_adapter
from .checkpoint_loader import CheckpointLoadReport, CheckpointSlicer
from .quantization import STAGE_QUANTIZATION_NONE, iter_bnb_linear_weight_names, parse_stage_quantization_spec, replace_linear_with_bnb_linear4bit
from .segment_stage import SegmentStage


logger = logging.getLogger(__name__)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(torch_dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype is None or torch_dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[str(torch_dtype).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch_dtype={torch_dtype!r}. Use auto, float32, float16, or bfloat16.") from exc


def _dtype_name(torch_dtype: torch.dtype | None) -> str:
    if torch_dtype is None:
        return "source"
    return str(torch_dtype).removeprefix("torch.")


def resolve_stage_checkpoint_path(stage_checkpoint: str | Path | None) -> Path | None:
    if stage_checkpoint is None:
        return None
    path = Path(stage_checkpoint)
    if path.suffix == ".safetensors":
        return path
    return path / "stage_model.safetensors"


def default_stage_checkpoint_path(
    *,
    model_family: str,
    model_path: str | Path,
    start_layer: int,
    end_layer: int,
    torch_dtype: torch.dtype | None,
    include_lm_head: bool = False,
) -> Path:
    model_name = Path(model_path).name
    checkpoint_name = f"{model_family}_{model_name}_layers_{start_layer}_{end_layer}_{_dtype_name(torch_dtype)}"
    if include_lm_head:
        checkpoint_name += "_with_lm_head"
    return Path(__file__).resolve().parent / "stage_checkpoints" / checkpoint_name / "stage_model.safetensors"


def build_segment_stage(
    model_family: str,
    model_path: str | Path,
    start_layer: int,
    end_layer: int,
    torch_dtype: str | torch.dtype | None = "auto",
    device: str | torch.device | None = "auto",
    *,
    attn_implementation: str = "eager",
    stage_checkpoint: str | Path | None = None,
    stage_checkpoint_dir: str | Path | None = None,
    include_lm_head: bool | None = None,
    return_report: bool = False,
) -> SegmentStage | tuple[SegmentStage, CheckpointLoadReport]:
    """Build a prompt-executable local stage from an existing stage checkpoint."""

    target_device = resolve_device(device)
    target_dtype = resolve_dtype(torch_dtype, target_device)
    adapter = get_adapter(model_family, Path(model_path), attn_implementation=attn_implementation)
    layer_range = adapter.validate_layer_range(start_layer, end_layer)
    local_config = adapter.make_local_config(layer_range)
    if stage_checkpoint is not None and stage_checkpoint_dir is not None:
        raise ValueError("Pass only one of stage_checkpoint or stage_checkpoint_dir.")
    stage_checkpoint_path = resolve_stage_checkpoint_path(stage_checkpoint if stage_checkpoint is not None else stage_checkpoint_dir)
    if stage_checkpoint_path is None:
        raise ValueError(
            "stage_checkpoint is required for profiling. "
            "Run export_stage_checkpoint.py first to materialize stage_model.safetensors."
        )
    if not stage_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Stage checkpoint does not exist: {stage_checkpoint_path}. "
            "Run export_stage_checkpoint.py first."
        )
    metadata = CheckpointSlicer.read_stage_checkpoint_metadata(stage_checkpoint_path)
    stage_includes_lm_head = (
        parse_metadata_bool(metadata.get("include_lm_head"), default=False)
        if include_lm_head is None
        else include_lm_head
    )

    logger.info(
        "Building %s stage from %s, global layers %d-%d (%d local layers), dtype=%s, device=%s, include_lm_head=%s.",
        adapter.family,
        model_path,
        layer_range.start_layer,
        layer_range.end_layer,
        layer_range.num_layers,
        target_dtype,
        target_device,
        stage_includes_lm_head,
    )

    with torch.device("meta"): # 在meta设备上构建阶段，避免占用实际GPU内存
        stage = SegmentStage(
            adapter=adapter,
            config=local_config,
            layer_range=layer_range,
            include_lm_head=stage_includes_lm_head,
        )

    expected_local_keys = set(stage.state_dict().keys())
    quantization_spec = parse_stage_quantization_spec(
        metadata.get("quantization_method", STAGE_QUANTIZATION_NONE),
        compute_dtype=resolve_optional_dtype(metadata.get("bnb_compute_dtype"), target_device),
    )
    logger.info("Loading materialized stage checkpoint: %s", stage_checkpoint_path)
    if quantization_spec.enabled:
        replace_linear_with_bnb_linear4bit(stage, quantization_spec)
        expected_local_keys = set(stage.state_dict().keys())
        stage.reset_runtime_modules(device=target_device)
        report = load_quantized_stage_from_checkpoint(
            stage=stage,
            stage_checkpoint_path=stage_checkpoint_path,
            expected_local_keys=expected_local_keys,
            target_device=target_device,
        )
    else:
        state_dict, report = CheckpointSlicer.load_materialized_stage_state_dict(
            stage_checkpoint_path=stage_checkpoint_path,
            adapter=adapter,
            layer_range=layer_range,
            expected_local_keys=expected_local_keys,
            torch_dtype=target_dtype,
            include_lm_head=stage_includes_lm_head,
        )

        incompatible = stage.load_state_dict(state_dict, strict=False, assign=True)
        del state_dict  # free memory
        report.missing_keys = sorted(set(report.missing_keys) | set(incompatible.missing_keys))
        report.unexpected_keys = sorted(set(report.unexpected_keys) | set(incompatible.unexpected_keys))
        logger.info("load_state_dict missing_keys=%s", report.missing_keys)
        logger.info("load_state_dict unexpected_keys=%s", report.unexpected_keys)

        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(
                "Materialized stage checkpoint did not exactly match the local stage. "
                f"missing={report.missing_keys}, unexpected={report.unexpected_keys}"
            )

        stage.reset_runtime_modules(device=target_device)
        stage.to(device=target_device)
    stage.eval()
    return (stage, report) if return_report else stage


def resolve_optional_dtype(value: str | None, device: torch.device) -> torch.dtype | None:
    if value in {None, "None", "none"}:
        return None
    normalized = value.removeprefix("torch.") if isinstance(value, str) else value
    return resolve_dtype(normalized, device)


def parse_metadata_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse metadata boolean value: {value!r}")


def load_quantized_stage_from_checkpoint(
    *,
    stage,
    stage_checkpoint_path: Path,
    expected_local_keys: set[str],
    target_device: torch.device,
) -> CheckpointLoadReport:
    import torch
    from safetensors import safe_open
    from transformers.integrations.bitsandbytes import set_module_quantized_tensor_to_device

    quantized_weight_names = set(iter_bnb_linear_weight_names(stage))
    loaded_local_keys: set[str] = set()
    used_checkpoint_keys: set[str] = set()
    unexpected_keys: set[str] = set()

    with safe_open(stage_checkpoint_path, framework="pt", device="cpu") as handle:
        file_keys = set(handle.keys())

        for weight_name in sorted(quantized_weight_names):
            if weight_name not in file_keys:
                continue
            quantized_stats = {}
            for checkpoint_key in sorted(file_keys):
                if checkpoint_key.startswith(weight_name + "."):
                    quantized_stats[checkpoint_key[len(weight_name) + 1 :]] = handle.get_tensor(checkpoint_key)
                    used_checkpoint_keys.add(checkpoint_key)
            set_module_quantized_tensor_to_device(
                stage,
                weight_name,
                target_device,
                value=handle.get_tensor(weight_name),
                quantized_stats=quantized_stats,
            )
            used_checkpoint_keys.add(weight_name)
            loaded_local_keys.add(weight_name)

        for checkpoint_key in sorted(file_keys - used_checkpoint_keys):
            if checkpoint_key not in expected_local_keys:
                unexpected_keys.add(checkpoint_key)
                continue
            set_module_quantized_tensor_to_device(
                stage,
                checkpoint_key,
                target_device,
                value=handle.get_tensor(checkpoint_key),
            )
            used_checkpoint_keys.add(checkpoint_key)
            loaded_local_keys.add(checkpoint_key)

    report = CheckpointLoadReport(
        selected_checkpoint_keys=sorted(used_checkpoint_keys),
        loaded_local_keys=sorted(loaded_local_keys),
        shard_to_keys={stage_checkpoint_path.name: sorted(used_checkpoint_keys)},
        missing_keys=sorted(expected_local_keys - loaded_local_keys),
        unexpected_keys=sorted(unexpected_keys),
        source="materialized_quantized_stage_checkpoint",
        stage_checkpoint_path=str(stage_checkpoint_path),
    )
    logger.info("Loaded quantized stage checkpoint: %s", stage_checkpoint_path)
    logger.info("Loaded local key count: %d.", report.loaded_key_count)
    logger.info("Missing keys after quantized stage checkpoint load: %s", report.missing_keys)
    logger.info("Unexpected keys after quantized stage checkpoint load: %s", report.unexpected_keys)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            "Quantized stage checkpoint did not exactly match the local stage. "
            f"missing={report.missing_keys}, unexpected={report.unexpected_keys}"
        )
    return report
