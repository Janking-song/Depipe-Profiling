"""CLI for profiling an inclusive Transformer block interval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from profiling.builder import build_segment_stage, resolve_device, resolve_dtype, resolve_stage_checkpoint_path
    from profiling.checkpoint_loader import CheckpointSlicer
    from profiling.sample_cache import (
        DEFAULT_PROMPT,
        DEFAULT_SAMPLE_SEED,
        DEFAULT_TARGET_TOKEN_LENGTH,
        choose_profile_sample_indices,
        load_profile_sample_cache,
        parse_sample_indices,
    )
else:
    from .builder import build_segment_stage, resolve_device, resolve_dtype, resolve_stage_checkpoint_path
    from .checkpoint_loader import CheckpointSlicer
    from .sample_cache import (
        DEFAULT_PROMPT,
        DEFAULT_SAMPLE_SEED,
        DEFAULT_TARGET_TOKEN_LENGTH,
        choose_profile_sample_indices,
        load_profile_sample_cache,
        parse_sample_indices,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a local executable stage from a small stage checkpoint.")
    parser.add_argument("--model-family", default=None, choices=["qwen2", "llama"])
    parser.add_argument(
        "--model-path",
        default=None,
        help="Directory with config/tokenizer assets. Defaults to the stage checkpoint directory.",
    )
    parser.add_argument("--start-layer", type=int, default=None)
    parser.add_argument("--end-layer", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--sample-file", default=None, help="Path to a cached profiling sample .pt file.")
    parser.add_argument("--num-profile-samples", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--sample-indices", default=None, help="Comma-separated cache indices. Overrides --num-profile-samples.")
    parser.add_argument("--print-sample-summary", action="store_true", help="Print the selected sample indices and source metadata.")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measure-runs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"])
    parser.add_argument("--attn-implementation", default="eager", help="HF attention implementation, e.g. eager or sdpa.")
    parser.add_argument(
        "--stage-checkpoint",
        default=None,
        help="Materialized stage .safetensors file, or a directory containing stage_model.safetensors.",
    )
    parser.add_argument("--stage-checkpoint-dir", default=None, help="Alias for --stage-checkpoint.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    if args.prompt is not None and args.sample_file is not None:
        raise ValueError("Pass only one of --prompt or --sample-file.")

    stage_checkpoint_path = resolve_profile_stage_checkpoint(args)
    metadata = CheckpointSlicer.read_stage_checkpoint_metadata(stage_checkpoint_path)
    model_family = args.model_family or metadata_value(metadata, "model_family")
    start_layer = args.start_layer if args.start_layer is not None else int(metadata_value(metadata, "start_layer"))
    end_layer = args.end_layer if args.end_layer is not None else int(metadata_value(metadata, "end_layer"))
    model_path = Path(args.model_path) if args.model_path is not None else stage_checkpoint_path.parent
    quantization_method = metadata.get("quantization_method", "none")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)
    stage, report = build_segment_stage(
        model_family=model_family,
        model_path=model_path,
        start_layer=start_layer,
        end_layer=end_layer,
        torch_dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
        stage_checkpoint=stage_checkpoint_path,
        return_report=True,
    )

    print(stage)

    sample_context = None
    if args.sample_file is not None:
        sample_context = prepare_cached_samples(
            sample_file=Path(args.sample_file),
            requested_model_family=model_family,
            num_profile_samples=args.num_profile_samples,
            sample_seed=args.sample_seed,
            sample_indices=parse_sample_indices(args.sample_indices),
        )
    else:
        sample_context = prepare_prompt_input(stage, args.prompt or DEFAULT_PROMPT)

    print_header(
        model_family=model_family,
        model_path=model_path,
        start_layer=start_layer,
        end_layer=end_layer,
        attn_implementation=args.attn_implementation,
        stage=stage,
        dtype=dtype,
        stage_checkpoint_path=report.stage_checkpoint_path,
        quantization_method=quantization_method,
        sample_context=sample_context,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
    )  # local mapping
    if args.print_sample_summary and sample_context["mode"] == "sample_file":
        print_sample_summary(sample_context)

    with torch.inference_mode():
        if sample_context["mode"] == "sample_file":
            profile_cached_samples(
                stage=stage,
                timing_labels=stage.timing_labels(include_embed_tokens=True),
                sample_context=sample_context,
                warmup_runs=args.warmup_runs,
                measure_runs=args.measure_runs,
            )
        else:
            measure_timings = measure_single_input(
                stage,
                input_ids=sample_context["input_ids"].to(stage.device),
                attention_mask=tensor_to_device(sample_context["attention_mask"], stage.device),
                warmup_runs=args.warmup_runs,
                measure_runs=args.measure_runs,
            )
            print_average(
                stage.timing_labels(include_embed_tokens=True),
                measure_timings,
                title="average time over measure runs only:",
            )


def resolve_profile_stage_checkpoint(args: argparse.Namespace) -> Path:
    if args.stage_checkpoint is not None and args.stage_checkpoint_dir is not None:
        raise ValueError("Pass only one of --stage-checkpoint or --stage-checkpoint-dir.")
    stage_checkpoint_path = resolve_stage_checkpoint_path(args.stage_checkpoint or args.stage_checkpoint_dir)
    if stage_checkpoint_path is not None:
        return stage_checkpoint_path
    if args.model_path is None:
        raise ValueError("Pass --stage-checkpoint, or pass --model-path pointing to a stage export directory.")
    return Path(args.model_path) / "stage_model.safetensors"


def metadata_value(metadata: dict[str, str], key: str) -> str:
    try:
        return metadata[key]
    except KeyError as exc:
        raise ValueError(f"Stage checkpoint metadata is missing {key!r}; pass it explicitly on the CLI.") from exc


def prepare_prompt_input(stage, prompt: str) -> dict:
    tokenizer = stage.load_tokenizer()
    encoded = tokenizer(prompt, return_tensors="pt")
    return {
        "mode": "prompt",
        "prompt": prompt,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
    }


def prepare_cached_samples(
    *,
    sample_file: Path,
    requested_model_family: str,
    num_profile_samples: int,
    sample_seed: int,
    sample_indices: list[int] | None,
) -> dict:
    bundle = load_profile_sample_cache(sample_file)
    metadata = bundle["metadata"]
    sample_model_family = str(metadata.get("model_family"))
    if sample_model_family != requested_model_family:
        raise ValueError(
            f"Profile sample file {sample_file} was built for model_family={sample_model_family!r}, "
            f"but the stage expects {requested_model_family!r}."
        )

    target_length = int(metadata["target_length"])
    if target_length != DEFAULT_TARGET_TOKEN_LENGTH:
        raise ValueError(
            f"Current profiling entrypoint expects cached samples of length {DEFAULT_TARGET_TOKEN_LENGTH}, "
            f"but {sample_file} contains target_length={target_length}."
        )

    selected_indices = choose_profile_sample_indices(
        total_samples=bundle["input_ids"].shape[0],
        num_profile_samples=num_profile_samples,
        seed=sample_seed,
        explicit_indices=sample_indices,
    )
    return {
        "mode": "sample_file",
        "sample_file": str(sample_file),
        "metadata": metadata,
        "input_ids": bundle["input_ids"],
        "attention_mask": bundle["attention_mask"],
        "token_length_before_truncation": bundle["token_length_before_truncation"],
        "source_index": bundle["source_index"],
        "source_prompt_preview": bundle["source_prompt_preview"],
        "selected_indices": selected_indices,
        "num_profile_samples": len(selected_indices),
        "sample_seed": sample_seed,
        "target_length": target_length,
    }


def measure_single_input(stage, *, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, warmup_runs: int, measure_runs: int) -> list[list]:
    """Return only profile samples collected from measure runs.

    Warmup forwards are executed first and intentionally discarded.
    """

    for _ in range(warmup_runs):
        _ = stage(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    measure_profiles: list[list] = []
    for _ in range(measure_runs):
        _, profiles = stage.forward_with_layer_timings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            include_embed_tokens=True,
        )
        measure_profiles.append(profiles)
    return measure_profiles


def tensor_to_device(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(device)


def profile_cached_samples(stage, *, timing_labels: list[str], sample_context: dict, warmup_runs: int, measure_runs: int) -> None:
    per_sample_averages: list[list] = []
    sample_records: list[dict] = []

    for cache_index in sample_context["selected_indices"]:
        input_ids = sample_context["input_ids"][cache_index].unsqueeze(0).to(stage.device)
        attention_mask = sample_context["attention_mask"][cache_index].unsqueeze(0).to(stage.device)
        measure_timings = measure_single_input(
            stage,
            input_ids=input_ids,
            attention_mask=attention_mask,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
        )
        averages = average_profiles(measure_timings)
        per_sample_averages.append(averages)
        sample_records.append(
            {
                "cache_index": cache_index,
                "source_index": int(sample_context["source_index"][cache_index]),
                "token_length_before_truncation": int(sample_context["token_length_before_truncation"][cache_index]),
                "prompt_preview": sample_context["source_prompt_preview"][cache_index],
                "averages": averages,
            }
        )

    print_per_sample_average(timing_labels, sample_records)
    print_average(timing_labels, per_sample_averages, title="overall average across sampled inputs (measure runs only):")


def print_header(
    *,
    model_family: str,
    model_path: Path,
    start_layer: int,
    end_layer: int,
    attn_implementation: str,
    stage,
    dtype: torch.dtype,
    stage_checkpoint_path: str | None,
    quantization_method: str,
    sample_context: dict,
    warmup_runs: int,
    measure_runs: int,
) -> None:
    print("=" * 79)
    print("segment profiling")
    print(f"model_family: {model_family}")
    print(f"model_path: {model_path}")
    print(f"global_layer_range: {start_layer}-{end_layer}")
    print(f"local_num_layers: {stage.num_profiled_layers}")
    print(f"device: {stage.device}")
    print(f"torch_dtype: {dtype}")
    print(f"attn_implementation: {attn_implementation}")  # eager √ (没有用flash attention) or sdpa (使用了flash attention)
    print(f"quantization_method: {quantization_method}")
    print(f"include_lm_head: {stage.include_lm_head}")
    print(f"warmup_runs: {warmup_runs}")
    print(f"measure_runs: {measure_runs}")
    if stage_checkpoint_path:
        print(f"stage_checkpoint_path: {stage_checkpoint_path}")
    if sample_context["mode"] == "sample_file":
        print(f"sample_file: {sample_context['sample_file']}")
        print(f"num_profile_samples: {sample_context['num_profile_samples']}")
        print(f"sample_seed: {sample_context['sample_seed']}")
        print(f"target_token_length: {sample_context['target_length']}")
    else:
        token_length = sample_context["input_ids"].shape[1]
        print(f"prompt_token_length: {token_length}")
    print("timing targets:")
    for label in stage.timing_labels(include_embed_tokens=True):
        print(f"  {label}")
    print("=" * 79)


def print_run(run_idx: int, timing_labels: list[str], timings: list) -> None:
    print(f"run {run_idx + 1}:")
    for label, profile in zip(timing_labels, timings, strict=True):
        print(
            f"  {label}: {profile_elapsed_ms(profile):.3f} ms "
            f"param {profile_parameter_memory_mib(profile):.2f} MiB "
            f"forward_peak {profile_forward_peak_memory_mib(profile):.2f} MiB"
        )


def print_sample_summary(sample_context: dict) -> None:
    print("=" * 79)
    print("selected cached samples:")
    for cache_index in sample_context["selected_indices"]:
        source_index = int(sample_context["source_index"][cache_index])
        token_length = int(sample_context["token_length_before_truncation"][cache_index])
        preview = sample_context["source_prompt_preview"][cache_index]
        print(
            f"  cache_index={cache_index:04d} source_index={source_index:04d} "
            f"original_tokens={token_length:04d} preview={preview!r}"
        )
    print("=" * 79)


def print_per_sample_average(timing_labels: list[str], sample_records: list[dict]) -> None:
    print("=" * 79)
    print("per-sample average time (measure runs only):")
    for record in sample_records:
        print(
            f"sample cache_index={record['cache_index']:04d} "
            f"source_index={record['source_index']:04d} "
            f"original_tokens={record['token_length_before_truncation']:04d}"
        )
        for label, profile in zip(timing_labels, record["averages"], strict=True):
            print(
                f"  {label}: avg {profile_elapsed_ms(profile):.3f} ms "
                f"param {profile_parameter_memory_mib(profile):.2f} MiB "
                f"forward_peak {profile_forward_peak_memory_mib(profile):.2f} MiB"
            )


def average_profiles(all_profiles: list[list]) -> list[dict[str, float]]:
    if not all_profiles:
        return []
    num_layers = len(all_profiles[0])
    return [
        {
            "elapsed_ms": sum(profile_elapsed_ms(run[idx]) for run in all_profiles) / len(all_profiles),
            "parameter_memory_mib": sum(profile_parameter_memory_mib(run[idx]) for run in all_profiles) / len(all_profiles),
            "forward_peak_memory_mib": sum(profile_forward_peak_memory_mib(run[idx]) for run in all_profiles) / len(all_profiles),
        }
        for idx in range(num_layers)
    ]


def profile_elapsed_ms(profile) -> float:
    if isinstance(profile, dict):
        return float(profile["elapsed_ms"])
    return float(profile.elapsed_ms)


def profile_parameter_memory_mib(profile) -> float:
    if isinstance(profile, dict):
        return float(profile.get("parameter_memory_mib", 0.0))
    return float(profile.parameter_memory_mib)


def profile_forward_peak_memory_mib(profile) -> float:
    if isinstance(profile, dict):
        return float(profile["forward_peak_memory_mib"])
    return float(profile.peak_memory_mib)


def print_average(timing_labels: list[str], all_timings: list[list], *, title: str) -> None:
    if not all_timings:
        return
    print("=" * 79)
    print(title)
    for idx, label in enumerate(timing_labels):
        avg_ms = sum(profile_elapsed_ms(run[idx]) for run in all_timings) / len(all_timings)
        avg_parameter_memory_mib = sum(profile_parameter_memory_mib(run[idx]) for run in all_timings) / len(all_timings)
        avg_forward_peak_memory_mib = sum(profile_forward_peak_memory_mib(run[idx]) for run in all_timings) / len(all_timings)
        print(
            f"  {label}: avg {avg_ms:.3f} ms "
            f"param {avg_parameter_memory_mib:.2f} MiB "
            f"forward_peak {avg_forward_peak_memory_mib:.2f} MiB"
        )


if __name__ == "__main__":
    main()
