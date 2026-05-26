"""CLI for profiling an inclusive Transformer block interval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

BYTES_PER_MIB = 1024.0 * 1024.0

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
    parser.add_argument(
        "--print-cuda-memory",
        action="store_true",
        help="Print PyTorch CUDA allocator memory statistics at key profiling points.",
    )
    parser.add_argument(
        "--active-cuda-oom-limit-mib",
        type=float,
        default=None,
        help=(
            "Enable active CUDA-memory OOM simulation. If the selected PyTorch CUDA "
            "allocator metric exceeds this threshold at a check point, the program "
            "raises torch.cuda.OutOfMemoryError and exits. Logs are printed in MiB."
        ),
    )
    parser.add_argument(
        "--active-cuda-oom-metric",
        default="max_reserved",
        choices=["allocated", "reserved", "max_allocated", "max_reserved"],
        help=(
            "Metric used by --active-cuda-oom-limit-mib. Use max_reserved for the "
            "most conservative budget check; use max_allocated if you only care about "
            "tensor-occupied memory."
        ),
    )
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
    if args.print_cuda_memory:
        print_cuda_mem("after_resolve_device_dtype", device)
        reset_cuda_peak_memory("before_stage_build", device)

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

    if args.print_cuda_memory:
        print_cuda_mem("after_stage_build", stage.device)
    enforce_active_cuda_oom_limit(
        "after_stage_build",
        stage.device,
        args.active_cuda_oom_limit_mib,
        args.active_cuda_oom_metric,
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

    if args.print_cuda_memory:
        print_cuda_mem("after_prepare_input_context", stage.device)
    enforce_active_cuda_oom_limit(
        "after_prepare_input_context",
        stage.device,
        args.active_cuda_oom_limit_mib,
        args.active_cuda_oom_metric,
    )

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
            if args.print_cuda_memory:
                reset_cuda_peak_memory("before_cached_profile", stage.device)
            profile_cached_samples(
                stage=stage,
                timing_labels=stage.timing_labels(include_embed_tokens=True),
                sample_context=sample_context,
                warmup_runs=args.warmup_runs,
                measure_runs=args.measure_runs,
                print_cuda_memory=args.print_cuda_memory,
                active_cuda_oom_limit_mib=args.active_cuda_oom_limit_mib,
                active_cuda_oom_metric=args.active_cuda_oom_metric,
            )
            if args.print_cuda_memory:
                print_cuda_mem("after_cached_profile_all", stage.device)
            enforce_active_cuda_oom_limit(
                "after_cached_profile_all",
                stage.device,
                args.active_cuda_oom_limit_mib,
                args.active_cuda_oom_metric,
            )
        else:
            input_ids = sample_context["input_ids"].to(stage.device)
            attention_mask = tensor_to_device(sample_context["attention_mask"], stage.device)
            if args.print_cuda_memory:
                print_cuda_mem("after_prompt_tensors_to_device", stage.device)
            enforce_active_cuda_oom_limit(
                "after_prompt_tensors_to_device",
                stage.device,
                args.active_cuda_oom_limit_mib,
                args.active_cuda_oom_metric,
            )
            if args.print_cuda_memory:
                reset_cuda_peak_memory("before_prompt_profile", stage.device)
            measure_timings = measure_single_input(
                stage,
                input_ids=input_ids,
                attention_mask=attention_mask,
                warmup_runs=args.warmup_runs,
                measure_runs=args.measure_runs,
                active_cuda_oom_limit_mib=args.active_cuda_oom_limit_mib,
                active_cuda_oom_metric=args.active_cuda_oom_metric,
            )
            if args.print_cuda_memory:
                print_cuda_mem("after_prompt_profile", stage.device)
            enforce_active_cuda_oom_limit(
                "after_prompt_profile",
                stage.device,
                args.active_cuda_oom_limit_mib,
                args.active_cuda_oom_metric,
            )
            print_average(
                stage.timing_labels(include_embed_tokens=True),
                measure_timings,
                title="average time over measure runs only:",
            )


def is_cuda_device(device: torch.device | str | None) -> bool:
    """Return True when the requested device is CUDA and CUDA is available."""
    if device is None:
        return torch.cuda.is_available()
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError):
        return False
    return parsed.type == "cuda" and torch.cuda.is_available()


def cuda_device_index(device: torch.device | str | None) -> int | None:
    """Resolve a torch CUDA device object/string to an integer CUDA device index."""
    if not is_cuda_device(device):
        return None
    parsed = torch.device(device)
    if parsed.index is not None:
        return parsed.index
    return torch.cuda.current_device()


def bytes_to_mib(value: int | float) -> float:
    return float(value) / BYTES_PER_MIB


def reset_cuda_peak_memory(tag: str, device: torch.device | str | None = None) -> None:
    """Reset PyTorch CUDA peak-memory counters for a profiling interval."""
    index = cuda_device_index(device)
    if index is None:
        return
    torch.cuda.synchronize(index)
    torch.cuda.reset_peak_memory_stats(index)
    print(f"[cuda_mem][{tag}] reset_peak_memory_stats device=cuda:{index}", flush=True)


def print_cuda_mem(tag: str, device: torch.device | str | None = None) -> None:
    """Print PyTorch CUDA allocator statistics.

    These values describe PyTorch's CUDA caching allocator, not total Jetson
    board-level RAM. Use them together with tegrastats and cgroup MemoryCurrent.
    """
    index = cuda_device_index(device)
    if index is None:
        print(f"[cuda_mem][{tag}] cuda unavailable or device is not CUDA", flush=True)
        return

    torch.cuda.synchronize(index)
    allocated = torch.cuda.memory_allocated(index)
    reserved = torch.cuda.memory_reserved(index)
    max_allocated = torch.cuda.max_memory_allocated(index)
    max_reserved = torch.cuda.max_memory_reserved(index)

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        mem_info = (
            f", cuda_free={bytes_to_mib(free_bytes):.2f} MiB"
            f", cuda_total={bytes_to_mib(total_bytes):.2f} MiB"
        )
    except (AttributeError, RuntimeError):
        mem_info = ""

    print(
        f"[cuda_mem][{tag}] device=cuda:{index}"
        f", allocated={bytes_to_mib(allocated):.2f} MiB"
        f", reserved={bytes_to_mib(reserved):.2f} MiB"
        f", max_allocated={bytes_to_mib(max_allocated):.2f} MiB"
        f", max_reserved={bytes_to_mib(max_reserved):.2f} MiB"
        f"{mem_info}",
        flush=True,
    )


def get_cuda_memory_snapshot(device: torch.device | str | None = None) -> dict[str, float] | None:
    """Return current PyTorch CUDA allocator metrics in MiB."""
    index = cuda_device_index(device)
    if index is None:
        return None
    torch.cuda.synchronize(index)
    return {
        "allocated": bytes_to_mib(torch.cuda.memory_allocated(index)),
        "reserved": bytes_to_mib(torch.cuda.memory_reserved(index)),
        "max_allocated": bytes_to_mib(torch.cuda.max_memory_allocated(index)),
        "max_reserved": bytes_to_mib(torch.cuda.max_memory_reserved(index)),
    }


def enforce_active_cuda_oom_limit(
    tag: str,
    device: torch.device | str | None,
    limit_mib: float | None,
    metric: str = "max_reserved",
) -> None:
    """Actively abort when a PyTorch CUDA allocator metric exceeds a user budget.

    This is an application-level OOM simulation/check. It does not prevent a CUDA
    allocation before it happens; instead, it checks allocator statistics at
    deterministic points and raises torch.cuda.OutOfMemoryError when the selected
    metric is already above the configured threshold.
    """
    if limit_mib is None:
        return
    snapshot = get_cuda_memory_snapshot(device)
    if snapshot is None:
        return
    if metric not in snapshot:
        raise ValueError(f"Unsupported active CUDA OOM metric: {metric!r}")

    observed_mib = snapshot[metric]
    if observed_mib <= limit_mib:
        print(
            f"[active_cuda_oom][{tag}] OK: {metric}={observed_mib:.2f} MiB "
            f"<= limit={limit_mib:.2f} MiB "
            f"(allocated={snapshot['allocated']:.2f} MiB, reserved={snapshot['reserved']:.2f} MiB, "
            f"max_allocated={snapshot['max_allocated']:.2f} MiB, max_reserved={snapshot['max_reserved']:.2f} MiB)",
            flush=True,
        )
        return

    message = (
        f"[active_cuda_oom][{tag}] simulated CUDA OOM: {metric}={observed_mib:.2f} MiB "
        f"> limit={limit_mib:.2f} MiB. "
        f"snapshot: allocated={snapshot['allocated']:.2f} MiB, "
        f"reserved={snapshot['reserved']:.2f} MiB, "
        f"max_allocated={snapshot['max_allocated']:.2f} MiB, "
        f"max_reserved={snapshot['max_reserved']:.2f} MiB."
    )
    print(message, flush=True)
    exception_cls = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    raise exception_cls(message)


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


def measure_single_input(
    stage,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    warmup_runs: int,
    measure_runs: int,
    active_cuda_oom_limit_mib: float | None = None,
    active_cuda_oom_metric: str = "max_reserved",
) -> list[list]:
    """Return only profile samples collected from measure runs.

    Warmup forwards are executed first and intentionally discarded.
    """

    for warmup_idx in range(warmup_runs):
        _ = stage(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        enforce_active_cuda_oom_limit(
            f"after_warmup_{warmup_idx + 1}",
            stage.device,
            active_cuda_oom_limit_mib,
            active_cuda_oom_metric,
        )

    measure_profiles: list[list] = []
    for measure_idx in range(measure_runs):
        _, profiles = stage.forward_with_layer_timings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            include_embed_tokens=True,
        )
        enforce_active_cuda_oom_limit(
            f"after_measure_{measure_idx + 1}",
            stage.device,
            active_cuda_oom_limit_mib,
            active_cuda_oom_metric,
        )
        measure_profiles.append(profiles)
    return measure_profiles


def tensor_to_device(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(device)


def profile_cached_samples(
    stage,
    *,
    timing_labels: list[str],
    sample_context: dict,
    warmup_runs: int,
    measure_runs: int,
    print_cuda_memory: bool = False,
    active_cuda_oom_limit_mib: float | None = None,
    active_cuda_oom_metric: str = "max_reserved",
) -> None:
    per_sample_averages: list[list] = []
    sample_records: list[dict] = []

    for sample_order, cache_index in enumerate(sample_context["selected_indices"]):
        input_ids = sample_context["input_ids"][cache_index].unsqueeze(0).to(stage.device)
        attention_mask = sample_context["attention_mask"][cache_index].unsqueeze(0).to(stage.device)
        if print_cuda_memory:
            print_cuda_mem(f"sample_{sample_order:04d}_cache_{cache_index:04d}_after_tensors_to_device", stage.device)
        enforce_active_cuda_oom_limit(
            f"sample_{sample_order:04d}_cache_{cache_index:04d}_after_tensors_to_device",
            stage.device,
            active_cuda_oom_limit_mib,
            active_cuda_oom_metric,
        )
        measure_timings = measure_single_input(
            stage,
            input_ids=input_ids,
            attention_mask=attention_mask,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
            active_cuda_oom_limit_mib=active_cuda_oom_limit_mib,
            active_cuda_oom_metric=active_cuda_oom_metric,
        )
        if print_cuda_memory:
            print_cuda_mem(f"sample_{sample_order:04d}_cache_{cache_index:04d}_after_profile", stage.device)
        enforce_active_cuda_oom_limit(
            f"sample_{sample_order:04d}_cache_{cache_index:04d}_after_profile",
            stage.device,
            active_cuda_oom_limit_mib,
            active_cuda_oom_metric,
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
