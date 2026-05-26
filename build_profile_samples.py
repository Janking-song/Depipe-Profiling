"""Build fixed-length profiling samples from a local prompt dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from profiling.sample_cache import (
        DEFAULT_SAMPLE_SEED,
        DEFAULT_TARGET_TOKEN_LENGTH,
        build_profile_sample_cache,
        default_profile_sample_path,
    )
else:
    from .sample_cache import (
        DEFAULT_SAMPLE_SEED,
        DEFAULT_TARGET_TOKEN_LENGTH,
        build_profile_sample_cache,
        default_profile_sample_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached fixed-length profiling samples from a local dataset.")
    parser.add_argument("--dataset-path", required=True, help="Local dataset directory. Remote loading is intentionally unsupported.")
    parser.add_argument("--model-family", required=True, choices=["qwen2", "llama"])
    parser.add_argument("--tokenizer-path", required=True, help="Local tokenizer directory for the target model family.")
    parser.add_argument("--output", default=None, help="Output .pt file. Defaults to profiling/profile_samples/{family}_{dataset}_len{N}.pt")
    parser.add_argument("--target-length", type=int, default=DEFAULT_TARGET_TOKEN_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")

    output_path = (
        Path(args.output)
        if args.output is not None
        else default_profile_sample_path(
            model_family=args.model_family,
            dataset_path=args.dataset_path,
            target_length=args.target_length,
        )
    )

    bundle = build_profile_sample_cache(
        dataset_path=args.dataset_path,
        model_family=args.model_family,
        tokenizer_path=args.tokenizer_path,
        output_path=output_path,
        target_length=args.target_length,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    metadata = bundle["metadata"]
    print(f"sample_file: {output_path}")
    print(f"model_family: {metadata['model_family']}")
    print(f"tokenizer_path: {metadata['tokenizer_path']}")
    print(f"dataset_path: {metadata['dataset_path']}")
    print(f"dataset_origin: {metadata['dataset_origin']}")
    print(f"target_length: {metadata['target_length']}")
    print(f"num_samples: {metadata['num_samples']}")
    print(f"seed: {metadata['seed']}")


if __name__ == "__main__":
    main()
