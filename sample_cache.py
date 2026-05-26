"""Utilities for local profiling sample preparation and loading."""

from __future__ import annotations

import logging
import random
import tempfile
from pathlib import Path
from typing import Any

import torch


logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "请继续写下去：人工智能正在改变"
DEFAULT_TARGET_TOKEN_LENGTH = 260
DEFAULT_SAMPLE_SEED = 42
PROMPT_PREVIEW_CHARS = 120
PROFILE_SAMPLE_FORMAT = "depipe_profile_samples_v1"


def default_profile_sample_path(
    *,
    model_family: str,
    dataset_path: str | Path,
    target_length: int,
) -> Path:
    dataset_name = Path(dataset_path).name or "dataset"
    return Path(__file__).resolve().parent / "profile_samples" / f"{model_family}_{dataset_name}_len{target_length}.pt"


def build_profile_sample_cache(
    *,
    dataset_path: str | Path,
    model_family: str,
    tokenizer_path: str | Path,
    output_path: str | Path,
    target_length: int = DEFAULT_TARGET_TOKEN_LENGTH,
    seed: int = DEFAULT_SAMPLE_SEED,
    max_samples: int | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    dataset, dataset_origin = load_local_prompt_dataset(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(Path(tokenizer_path), local_files_only=True)

    input_id_rows: list[list[int]] = []
    attention_mask_rows: list[list[int]] = []
    token_lengths_before_truncation: list[int] = []
    source_indices: list[int] = []
    source_prompt_previews: list[str] = []

    skipped_missing_prompt = 0
    skipped_short = 0

    for example_idx, example in enumerate(dataset):
        prompt = extract_first_user_prompt(example)
        if prompt is None:
            skipped_missing_prompt += 1
            continue

        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded.get("attention_mask") or [1] * len(input_ids))
        original_length = len(input_ids)
        if original_length < target_length:
            skipped_short += 1
            continue

        input_id_rows.append(input_ids[:target_length])
        attention_mask_rows.append(attention_mask[:target_length])
        token_lengths_before_truncation.append(original_length)
        source_indices.append(example_idx)
        source_prompt_previews.append(make_prompt_preview(prompt))

    if not input_id_rows:
        raise ValueError(
            f"No prompts in {dataset_path} produced >= {target_length} tokens with tokenizer {tokenizer_path}. "
            f"skipped_missing_prompt={skipped_missing_prompt}, skipped_short={skipped_short}"
        )

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive when provided, got {max_samples}.")
        if max_samples < len(input_id_rows):
            rng = random.Random(seed)
            selected = list(range(len(input_id_rows)))
            rng.shuffle(selected)
            selected = selected[:max_samples]
            input_id_rows = [input_id_rows[idx] for idx in selected]
            attention_mask_rows = [attention_mask_rows[idx] for idx in selected]
            token_lengths_before_truncation = [token_lengths_before_truncation[idx] for idx in selected]
            source_indices = [source_indices[idx] for idx in selected]
            source_prompt_previews = [source_prompt_previews[idx] for idx in selected]

    bundle = {
        "metadata": {
            "format": PROFILE_SAMPLE_FORMAT,
            "model_family": model_family,
            "tokenizer_path": str(tokenizer_path),
            "dataset_path": str(dataset_path),
            "dataset_origin": dataset_origin,
            "target_length": int(target_length),
            "add_special_tokens": True,
            "num_samples": len(input_id_rows),
            "seed": int(seed),
            "max_samples": None if max_samples is None else int(max_samples),
        },
        "input_ids": torch.tensor(input_id_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_rows, dtype=torch.long),
        "token_length_before_truncation": torch.tensor(token_lengths_before_truncation, dtype=torch.int32),
        "source_index": torch.tensor(source_indices, dtype=torch.int32),
        "source_prompt_preview": list(source_prompt_previews),
    }
    save_profile_sample_cache(bundle, output_path)
    logger.info(
        "Wrote %d profiling sample(s) with target_length=%d to %s.",
        bundle["metadata"]["num_samples"],
        target_length,
        output_path,
    )
    return bundle


def save_profile_sample_cache(bundle: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)


def load_profile_sample_cache(sample_file: str | Path) -> dict[str, Any]:
    sample_file = Path(sample_file)
    if not sample_file.exists():
        raise FileNotFoundError(f"Profile sample file does not exist: {sample_file}")

    bundle = torch.load(sample_file, map_location="cpu")
    if not isinstance(bundle, dict):
        raise TypeError(f"Profile sample file must contain a dict, got {type(bundle).__name__}.")

    metadata = bundle.get("metadata")
    input_ids = bundle.get("input_ids")
    attention_mask = bundle.get("attention_mask")
    token_lengths = bundle.get("token_length_before_truncation")
    source_indices = bundle.get("source_index")
    previews = bundle.get("source_prompt_preview")

    if not isinstance(metadata, dict):
        raise ValueError(f"Profile sample file {sample_file} is missing metadata.")
    if metadata.get("format") != PROFILE_SAMPLE_FORMAT:
        raise ValueError(
            f"Unsupported profile sample format in {sample_file}: {metadata.get('format')!r}. "
            f"Expected {PROFILE_SAMPLE_FORMAT!r}."
        )
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError(f"Profile sample file {sample_file} must contain input_ids shaped [N, L].")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        raise ValueError(f"Profile sample file {sample_file} must contain attention_mask with the same shape as input_ids.")
    if not isinstance(token_lengths, torch.Tensor) or token_lengths.ndim != 1 or token_lengths.shape[0] != input_ids.shape[0]:
        raise ValueError(f"Profile sample file {sample_file} has invalid token_length_before_truncation.")
    if not isinstance(source_indices, torch.Tensor) or source_indices.ndim != 1 or source_indices.shape[0] != input_ids.shape[0]:
        raise ValueError(f"Profile sample file {sample_file} has invalid source_index.")
    if not isinstance(previews, list) or len(previews) != input_ids.shape[0]:
        raise ValueError(f"Profile sample file {sample_file} has invalid source_prompt_preview.")

    target_length = int(metadata.get("target_length", -1))
    if input_ids.shape[1] != target_length:
        raise ValueError(
            f"Profile sample file {sample_file} contains input_ids with length {input_ids.shape[1]}, "
            f"but metadata target_length={target_length}."
        )
    return bundle


def parse_sample_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    indices: list[int] = []
    for part in value.split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            index = int(piece)
        except ValueError as exc:
            raise ValueError(f"Invalid sample index {piece!r}. Use a comma-separated list of integers.") from exc
        if index < 0:
            raise ValueError(f"Sample indices must be non-negative, got {index}.")
        indices.append(index)
    if not indices:
        raise ValueError("sample_indices cannot be empty when provided.")
    return indices


def choose_profile_sample_indices(
    *,
    total_samples: int,
    num_profile_samples: int,
    seed: int,
    explicit_indices: list[int] | None = None,
) -> list[int]:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")
    if explicit_indices is not None:
        validate_profile_sample_indices(explicit_indices, total_samples)
        return list(explicit_indices)
    if num_profile_samples <= 0:
        raise ValueError(f"num_profile_samples must be positive, got {num_profile_samples}.")
    if num_profile_samples > total_samples:
        raise ValueError(
            f"Requested {num_profile_samples} profile samples, but the cache only contains {total_samples} samples."
        )

    rng = random.Random(seed)
    return rng.sample(range(total_samples), k=num_profile_samples)


def validate_profile_sample_indices(indices: list[int], total_samples: int) -> None:
    seen: set[int] = set()
    for index in indices:
        if index < 0 or index >= total_samples:
            raise IndexError(f"Sample index {index} is out of range for a cache with {total_samples} samples.")
        if index in seen:
            raise ValueError(f"Duplicate sample index {index} is not allowed.")
        seen.add(index)


def load_local_prompt_dataset(dataset_path: str | Path):
    from datasets import load_dataset, load_from_disk

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if dataset_path.is_dir():
        try:
            dataset_obj = load_from_disk(str(dataset_path))
        except Exception as exc:
            logger.info("load_from_disk failed for %s: %s", dataset_path, exc)
        else:
            return normalize_loaded_dataset(dataset_obj, dataset_path, origin="load_from_disk")

    raw_files = discover_raw_dataset_files(dataset_path)
    if raw_files:
        loader_name, files = raw_files
        data_files = [str(path) for path in files]
        dataset = load_dataset(
            loader_name,
            data_files=data_files,
            split="train",
            cache_dir=str(Path(tempfile.gettempdir()) / "depipe_datasets_cache"),
        )
        return normalize_loaded_dataset(dataset, dataset_path, origin=f"load_dataset:{loader_name}")

    if dataset_path.is_file():
        visible_files = [dataset_path.name]
    else:
        visible_files = sorted(
            str(path.relative_to(dataset_path))
            for path in dataset_path.rglob("*")
            if path.is_file() and ".cache/huggingface" not in str(path.relative_to(dataset_path))
        )
    visible_summary = ", ".join(visible_files[:10]) if visible_files else "no visible dataset files"
    raise FileNotFoundError(
        f"Could not load a local dataset from {dataset_path}. "
        "Expected a datasets.load_from_disk directory or raw json/jsonl/parquet files. "
        f"Found: {visible_summary}"
    )


def normalize_loaded_dataset(dataset_obj, dataset_path: Path, *, origin: str):
    from datasets import Dataset, DatasetDict

    if isinstance(dataset_obj, Dataset):
        if len(dataset_obj) == 0:
            raise ValueError(f"Dataset at {dataset_path} is empty.")
        return dataset_obj, origin

    if isinstance(dataset_obj, DatasetDict):
        if not dataset_obj:
            raise ValueError(f"Dataset dict at {dataset_path} is empty.")
        split_name = "train" if "train" in dataset_obj else next(iter(dataset_obj))
        dataset = dataset_obj[split_name]
        if len(dataset) == 0:
            raise ValueError(f"Dataset split {split_name!r} at {dataset_path} is empty.")
        return dataset, f"{origin}:{split_name}"

    raise TypeError(f"Unsupported dataset object loaded from {dataset_path}: {type(dataset_obj).__name__}")


def discover_raw_dataset_files(dataset_path: Path) -> tuple[str, list[Path]] | None:
    if dataset_path.is_file():
        suffix = dataset_path.suffix.lower()
        if suffix == ".parquet":
            return "parquet", [dataset_path]
        if suffix in {".json", ".jsonl"}:
            return "json", [dataset_path]
        return None

    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    if parquet_files:
        return "parquet", parquet_files

    jsonl_files = sorted(dataset_path.rglob("*.jsonl"))
    if jsonl_files:
        return "json", jsonl_files

    json_files = [
        path
        for path in sorted(dataset_path.rglob("*.json"))
        if path.name != "dataset_info.json" and path.name != "state.json"
    ]
    if json_files:
        return "json", json_files
    return None


def extract_first_user_prompt(example: dict[str, Any]) -> str | None:
    for field_name in ("conversations", "messages"):
        turns = example.get(field_name)
        prompt = extract_prompt_from_turns(turns)
        if prompt is not None:
            return prompt

    for field_name in ("prompt", "instruction", "question"):
        value = example.get(field_name)
        if isinstance(value, str):
            normalized = normalize_text(value)
            if normalized:
                return normalized
    return None


def extract_prompt_from_turns(turns: Any) -> str | None:
    if not isinstance(turns, list) or not turns:
        return None

    first_turn = turns[0]
    if isinstance(first_turn, str):
        return normalize_text(first_turn)

    if not isinstance(first_turn, dict):
        return None

    role = normalize_role(
        first_turn.get("role")
        or first_turn.get("from")
        or first_turn.get("speaker")
        or first_turn.get("author")
    )
    if role is None or role not in {"user", "human"}:
        return None

    for key in ("content", "value", "text", "prompt", "instruction"):
        value = first_turn.get(key)
        if isinstance(value, str):
            normalized = normalize_text(value)
            if normalized:
                return normalized
    return None


def normalize_role(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def normalize_text(value: str) -> str | None:
    normalized = " ".join(value.strip().split())
    return normalized or None


def make_prompt_preview(prompt: str, *, max_chars: int = PROMPT_PREVIEW_CHARS) -> str:
    if len(prompt) <= max_chars:
        return prompt
    return prompt[: max_chars - 3] + "..."
