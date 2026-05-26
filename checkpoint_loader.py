"""Stage checkpoint materialization and loading."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .adapters.base import BaseModelAdapter, LayerRange


logger = logging.getLogger(__name__)


@dataclass
class CheckpointLoadReport:
    selected_checkpoint_keys: list[str] = field(default_factory=list)
    loaded_local_keys: list[str] = field(default_factory=list)
    shard_to_keys: dict[str, list[str]] = field(default_factory=dict)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    source: str = ""
    stage_checkpoint_path: str | None = None

    @property
    def loaded_key_count(self) -> int:
        return len(self.loaded_local_keys)


class CheckpointSlicer:
    """Materializes a small stage checkpoint from a full safetensors checkpoint."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.weight_map = self._load_weight_map()

    def materialize_segment_checkpoint(
        self,
        *,
        adapter: BaseModelAdapter,
        layer_range: LayerRange,
        expected_local_keys: set[str],
        torch_dtype: torch.dtype | None,
        output_path: str | Path,
        include_lm_head: bool = False,
    ) -> CheckpointLoadReport:
        """Write a small safetensors file containing only selected original keys."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selected = self._select_checkpoint_keys(adapter, layer_range, include_lm_head=include_lm_head)
        grouped = self._group_by_shard(selected)
        stage_checkpoint: dict[str, torch.Tensor] = {}
        loaded_local_keys: list[str] = []
        unexpected_keys: list[str] = []

        logger.info("Materializing %d selected tensor(s) into %s.", len(selected), output_path)
        for shard_name, keys in grouped.items():
            logger.info("Reading shard %s with %d selected tensor(s).", shard_name, len(keys))
            shard_path = self.model_path / shard_name
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for checkpoint_key in keys:
                    local_key = adapter.map_checkpoint_key_to_local(
                        checkpoint_key,
                        layer_range,
                        include_lm_head=include_lm_head,
                    )
                    if local_key is None or local_key not in expected_local_keys:
                        if adapter.is_ignorable_checkpoint_key(checkpoint_key, layer_range):
                            continue
                        unexpected_keys.append(checkpoint_key)
                        continue
                    tensor = handle.get_tensor(checkpoint_key)
                    if torch_dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype=torch_dtype)
                    # The materialized file keeps original checkpoint keys. The
                    # loader remaps them to local stage keys when the stage is built.
                    stage_checkpoint[checkpoint_key] = tensor
                    loaded_local_keys.append(local_key)

        report = CheckpointLoadReport(
            selected_checkpoint_keys=sorted(stage_checkpoint),
            loaded_local_keys=sorted(loaded_local_keys),
            shard_to_keys={name: sorted(keys) for name, keys in grouped.items()},
            missing_keys=sorted(expected_local_keys - set(loaded_local_keys)),
            unexpected_keys=sorted(set(unexpected_keys)),
            source="stage_checkpoint_materialization",
            stage_checkpoint_path=str(output_path),
        )
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(
                "Cannot materialize stage checkpoint because the selected source weights "
                f"do not match the local stage. missing={report.missing_keys}, unexpected={report.unexpected_keys}"
            )

        metadata = {
            "format": "depipe_segment_stage",
            "model_family": adapter.family,
            "source_model_path": str(adapter.model_path),
            "start_layer": str(layer_range.start_layer),
            "end_layer": str(layer_range.end_layer),
            "num_local_layers": str(layer_range.num_layers),
            "torch_dtype": "source" if torch_dtype is None else str(torch_dtype),
            "include_lm_head": str(include_lm_head),
        }
        save_file(stage_checkpoint, output_path, metadata=metadata)
        del stage_checkpoint

        logger.info("Wrote materialized stage checkpoint: %s", output_path)
        return report

    def load_local_stage_state_dict_from_source_checkpoint(
        self,
        *,
        adapter: BaseModelAdapter,
        layer_range: LayerRange,
        expected_local_keys: set[str],
        torch_dtype: torch.dtype | None,
        include_lm_head: bool = False,
    ) -> tuple[dict[str, torch.Tensor], CheckpointLoadReport]:
        """Load selected source tensors and remap them to local stage keys."""

        selected = self._select_checkpoint_keys(adapter, layer_range, include_lm_head=include_lm_head)
        grouped = self._group_by_shard(selected)
        state_dict: dict[str, torch.Tensor] = {}
        unexpected_keys: list[str] = []

        logger.info("Loading %d selected tensor(s) from %d source shard(s).", len(selected), len(grouped))
        for shard_name, keys in grouped.items():
            logger.info("Reading shard %s with %d selected tensor(s).", shard_name, len(keys))
            shard_path = self.model_path / shard_name
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for checkpoint_key in keys:
                    local_key = adapter.map_checkpoint_key_to_local(
                        checkpoint_key,
                        layer_range,
                        include_lm_head=include_lm_head,
                    )
                    if local_key is None or local_key not in expected_local_keys:
                        if adapter.is_ignorable_checkpoint_key(checkpoint_key, layer_range):
                            continue
                        unexpected_keys.append(checkpoint_key)
                        continue
                    tensor = handle.get_tensor(checkpoint_key)
                    if torch_dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype=torch_dtype)
                    state_dict[local_key] = tensor

        report = CheckpointLoadReport(
            selected_checkpoint_keys=sorted(selected),
            loaded_local_keys=sorted(state_dict),
            shard_to_keys={name: sorted(keys) for name, keys in grouped.items()},
            missing_keys=sorted(expected_local_keys - set(state_dict)),
            unexpected_keys=sorted(set(unexpected_keys)),
            source="source_checkpoint_local_stage",
        )
        logger.info("Loaded local key count from source checkpoint: %d.", report.loaded_key_count)
        logger.info("Missing local keys after source load: %s", report.missing_keys)
        logger.info("Unexpected local keys after source load: %s", report.unexpected_keys)
        return state_dict, report

    @staticmethod
    def load_materialized_stage_state_dict(
        *,
        stage_checkpoint_path: str | Path,
        adapter: BaseModelAdapter,
        layer_range: LayerRange,
        expected_local_keys: set[str],
        torch_dtype: torch.dtype,
        include_lm_head: bool = False,
    ) -> tuple[dict[str, torch.Tensor], CheckpointLoadReport]:
        """Load a previously materialized local stage checkpoint."""

        stage_checkpoint_path = Path(stage_checkpoint_path)
        if not stage_checkpoint_path.exists():
            raise FileNotFoundError(f"Stage checkpoint does not exist: {stage_checkpoint_path}")

        state_dict: dict[str, torch.Tensor] = {}
        unexpected_keys: list[str] = []
        with safe_open(stage_checkpoint_path, framework="pt", device="cpu") as handle:
            file_keys = set(handle.keys())
            for checkpoint_key in sorted(file_keys):
                local_key = adapter.map_checkpoint_key_to_local(
                    checkpoint_key,
                    layer_range,
                    include_lm_head=include_lm_head,
                )
                if local_key is None and checkpoint_key in expected_local_keys:
                    local_key = checkpoint_key
                if local_key is None or local_key not in expected_local_keys:
                    if adapter.is_ignorable_checkpoint_key(checkpoint_key, layer_range):
                        continue
                    unexpected_keys.append(checkpoint_key)
                    continue
                tensor = handle.get_tensor(checkpoint_key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=torch_dtype)
                state_dict[local_key] = tensor

        report = CheckpointLoadReport(
            selected_checkpoint_keys=sorted(file_keys),
            loaded_local_keys=sorted(state_dict),
            shard_to_keys={stage_checkpoint_path.name: sorted(file_keys)},
            missing_keys=sorted(expected_local_keys - set(state_dict)),
            unexpected_keys=sorted(set(unexpected_keys)),
            source="materialized_stage_checkpoint",
            stage_checkpoint_path=str(stage_checkpoint_path),
        )
        logger.info("Loaded materialized stage checkpoint: %s", stage_checkpoint_path)
        logger.info("Loaded local key count: %d.", report.loaded_key_count)
        logger.info("Missing keys after stage checkpoint load: %s", report.missing_keys)
        logger.info("Unexpected keys after stage checkpoint load: %s", report.unexpected_keys)
        return state_dict, report

    @staticmethod
    def read_stage_checkpoint_metadata(stage_checkpoint_path: str | Path) -> dict[str, str]:
        stage_checkpoint_path = Path(stage_checkpoint_path)
        with safe_open(stage_checkpoint_path, framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})

    def _load_weight_map(self) -> dict[str, str]:
        index_file = self._find_safetensors_index()
        if index_file is not None:
            with index_file.open("r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"Invalid safetensors index: {index_file}")
            return {key: str(shard) for key, shard in weight_map.items()}

        safetensors_files = sorted(self.model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(
                f"No safetensors checkpoint found in {self.model_path}. "
                "The stage checkpoint materializer currently expects .safetensors weights."
            )

        weight_map: dict[str, str] = {}
        for shard_path in safetensors_files:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    weight_map[key] = shard_path.name
        return weight_map

    def _find_safetensors_index(self) -> Path | None:
        preferred = self.model_path / "model.safetensors.index.json"
        if preferred.exists():
            return preferred
        matches = sorted(self.model_path.glob("*.safetensors.index.json"))
        return matches[0] if matches else None

    def _select_checkpoint_keys(self, adapter: BaseModelAdapter, layer_range: LayerRange, *, include_lm_head: bool = False) -> dict[str, str]:
        labeled_prefixes = adapter.required_prefixes_with_labels(layer_range, include_lm_head=include_lm_head)
        for label, prefix in labeled_prefixes:
            if not any(key.startswith(prefix) for key in self.weight_map):
                raise KeyError(f"Checkpoint is missing required {label} prefix: {prefix}")

        prefixes = tuple(prefix for _, prefix in labeled_prefixes)
        return {key: shard for key, shard in self.weight_map.items() if adapter.prefix_matches(key, prefixes)}

    @staticmethod
    def _group_by_shard(selected: dict[str, str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for key, shard in selected.items():
            grouped[shard].append(key)
        return dict(sorted(grouped.items()))
