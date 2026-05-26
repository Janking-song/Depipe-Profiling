"""Common adapter interface for local executable profiling stages."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class LayerRange:
    """Inclusive global Transformer block range."""

    start_layer: int
    end_layer: int

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer + 1

    @property
    def global_layers(self) -> list[int]:
        return list(range(self.start_layer, self.end_layer + 1))


class BaseModelAdapter(ABC):
    """Describes how one HF decoder-only model family is segmented.

    The loader stays generic by asking the adapter for checkpoint prefixes and
    checkpoint->local key remapping. Local layer numbering always starts at 0.
    """

    family: str
    expected_model_types: tuple[str, ...]
    checkpoint_model_prefix = "model"
    embed_name = "embed_tokens"
    layers_name = "layers"
    norm_name = "norm"
    lm_head_name = "lm_head"

    def __init__(self, model_path: Path, *, attn_implementation: str = "eager") -> None:
        self.model_path = Path(model_path)
        self.attn_implementation = attn_implementation
        self.config = self.load_config()
        self.validate_config()

    def load_config(self):
        try:
            from transformers import AutoConfig
        except ImportError as exc:
            raise ImportError("transformers is required. Activate the env that provides the official HF package.") from exc
        return AutoConfig.from_pretrained(self.model_path, local_files_only=True)

    def validate_config(self) -> None:
        model_type = getattr(self.config, "model_type", None)
        if model_type not in self.expected_model_types:
            expected = ", ".join(self.expected_model_types)
            raise ValueError(
                f"{self.family} adapter cannot handle config.model_type={model_type!r}; expected one of: {expected}"
            )

    def total_num_layers(self) -> int:
        return int(self.config.num_hidden_layers)

    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    def validate_layer_range(self, start_layer: int, end_layer: int) -> LayerRange:
        total = self.total_num_layers()
        if start_layer < 0:
            raise ValueError(f"start_layer must be >= 0, got {start_layer}")
        if end_layer < start_layer:
            raise ValueError(f"end_layer must be >= start_layer, got {end_layer} < {start_layer}")
        if end_layer >= total:
            raise ValueError(f"end_layer must be < total_num_layers ({total}), got {end_layer}")
        return LayerRange(start_layer=start_layer, end_layer=end_layer)

    def make_local_config(self, layer_range: LayerRange):
        """Return a config copy whose layers are local to the selected range."""

        config = copy.deepcopy(self.config)
        config.num_hidden_layers = layer_range.num_layers
        config._attn_implementation = self.attn_implementation

        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is not None:
            config.layer_types = list(layer_types[layer_range.start_layer : layer_range.end_layer + 1])
        return config

    def checkpoint_embed_prefixes(self) -> tuple[str, ...]:
        return (f"{self.checkpoint_model_prefix}.{self.embed_name}.",)

    def checkpoint_norm_prefixes(self) -> tuple[str, ...]:
        return (f"{self.checkpoint_model_prefix}.{self.norm_name}.",)

    def checkpoint_lm_head_prefixes(self) -> tuple[str, ...]:
        if self.lm_head_is_tied(self.config):
            return ()
        return (f"{self.lm_head_name}.",)

    def checkpoint_layer_prefix(self, global_layer: int) -> str:
        return f"{self.checkpoint_model_prefix}.{self.layers_name}.{global_layer}."

    def required_prefixes_with_labels(self, layer_range: LayerRange, *, include_lm_head: bool = False) -> list[tuple[str, str]]:
        prefixes: list[tuple[str, str]] = []
        prefixes.extend((self.embed_name, prefix) for prefix in self.checkpoint_embed_prefixes())
        prefixes.extend((self.norm_name, prefix) for prefix in self.checkpoint_norm_prefixes())
        prefixes.extend((f"global_layer_{idx}", self.checkpoint_layer_prefix(idx)) for idx in layer_range.global_layers)
        if include_lm_head:
            prefixes.extend((self.lm_head_name, prefix) for prefix in self.checkpoint_lm_head_prefixes())
        return prefixes

    def required_checkpoint_prefixes(self, layer_range: LayerRange, *, include_lm_head: bool = False) -> tuple[str, ...]:
        return tuple(prefix for _, prefix in self.required_prefixes_with_labels(layer_range, include_lm_head=include_lm_head))

    def map_checkpoint_key_to_local(
        self,
        checkpoint_key: str,
        layer_range: LayerRange,
        *,
        include_lm_head: bool = False,
    ) -> str | None:
        """Map full-model keys like model.layers.8.* to local keys like layers.0.*."""

        for prefix in self.checkpoint_embed_prefixes():
            if checkpoint_key.startswith(prefix):
                suffix = checkpoint_key[len(prefix) :]
                return f"{self.embed_name}.{suffix}"

        for prefix in self.checkpoint_norm_prefixes():
            if checkpoint_key.startswith(prefix):
                suffix = checkpoint_key[len(prefix) :]
                return f"{self.norm_name}.{suffix}"

        if include_lm_head:
            for prefix in self.checkpoint_lm_head_prefixes():
                if checkpoint_key.startswith(prefix):
                    suffix = checkpoint_key[len(prefix) :]
                    return f"{self.lm_head_name}.{suffix}"

        for global_layer in layer_range.global_layers:
            prefix = self.checkpoint_layer_prefix(global_layer)
            if checkpoint_key.startswith(prefix):
                local_layer = global_layer - layer_range.start_layer
                suffix = checkpoint_key[len(prefix) :]
                return f"{self.layers_name}.{local_layer}.{suffix}"

        return None

    def is_ignorable_checkpoint_key(self, checkpoint_key: str, layer_range: LayerRange) -> bool:
        return False

    def build_layers(self, config, layer_range: LayerRange) -> nn.ModuleList:
        return nn.ModuleList([self.build_decoder_layer(config, local_idx) for local_idx in range(layer_range.num_layers)])

    def lm_head_is_tied(self, config) -> bool:
        return bool(getattr(config, "tie_word_embeddings", False))

    def build_lm_head(self, config) -> nn.Module | None:
        if self.lm_head_is_tied(config):
            return None
        return nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward_lm_head(self, hidden_states: torch.Tensor, *, embed_tokens: nn.Module, lm_head: nn.Module | None) -> torch.Tensor:
        if lm_head is None:
            return torch.nn.functional.linear(hidden_states, embed_tokens.weight)
        return lm_head(hidden_states)

    def prepare_attention_mask(
        self,
        *,
        config,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        raise NotImplementedError

    def attention_mask_for_layer(self, decoder_layer: nn.Module, prepared_attention_mask):
        return prepared_attention_mask

    def forward_decoder_layer(
        self,
        decoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        attention_mask,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    @abstractmethod
    def build_embed_tokens(self, config) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def build_decoder_layer(self, config, local_layer_idx: int) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def build_final_norm(self, config) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def build_rotary_embedding(self, config, device: torch.device | str | None = None) -> nn.Module:
        raise NotImplementedError

    @staticmethod
    def prefix_matches(key: str, prefixes: Iterable[str]) -> bool:
        return any(key.startswith(prefix) for prefix in prefixes)
