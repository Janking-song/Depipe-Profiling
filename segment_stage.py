"""Local executable stage used for layer-interval profiling."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

import torch
from torch import nn

from .adapters.base import BaseModelAdapter, LayerRange


BYTES_PER_MIB = 1024.0 * 1024.0


@dataclass(frozen=True)
class LayerProfile:
    elapsed_ms: float
    parameter_memory_mib: float
    peak_memory_mib: float


class SegmentStage(nn.Module):
    """Runs prompt-derived input_ids through embedding, selected blocks, and norm.

    Embedding and final norm are retained so the stage can execute directly from
    tokenizer output. Embedding can be included in the profiling table. When
    include_lm_head is enabled, final norm and lm_head are also profiled so this
    stage behaves like a full causal LM forward.
    """

    def __init__(self, *, adapter: BaseModelAdapter, config, layer_range: LayerRange, include_lm_head: bool = False) -> None:
        super().__init__()
        self.adapter = adapter
        self.config = config
        self.model_family = adapter.family
        self.model_path = Path(adapter.model_path)
        self.layer_range = layer_range
        self.layer_global_indices = layer_range.global_layers
        self.include_lm_head = include_lm_head

        self.embed_tokens = adapter.build_embed_tokens(config)
        self.layers = adapter.build_layers(config, layer_range)
        self.norm = adapter.build_final_norm(config)
        self.lm_head = adapter.build_lm_head(config) if include_lm_head else None
        self.rotary_emb = adapter.build_rotary_embedding(config)

    @property
    def num_profiled_layers(self) -> int:
        return len(self.layers)

    @property
    def layer_mapping(self) -> list[tuple[int, int]]:
        return list(enumerate(self.layer_global_indices))

    def timing_labels(self, *, include_embed_tokens: bool = False) -> list[str]:
        labels = ["embed_tokens"] if include_embed_tokens else []
        labels.extend(
            f"local_layer_{local_idx:02d} / global_layer_{global_idx:02d}"
            for local_idx, global_idx in self.layer_mapping
        )
        if self.include_lm_head:
            labels.extend(("norm", "lm_head"))
        return labels

    def parameter_memory_mib_by_label(self, *, include_embed_tokens: bool = False) -> list[float]:
        values = [self._module_parameter_memory_mib(self.embed_tokens)] if include_embed_tokens else []
        values.extend(self._module_parameter_memory_mib(decoder_layer) for decoder_layer in self.layers)
        if self.include_lm_head:
            values.append(self._module_parameter_memory_mib(self.norm))
            values.append(self._module_parameter_memory_mib(self.lm_head))
        return values

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def load_tokenizer(self, **kwargs):
        from transformers import AutoTokenizer

        kwargs.setdefault("local_files_only", True)
        return AutoTokenizer.from_pretrained(self.model_path, **kwargs)

    def reset_runtime_modules(self, device: torch.device | str | None = None) -> None:
        """Rebuild non-persistent runtime modules after meta-device construction."""

        self.rotary_emb = self.adapter.build_rotary_embedding(self.config, device=device)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        hidden_states, attention_mask, position_ids, cache_position, position_embeddings = self._prepare_forward_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )

        prepared_attention_mask = self.adapter.prepare_attention_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        for decoder_layer in self.layers:
            layer_mask = self.adapter.attention_mask_for_layer(decoder_layer, prepared_attention_mask)
            hidden_states = self.adapter.forward_decoder_layer(
                decoder_layer,
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.norm(hidden_states)
        if self.include_lm_head:
            return self.adapter.forward_lm_head(hidden_states, embed_tokens=self.embed_tokens, lm_head=self.lm_head)
        return hidden_states

    def forward_with_layer_timings(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        include_embed_tokens: bool = False,
    ) -> tuple[torch.Tensor, list[LayerProfile]]:
        profiles: list[LayerProfile] = []
        hidden_states, attention_mask, position_ids, cache_position, position_embeddings = self._prepare_forward_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            profiles=profiles if include_embed_tokens else None,
        )
        prepared_attention_mask = self.adapter.prepare_attention_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
        )

        for decoder_layer in self.layers:
            layer_mask = self.adapter.attention_mask_for_layer(decoder_layer, prepared_attention_mask)
            hidden_states, profile = self._profile_operation(
                hidden_states.device,
                lambda decoder_layer=decoder_layer, hidden_states=hidden_states, layer_mask=layer_mask: self.adapter.forward_decoder_layer(
                    decoder_layer,
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                ),
                parameter_memory_mib=self._module_parameter_memory_mib(decoder_layer),
            )
            profiles.append(profile)

        if self.include_lm_head:
            hidden_states, profile = self._profile_operation(
                hidden_states.device,
                lambda: self.norm(hidden_states),
                parameter_memory_mib=self._module_parameter_memory_mib(self.norm),
            )
            profiles.append(profile)

            logits, profile = self._profile_operation(
                hidden_states.device,
                lambda: self.adapter.forward_lm_head(hidden_states, embed_tokens=self.embed_tokens, lm_head=self.lm_head),
                parameter_memory_mib=self._module_parameter_memory_mib(self.lm_head),
            )
            profiles.append(profile)
            return logits, profiles

        return self.norm(hidden_states), profiles

    def _prepare_forward_inputs(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        use_cache: bool,
        profiles: list[LayerProfile] | None = None,
    ):
        if use_cache:
            raise ValueError("SegmentStage profiling uses prefill forward only; pass use_cache=False.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if profiles is not None and inputs_embeds is not None:
            raise ValueError("embed_tokens timing requires input_ids, not precomputed inputs_embeds.")

        if inputs_embeds is None:
            if profiles is None:
                inputs_embeds = self.embed_tokens(input_ids)
            else:
                inputs_embeds, profile = self._profile_operation(
                    input_ids.device,
                    lambda: self.embed_tokens(input_ids),
                    parameter_memory_mib=self._module_parameter_memory_mib(self.embed_tokens),
                )
                profiles.append(profile)
        hidden_states = inputs_embeds

        cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        else:
            position_ids = position_ids.to(device=hidden_states.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=hidden_states.device)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return hidden_states, attention_mask, position_ids, cache_position, position_embeddings

    @staticmethod
    def _sync_if_cuda(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _profile_operation(self, device: torch.device, operation, *, parameter_memory_mib: float = 0.0) -> tuple[torch.Tensor, LayerProfile]:
        self._sync_if_cuda(device)
        baseline_bytes = self._cuda_memory_allocated(device)
        self._reset_cuda_peak_if_available(device)
        start = time.perf_counter()
        result = operation()
        result_device = result.device if isinstance(result, torch.Tensor) else device
        self._sync_if_cuda(result_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        peak_memory_mib = self._cuda_peak_delta_mib(result_device, baseline_bytes)
        return result, LayerProfile(
            elapsed_ms=elapsed_ms,
            parameter_memory_mib=parameter_memory_mib,
            peak_memory_mib=peak_memory_mib,
        )

    @staticmethod
    def _module_parameter_memory_mib(module: nn.Module | None) -> float:
        if module is None:
            return 0.0
        seen: set[tuple[str, int]] = set()
        total_bytes = 0
        for parameter in module.parameters(recurse=True):
            if parameter.is_meta:
                key = ("meta", id(parameter))
            else:
                key = (str(parameter.device), parameter.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            total_bytes += parameter.numel() * parameter.element_size()
        return total_bytes / BYTES_PER_MIB

    @staticmethod
    def _cuda_memory_allocated(device: torch.device) -> int:
        if device.type != "cuda":
            return 0
        return torch.cuda.memory_allocated(device)

    @staticmethod
    def _reset_cuda_peak_if_available(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def _cuda_peak_delta_mib(device: torch.device, baseline_bytes: int) -> float:
        if device.type != "cuda":
            return 0.0
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return max(0.0, (peak_bytes - baseline_bytes) / BYTES_PER_MIB)
