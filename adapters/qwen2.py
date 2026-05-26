"""Qwen2 adapter backed by the official transformers implementation."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseModelAdapter


class Qwen2Adapter(BaseModelAdapter):
    family = "qwen2"
    expected_model_types = ("qwen2",)

    def build_embed_tokens(self, config) -> nn.Module:
        return nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def build_decoder_layer(self, config, local_layer_idx: int) -> nn.Module:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        return Qwen2DecoderLayer(config, layer_idx=local_layer_idx)

    def build_final_norm(self, config) -> nn.Module:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

        return Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def build_rotary_embedding(self, config, device: torch.device | str | None = None) -> nn.Module:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

        return Qwen2RotaryEmbedding(config=config, device=device)

    def prepare_attention_mask(
        self,
        *,
        config,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        mask_kwargs = {
            "config": config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if "sliding_attention" in getattr(config, "layer_types", []):
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        return masks

    def attention_mask_for_layer(self, decoder_layer: nn.Module, prepared_attention_mask):
        return prepared_attention_mask[decoder_layer.attention_type]
