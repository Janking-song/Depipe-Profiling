"""Llama-family adapter skeleton using official Hugging Face classes."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseModelAdapter, LayerRange


class LlamaAdapter(BaseModelAdapter):
    family = "llama"
    expected_model_types = ("llama",)

    def build_embed_tokens(self, config) -> nn.Module:
        return nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def build_decoder_layer(self, config, local_layer_idx: int) -> nn.Module:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        return LlamaDecoderLayer(config, layer_idx=local_layer_idx)

    def build_final_norm(self, config) -> nn.Module:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        return LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def build_rotary_embedding(self, config, device: torch.device | str | None = None) -> nn.Module:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        return LlamaRotaryEmbedding(config=config, device=device)

    def is_ignorable_checkpoint_key(self, checkpoint_key: str, layer_range: LayerRange) -> bool:
        return checkpoint_key.endswith(".self_attn.rotary_emb.inv_freq")

    def prepare_attention_mask(
        self,
        *,
        config,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        from transformers.masking_utils import create_causal_mask

        return create_causal_mask(
            config=config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
