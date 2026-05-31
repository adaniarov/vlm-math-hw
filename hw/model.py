from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hw.constants import IGNORE_INDEX


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        self.norm = nn.LayerNorm(vision_hidden_size)
        self.pool = nn.AdaptiveAvgPool1d(num_image_tokens)
        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        x = self.norm(vision_hidden_states)
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(x)


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    merged = input_embeds.clone()
    mask = input_ids == image_token_id
    merged[mask] = visual_embeds.reshape(-1, visual_embeds.size(-1)).to(merged.dtype)
    return merged


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def _build_inputs_embeds(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        batch_size, num_tiles = pixel_values.shape[:2]

        vision_out = self.vision_encoder(pixel_values.flatten(0, 1))
        hidden_states = vision_out.last_hidden_state
        visual_embeds = self.adapter(hidden_states)
        visual_embeds = visual_embeds.reshape(
            batch_size, num_tiles * self.config.num_image_tokens, self.config.text_hidden_size
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        return merge_visual_embeddings(inputs_embeds, input_ids, visual_embeds, self.config.image_token_id)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss."""
        inputs_embeds = self._build_inputs_embeds(batch)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        inputs_embeds = self._build_inputs_embeds(batch)
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            **generation_kwargs,
        )


class MockVisionEncoder(nn.Module):
    """Tiny patch-embedding vision encoder for CPU smoke runs."""

    def __init__(self, hidden_size: int = 32, patch_size: int = 32) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> Any:
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=x)


class MockCausalLM(nn.Module):
    """Tiny single-layer causal LM for CPU smoke runs."""

    def __init__(self, vocab_size: int, hidden_size: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Any:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        seq_len = inputs_embeds.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=inputs_embeds.device),
            diagonal=1,
        )
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        attended, _ = self.attn(
            inputs_embeds,
            inputs_embeds,
            inputs_embeds,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.norm1(inputs_embeds + attended)
        hidden = self.norm2(hidden + self.ffn(hidden))
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
        return SimpleNamespace(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        eos_token_id: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current = inputs_embeds
        mask = attention_mask
        generated = []
        for _ in range(max_new_tokens):
            out = self.forward(inputs_embeds=current, attention_mask=mask)
            next_token = out.logits[:, -1, :].argmax(dim=-1)
            generated.append(next_token)
            next_embed = self.embed_tokens(next_token).unsqueeze(1)
            current = torch.cat([current, next_embed], dim=1)
            if mask is not None:
                ones = torch.ones((mask.size(0), 1), dtype=mask.dtype, device=mask.device)
                mask = torch.cat([mask, ones], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return torch.stack(generated, dim=1)
