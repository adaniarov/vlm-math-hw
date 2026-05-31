from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size]."""
        image = image.convert("RGB")
        size = self.config.image_size
        tiles = []
        for tile in self._split_tiles(image):
            tile = tile.resize((size, size))
            arr = np.asarray(tile, dtype=np.float32) / 255.0
            tiles.append(arr)
        pixel_values = torch.from_numpy(np.stack(tiles, axis=0))
        pixel_values = pixel_values.permute(0, 3, 1, 2).contiguous()
        pixel_values = (pixel_values - 0.5) / 0.5
        return pixel_values

    def _split_tiles(self, image: Image.Image) -> list[Image.Image]:
        n = self.config.num_tiles
        if n <= 1:
            return [image]
        grid = int(round(math.sqrt(n)))
        width, height = image.size
        tile_w = width // grid
        tile_h = height // grid
        tiles = []
        for i in range(grid):
            for j in range(grid):
                left = j * tile_w
                upper = i * tile_h
                tiles.append(image.crop((left, upper, left + tile_w, upper + tile_h)))
        return tiles[:n]

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        num_visual = self.config.num_tiles * self.config.num_image_tokens
        image_tokens = " ".join([IMAGE_TOKEN] * num_visual)
        image_block = f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}"
        options_text = "\n".join(sample.options)
        prompt = (
            f"{image_block}\n"
            "Реши визуально-математическую задачу. Выбери один вариант ответа и напиши только букву.\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            "Ответ:"
        )
        if include_answer:
            prompt = f"{prompt} {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        prompt = self.build_prompt(sample, include_answer=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(f" {sample.answer}", add_special_tokens=False)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            answer_ids = answer_ids + [eos_id]

        input_ids = (prompt_ids + answer_ids)[: self.config.max_length]
        labels = ([self.config.ignore_index] * len(prompt_ids) + answer_ids)[: self.config.max_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def build_generation_inputs(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        prompt = self.build_prompt(sample, include_answer=False)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)[: self.config.max_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "pixel_values": self.preprocess_image(sample.image).unsqueeze(0),
        }

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values."""
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            pad = max_len - item["input_ids"].size(0)
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad,), self.config.ignore_index, dtype=torch.long)]))

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        }
