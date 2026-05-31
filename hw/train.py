from __future__ import annotations

import argparse
import math
import random
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hw.constants import IGNORE_INDEX, IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig, MockCausalLM, MockVisionEncoder
from hw.processor import MathVLMProcessor, ProcessorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str | None) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name or "cpu"


class MockTokenizer:
    """Whitespace tokenizer with a fixed vocab for CPU smoke runs."""

    def __init__(self, vocab_size: int = 1024) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.special_tokens = {
            "<pad>": 0,
            "<eos>": 1,
            IMAGE_TOKEN: 2,
            IMAGE_START_TOKEN: 3,
            IMAGE_END_TOKEN: 4,
        }
        self.num_reserved = 5
        self.id_to_special = {idx: token for token, idx in self.special_tokens.items()}

    def _token_id(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.num_reserved + zlib.crc32(token.encode("utf-8")) % (self.vocab_size - self.num_reserved)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._token_id(token) for token in text.split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self.id_to_special:
                if not skip_special_tokens:
                    tokens.append(self.id_to_special[token_id])
        return " ".join(tokens)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_id(token)

    def __len__(self) -> int:
        return self.vocab_size


def build_processor_config(processor_cfg: dict[str, Any]) -> ProcessorConfig:
    return ProcessorConfig(
        image_size=int(processor_cfg.get("image_size", 224)),
        num_tiles=int(processor_cfg.get("num_tiles", 1)),
        tile_overlap=float(processor_cfg.get("tile_overlap", 0.0)),
        num_image_tokens=int(processor_cfg.get("num_image_tokens", 49)),
        max_length=int(processor_cfg.get("max_length", 512)),
        ignore_index=int(processor_cfg.get("ignore_index", IGNORE_INDEX)),
    )


def build_tokenizer() -> MockTokenizer:
    return MockTokenizer()


def build_model(processor_config: ProcessorConfig, tokenizer: MockTokenizer) -> MathVLM:
    vision_hidden_size = 32
    text_hidden_size = 64
    patch_size = max(8, processor_config.image_size // 7)
    vision_encoder = MockVisionEncoder(hidden_size=vision_hidden_size, patch_size=patch_size)
    language_model = MockCausalLM(vocab_size=len(tokenizer), hidden_size=text_hidden_size)
    config = ModelConfig(
        vision_hidden_size=vision_hidden_size,
        text_hidden_size=text_hidden_size,
        num_image_tokens=processor_config.num_image_tokens,
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
    )
    return MathVLM(vision_encoder, language_model, config)


def save_adapter(model: MathVLM, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.adapter.state_dict()
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import save_file

            save_file(state, str(path))
            return
        except Exception:
            path = path.with_suffix(".pt")
    torch.save(state, str(path))


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    output = model(batch)
    loss = output["loss"] if isinstance(output, dict) else output.loss
    if not torch.isfinite(loss):
        raise ValueError("training loss is not finite")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return float(loss.item())


def run_training(config: dict[str, Any], fast_train: bool = False) -> float:
    """Main training entry point."""
    set_seed(int(config.get("seed", 42)))
    model_cfg = config["model"]
    trainer_cfg = config["trainer"]
    data_cfg = config["data"]

    device = resolve_device(trainer_cfg.get("device", "cpu"))
    tokenizer = build_tokenizer()
    processor = MathVLMProcessor(tokenizer, build_processor_config(config["processor"]))
    model = build_model(processor.config, tokenizer).to(device)
    if model_cfg.get("freeze_vision", True) or model_cfg.get("freeze_llm", True):
        model.freeze_backbones()

    dataset = MathVQADataset(
        data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )
    if len(dataset) == 0:
        raise ValueError("training dataset is empty")

    local_batch_size = int(trainer_cfg.get("local_batch_size", 1))
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda samples: processor.collate([processor(s) for s in samples]),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(trainer_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    accum_steps = max(1, int(trainer_cfg.get("global_batch_size", 1)) // max(1, local_batch_size))
    max_steps = int(trainer_cfg.get("max_steps", 0))
    if fast_train:
        max_steps = min(max_steps, 2) if max_steps else 2

    model.train()
    optimizer.zero_grad()
    step = 0
    micro = 0
    running = 0.0
    last_loss = float("nan")
    while step < max_steps:
        steps_before = step
        for raw_batch in loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            loss = model(batch).loss / accum_steps
            if not torch.isfinite(loss):
                raise ValueError("training loss is not finite")
            loss.backward()
            running += float(loss.item()) * accum_steps
            micro += 1
            if micro % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                last_loss = running / accum_steps
                running = 0.0
                print(f"step {step}/{max_steps} loss {last_loss:.4f}")
                if step >= max_steps:
                    break
        if step == steps_before:
            break

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        save_adapter(model, save_path)
    print(f"final_train_loss {last_loss:.4f}")
    return last_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
