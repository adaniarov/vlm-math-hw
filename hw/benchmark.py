from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output."""
    if not text:
        return None
    upper = text.upper()
    allowed = "".join(choices)
    patterns = [
        rf"ОТВЕТ[:\s]+([{allowed}])\b",
        rf"ANSWER\s*(?:IS)?[:\s]+([{allowed}])\b",
        rf"\(([{allowed}])\)",
        rf"^\s*([{allowed}])\b",
        rf"\b([{allowed}])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop."""
    import torch

    from hw.dataset import MathVQADataset
    from hw.processor import MathVLMProcessor
    from hw.train import build_model, build_processor_config, build_tokenizer, resolve_device

    model_cfg = config["model"]
    data_cfg = config["data"]
    inference_cfg = config.get("inference", {})

    device = resolve_device(inference_cfg.get("device", "cpu"))
    tokenizer = build_tokenizer()
    processor = MathVLMProcessor(tokenizer, build_processor_config(config["processor"]))
    model = build_model(processor.config, tokenizer).to(device)
    model.eval()

    adapter_path = model_cfg.get("adapter_path")
    if adapter_path and Path(adapter_path).exists():
        model.adapter.load_state_dict(torch.load(adapter_path, map_location=device))

    if toy:
        manifest = "assets/toy_math_vqa/manifest.jsonl"
        split = "dev"
    else:
        manifest = data_cfg["eval_manifest"]
        split = data_cfg.get("split", "dev")
    dataset = MathVQADataset(manifest, split=split, max_samples=data_cfg.get("max_samples"))

    max_new_tokens = int(inference_cfg.get("max_new_tokens", 16))
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        inputs = processor.build_generation_inputs(sample)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generated = model.generate(inputs, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        rows.append(
            {
                "id": sample.id,
                "prediction": parse_mc_answer(output_text),
                "answer": sample.answer,
                "subject": sample.subject,
                "output": output_text,
            }
        )

    output_path = inference_cfg.get("output_path")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(output_path).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
