#!/usr/bin/env python3
"""Run local Llama strategies and write raw JSONL for router experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.graders import extract_answer_field, extract_confidence, grade_answer
from routing.prompts import direct_prompt, fewshot_prompt, reasoning_prompt


@dataclass(frozen=True)
class Generation:
    text: str
    elapsed_seconds: float


class TextGenerator(Protocol):
    def generate(self, model: str, prompt: str, temperature: float) -> Generation:
        ...


class OllamaGenerator:
    def __init__(self, host: str, max_tokens: int) -> None:
        self.url = f"{host.rstrip('/')}/api/generate"
        self.max_tokens = max_tokens

    def generate(self, model: str, prompt: str, temperature: float) -> Generation:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": self.max_tokens},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama. Use --backend hf for local Hugging Face "
                "weights, or start Ollama and pull the requested models."
            ) from exc
        return Generation(result.get("response", ""), time.perf_counter() - started)


class HFGenerator:
    def __init__(self, max_tokens: int, tokenizer_overrides: dict[str, str] | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.max_tokens = max_tokens
        self.cache: dict[str, tuple[object, object]] = {}
        self.tokenizer_overrides = tokenizer_overrides or {}

    def _load(self, model: str) -> tuple[object, object]:
        if model not in self.cache:
            tokenizer_source = self.tokenizer_overrides.get(model, model)
            tokenizer = self.AutoTokenizer.from_pretrained(
                tokenizer_source,
                local_files_only=True,
            )
            dtype = self.torch.float16 if self.torch.cuda.is_available() else self.torch.float32
            loaded_model = self.AutoModelForCausalLM.from_pretrained(
                model,
                dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
            loaded_model.to(device)
            loaded_model.eval()
            self.cache[model] = (tokenizer, loaded_model)
        return self.cache[model]

    def generate(self, model: str, prompt: str, temperature: float) -> Generation:
        tokenizer, loaded_model = self._load(model)
        device = next(loaded_model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        started = time.perf_counter()
        with self.torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": self.max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
            output_ids = loaded_model.generate(
                **generation_kwargs,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return Generation(text, time.perf_counter() - started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--cheap-model", required=True)
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--medium-model", default=None)
    parser.add_argument(
        "--cheap-tokenizer",
        default=None,
        help="Optional tokenizer source for the cheap model. Useful when model weights are cached without tokenizer files.",
    )
    parser.add_argument("--strong-tokenizer", default=None)
    parser.add_argument("--medium-tokenizer", default=None)
    parser.add_argument("--include-medium", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cheap-samples", type=int, default=1)
    parser.add_argument("--cheap-cost", type=float, default=1.0)
    parser.add_argument("--medium-cost", type=float, default=3.0)
    parser.add_argument("--strong-cost", type=float, default=8.0)
    return parser.parse_args()


def make_generator(args: argparse.Namespace) -> TextGenerator:
    if args.backend == "ollama":
        return OllamaGenerator(args.ollama_host, args.max_tokens)
    overrides = {}
    if args.cheap_tokenizer:
        overrides[args.cheap_model] = args.cheap_tokenizer
    if args.strong_tokenizer:
        overrides[args.strong_model] = args.strong_tokenizer
    if args.medium_model and args.medium_tokenizer:
        overrides[args.medium_model] = args.medium_tokenizer
    return HFGenerator(args.max_tokens, tokenizer_overrides=overrides)


def strategy_result(
    generator: TextGenerator,
    model: str,
    prompt: str,
    gold: str,
    answer_type: str,
    cost: float,
    temperature: float,
    sample_count: int = 1,
) -> dict[str, object]:
    generation = generator.generate(model, prompt, temperature=temperature)
    samples: list[str] = []
    for _ in range(max(0, sample_count - 1)):
        sample = generator.generate(model, prompt, temperature=max(temperature, 0.7))
        samples.append(extract_answer_field(sample.text))
    output = generation.text
    return {
        "output": output,
        "parsed_answer": extract_answer_field(output),
        "correct": grade_answer(output, gold, answer_type),
        "cost": cost * max(1, sample_count),
        "confidence": extract_confidence(output),
        "samples": samples,
        "elapsed_seconds": generation.elapsed_seconds,
    }


def main() -> None:
    args = parse_args()
    tasks = pd.read_csv(args.tasks)
    if args.limit is not None:
        tasks = tasks.head(args.limit)

    generator = make_generator(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for idx, row in tasks.iterrows():
            question = str(row["prompt"])
            task_type = str(row["task_type"])
            answer_type = str(row["answer_type"])
            gold = str(row["answer"])
            record: dict[str, object] = {
                "example_id": row["example_id"],
                "task_type": task_type,
                "answer_type": answer_type,
                "prompt": question,
                "answer": gold,
            }
            record["cheap_direct"] = strategy_result(
                generator,
                args.cheap_model,
                direct_prompt(question, task_type, answer_type),
                gold,
                answer_type,
                args.cheap_cost,
                args.temperature,
                args.cheap_samples,
            )
            if args.include_medium:
                record["medium_fewshot"] = strategy_result(
                    generator,
                    args.medium_model or args.cheap_model,
                    fewshot_prompt(question, task_type, answer_type),
                    gold,
                    answer_type,
                    args.medium_cost,
                    args.temperature,
                )
            record["strong_reasoning"] = strategy_result(
                generator,
                args.strong_model,
                reasoning_prompt(question, task_type, answer_type),
                gold,
                answer_type,
                args.strong_cost,
                args.temperature,
            )
            f.write(json.dumps(record) + "\n")
            cheap = record["cheap_direct"]["correct"]
            strong = record["strong_reasoning"]["correct"]
            print(f"[{idx + 1}/{len(tasks)}] {row['example_id']}: cheap={cheap} strong={strong}")
    print(f"Wrote raw outputs to {args.output}")


if __name__ == "__main__":
    main()
