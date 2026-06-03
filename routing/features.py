"""Feature extraction for cost-aware escalation experiments."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


UNCERTAINTY_PHRASES = (
    "not sure",
    "i am not sure",
    "i'm not sure",
    "uncertain",
    "maybe",
    "probably",
    "cannot determine",
    "can't determine",
    "insufficient information",
)


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_prompt_features(prompt: str) -> dict[str, Any]:
    words = re.findall(r"\b\w+\b", prompt)
    lower_words = [word.lower() for word in words]
    numbers = re.findall(r"[-+]?\d*\.?\d+", prompt)
    math_symbols = set("=+-*/^<>")
    return {
        "prompt_chars": len(prompt),
        "prompt_words": len(words),
        "question_start": lower_words[0] if lower_words else "unknown",
        "num_numbers": len(numbers),
        "has_math_symbols": int(any(ch in math_symbols for ch in prompt)),
        "has_code_like_text": int(
            "```" in prompt
            or "def " in prompt
            or "class " in prompt
            or "return " in prompt
        ),
        "question_mark_count": prompt.count("?"),
        "capitalized_word_count": len(re.findall(r"\b[A-Z][a-z]+\b", prompt)),
        "contains_parenthesis": int("(" in prompt or ")" in prompt),
        "contains_quote": int('"' in prompt or "'" in prompt),
        "has_comparison_word": int(
            any(
                word in lower_words
                for word in (
                    "more",
                    "less",
                    "than",
                    "same",
                    "different",
                    "outnumber",
                    "larger",
                    "smaller",
                    "before",
                    "after",
                    "longer",
                    "older",
                    "younger",
                    "most",
                )
            )
        ),
        "has_negation_word": int(
            any(
                word in lower_words
                for word in ("not", "no", "never", "without", "lack", "lacks", "doubt")
            )
        ),
        "has_quantity_word": int(
            any(
                word in lower_words
                for word in ("enough", "all", "entire", "each", "every", "only", "many", "few")
            )
        ),
    }


def extract_response_features(
    response: str,
    confidence: float | int | None = None,
    samples: list[str] | None = None,
    parsed_answer: str | None = None,
) -> dict[str, Any]:
    lower = response.lower()
    words = re.findall(r"\b\w+\b", response)
    normalized_answer = normalize_answer(parsed_answer or "")
    if normalized_answer.startswith("yes"):
        cheap_parsed_answer = "yes"
    elif normalized_answer.startswith("no"):
        cheap_parsed_answer = "no"
    else:
        cheap_parsed_answer = "other"
    features: dict[str, Any] = {
        "cheap_answer_chars": len(response),
        "cheap_answer_words": len(words),
        "cheap_parsed_answer": cheap_parsed_answer,
        "cheap_contains_uncertainty": int(
            any(phrase in lower for phrase in UNCERTAINTY_PHRASES)
        ),
        "cheap_self_confidence": float(confidence) if confidence is not None else -1.0,
        "cheap_sample_agreement": sample_agreement(samples),
    }
    return features


def sample_agreement(samples: list[str] | None) -> float:
    if not samples:
        return -1.0
    normalized = [normalize_answer(sample) for sample in samples if sample is not None]
    if not normalized:
        return -1.0
    counts = Counter(normalized)
    return max(counts.values()) / len(normalized)


def build_feature_row(
    record: dict[str, Any],
    cheap_strategy: str,
) -> dict[str, Any]:
    prompt = str(record.get("prompt", ""))
    cheap = record.get(cheap_strategy, {}) or {}
    row: dict[str, Any] = {
        "example_id": record.get("example_id"),
        "task_type": record.get("task_type", "unknown"),
        "answer_type": record.get("answer_type", "unknown"),
        "prompt": prompt,
    }
    row.update(extract_prompt_features(prompt))
    row.update(
        extract_response_features(
            response=str(cheap.get("output", "")),
            confidence=cheap.get("confidence"),
            samples=cheap.get("samples"),
            parsed_answer=cheap.get("parsed_answer"),
        )
    )
    return row
