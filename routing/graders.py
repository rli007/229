"""Automatic graders for simple routing datasets."""

from __future__ import annotations

import math
import re


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9.\-+\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_answer_field(output: str) -> str:
    for line in output.splitlines():
        if line.lower().strip().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else output.strip()


def extract_confidence(output: str) -> float | None:
    for line in output.splitlines():
        if line.lower().strip().startswith("confidence:"):
            match = re.search(r"[-+]?\d*\.?\d+", line)
            if not match:
                return None
            value = float(match.group())
            if 1.0 < value <= 100.0:
                value /= 100.0
            return max(0.0, min(1.0, value))
    return None


def grade_answer(prediction: str, gold: str, answer_type: str) -> int:
    pred = extract_answer_field(prediction)
    answer_type = answer_type.lower().strip()
    if answer_type == "number":
        return int(_grade_number(pred, gold))
    if answer_type == "choice":
        return int(_grade_choice(pred, gold))
    if answer_type == "yesno":
        return int(_grade_yesno(pred, gold))
    return int(normalize_text(pred) == normalize_text(gold))


def _grade_number(prediction: str, gold: str) -> bool:
    pred_match = re.findall(r"[-+]?\d*\.?\d+", prediction.replace(",", ""))
    gold_match = re.findall(r"[-+]?\d*\.?\d+", gold.replace(",", ""))
    if not pred_match or not gold_match:
        return normalize_text(prediction) == normalize_text(gold)
    pred_value = float(pred_match[-1])
    gold_value = float(gold_match[-1])
    return math.isclose(pred_value, gold_value, rel_tol=1e-4, abs_tol=1e-4)


def _grade_choice(prediction: str, gold: str) -> bool:
    gold_letter = normalize_text(gold)[:1]
    match = re.search(r"\b([A-D])\b", prediction.upper())
    if match:
        return match.group(1).lower() == gold_letter
    return normalize_text(prediction)[:1] == gold_letter


def _grade_yesno(prediction: str, gold: str) -> bool:
    pred_words = normalize_text(prediction).split()
    gold_label = "yes" if normalize_text(gold).startswith("yes") else "no"
    if "yes" in pred_words:
        return gold_label == "yes"
    if "no" in pred_words:
        return gold_label == "no"
    return False

