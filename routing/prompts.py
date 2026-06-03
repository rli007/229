"""Prompt templates for local Llama escalation experiments."""

from __future__ import annotations


def direct_prompt(question: str, task_type: str, answer_type: str) -> str:
    return f"""You are answering a {task_type} question.
Give only the final answer and a confidence score from 0 to 1.

Question:
{question}

{_format_hint(answer_type)}

Use exactly this format:
answer: <final answer>
confidence: <number between 0 and 1>
"""


def brief_reasoning_prompt(question: str, task_type: str, answer_type: str) -> str:
    return f"""You are answering a {task_type} question.
Give one short sentence of reasoning, then the final answer and a confidence score from 0 to 1.

Question:
{question}

{_format_hint(answer_type)}

Use exactly this format:
reasoning: <one short sentence>
answer: <final answer>
confidence: <number between 0 and 1>
"""


def full_reasoning_prompt(question: str, task_type: str, answer_type: str) -> str:
    return f"""You are answering a {task_type} question.
Reason step by step. Then give the final answer and a confidence score from 0 to 1.

Question:
{question}

{_format_hint(answer_type)}

Use exactly this format:
reasoning: <step-by-step reasoning>
answer: <final answer>
confidence: <number between 0 and 1>
"""


def reasoning_prompt(question: str, task_type: str, answer_type: str) -> str:
    return f"""You are answering a {task_type} question.
Reason carefully but keep the explanation short.

Question:
{question}

{_format_hint(answer_type)}

Use exactly this format:
reasoning: <brief reasoning>
answer: <final answer>
"""


def fewshot_prompt(question: str, task_type: str, answer_type: str) -> str:
    return f"""You are answering a {task_type} question.
Follow the examples and output the requested format.

Example:
Question: What is 7 plus 5?
answer: 12
confidence: 0.95

Example:
Question: Can a fish live on dry land for a long time? Answer yes or no.
answer: no
confidence: 0.90

Question:
{question}

{_format_hint(answer_type)}

Use exactly this format:
answer: <final answer>
confidence: <number between 0 and 1>
"""


def _format_hint(answer_type: str) -> str:
    answer_type = answer_type.lower().strip()
    if answer_type == "number":
        return "The final answer should be a number."
    if answer_type == "choice":
        return "The final answer should be one letter: A, B, C, or D."
    if answer_type == "yesno":
        return "The final answer should be yes or no."
    return "The final answer should be concise."
