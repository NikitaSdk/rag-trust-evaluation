"""Спільний інтерфейс методів оцінки."""

from __future__ import annotations

import time
from typing import Callable


class Evaluator:
    name = "base"
    needs_reference = False
    llm_judge = False
    criteria: list[str] = []

    def __init__(self):
        self.last_cost = 0.0
        self.last_latency = 0.0

    def score(self, question: str, contexts: list[dict] | None, answer: str, reference: str | None = None) -> dict:
        raise NotImplementedError


def tracked(method: Callable) -> Callable:
    def wrapper(self, *args, **kwargs):
        self.last_cost = 0.0
        started = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.last_latency = time.perf_counter() - started

    return wrapper


def clamp_likert(value) -> float:
    score = float(value)
    if score > 5 and score <= 100:
        score = 1 + (score / 100) * 4
    return max(1.0, min(5.0, score))


def fill_prompt(template: str, **kwargs: str) -> str:
    text = template
    for key, value in kwargs.items():
        text = text.replace("{{" + key + "}}", value if value is not None else "")
    return text
