"""Єдина точка входу: питання і режим → відповідь і контекст."""

from __future__ import annotations

import time

import config
from llm import generate


def _retrieve(mode: str, question: str) -> list[dict]:
    import retrievers

    search = {
        "local": retrievers.local_search,
        "web": retrievers.web_search,
        "combined": retrievers.combined_search,
    }
    return search[mode](question)


def run(question: str, mode: str) -> dict:
    if mode not in config.MODES:
        raise ValueError(f"Невідомий режим: {mode}. Очікується одне з {config.MODES}")
    started = time.perf_counter()
    contexts = [] if mode == "none" else _retrieve(mode, question)
    answer = generate(question, contexts or None)
    latency = time.perf_counter() - started
    return {
        "question": question,
        "mode": mode,
        "contexts": contexts,
        "answer": answer,
        "latency": latency,
        "tokens": int(getattr(generate, "last_tokens", 0) or 0),
        "cost": float(getattr(generate, "last_cost", 0.0) or 0.0),
    }
