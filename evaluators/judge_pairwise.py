"""Парне порівняння з еталоном. Два проходи в різному порядку, оцінка усереднюється."""

from __future__ import annotations

import config
from evaluators.base import Evaluator, fill_prompt, tracked
from llm import complete_json, format_context_block, load_prompt


def _winner_score(winner: str, answer_is_a: bool) -> float:
    label = str(winner or "").strip().lower()
    if label in {"tie", "equal", "нічия", "рівні"}:
        return 0.5
    if label in {"a", "відповідь a", "answer a"}:
        return 1.0 if answer_is_a else 0.0
    if label in {"b", "відповідь b", "answer b"}:
        return 0.0 if answer_is_a else 1.0
    return 0.5


class PairwiseEval(Evaluator):
    name = "pairwise"
    needs_reference = True
    llm_judge = True
    criteria = ["win_rate"]

    def __init__(self, model: str | None = None, provider: str = "anthropic", name: str | None = None):
        super().__init__()
        self.provider = provider
        self.model = model or (config.JUDGE_MODEL if provider == "anthropic" else config.JUDGE_MODEL_OPENAI)
        self.name = name or ("pairwise" if provider == "anthropic" else f"pairwise_{self.model}")

    def compare(self, question, contexts, answer_a, answer_b) -> float:
        prompt = fill_prompt(
            load_prompt("judge_pairwise.txt"),
            QUESTION=question,
            CONTEXT=format_context_block(contexts),
            ANSWER_A=answer_a or "",
            ANSWER_B=answer_b or "",
        )
        data, cost = complete_json(prompt, model=self.model, provider=self.provider)
        self.last_cost += cost
        return _winner_score(data.get("winner"), answer_is_a=True)

    @tracked
    def score(self, question, contexts, answer, reference=None) -> dict:
        if not reference:
            return {}
        forward = self.compare(question, contexts, answer, reference)
        swapped = self.compare(question, contexts, reference, answer)
        return {"win_rate": (forward + (1.0 - swapped)) / 2}
