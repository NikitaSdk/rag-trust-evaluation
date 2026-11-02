"""Суддя без рубрики: одна оцінка 1–5."""

from __future__ import annotations

import config
from evaluators.base import Evaluator, clamp_likert, fill_prompt, tracked
from llm import complete_json, format_context_block, load_prompt


class JudgePlain(Evaluator):
    name = "judge_plain"
    llm_judge = True
    criteria = ["score"]

    def __init__(self, model: str | None = None, provider: str = "anthropic", name: str | None = None):
        super().__init__()
        self.provider = provider
        self.model = model or (config.JUDGE_MODEL if provider == "anthropic" else config.JUDGE_MODEL_OPENAI)
        self.name = name or ("judge_plain" if provider == "anthropic" else f"judge_plain_{self.model}")

    @tracked
    def score(self, question, contexts, answer, reference=None) -> dict:
        prompt = fill_prompt(
            load_prompt("judge_plain.txt"),
            QUESTION=question,
            CONTEXT=format_context_block(contexts),
            ANSWER=answer or "",
        )
        data, cost = complete_json(prompt, model=self.model, provider=self.provider)
        self.last_cost += cost
        return {"score": clamp_likert(data.get("score"))}
