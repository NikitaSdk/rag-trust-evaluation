"""Корисність відповіді для рішення: ризики, альтернативи, обмеження, підстави."""

from __future__ import annotations

import config
from evaluators.base import Evaluator, clamp_likert, fill_prompt, tracked
from llm import complete_json, format_context_block, load_prompt


class DecisionEval(Evaluator):
    name = "decision"
    llm_judge = True
    criteria = ["usefulness", "missing_count"]

    def __init__(self, model: str | None = None, provider: str = "anthropic", name: str | None = None):
        super().__init__()
        self.provider = provider
        self.model = model or (config.JUDGE_MODEL if provider == "anthropic" else config.JUDGE_MODEL_OPENAI)
        self.name = name or ("decision" if provider == "anthropic" else f"decision_{self.model}")

    @tracked
    def score(self, question, contexts, answer, reference=None) -> dict:
        prompt = fill_prompt(
            load_prompt("decision_usefulness.txt"),
            QUESTION=question,
            CONTEXT=format_context_block(contexts),
            ANSWER=answer or "",
        )
        data, cost = complete_json(prompt, model=self.model, provider=self.provider)
        self.last_cost += cost
        missing = data.get("missing") or data.get("недостатньо") or []
        if isinstance(missing, str):
            missing = [missing] if missing.strip() else []
        return {
            "usefulness": clamp_likert(data.get("score")),
            "missing_count": float(len(missing)),
        }
