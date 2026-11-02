"""Узгодженість: кілька генерацій із температурою 0.7 і середній попарний ROUGE-L."""

from __future__ import annotations

import config
from evaluators.base import Evaluator, tracked
from evaluators.rouge_eval import RougeEval
from llm import generate


class SelfCheckEval(Evaluator):
    name = "selfcheck"
    llm_judge = False
    criteria = ["agreement"]

    def __init__(self, samples: int | None = None):
        super().__init__()
        self.samples = samples or config.SELFCHECK_SAMPLES

    @tracked
    def score(self, question, contexts, answer, reference=None) -> dict:
        samples = []
        for _ in range(self.samples):
            samples.append(
                generate(question, contexts or None, temperature=config.SELFCHECK_TEMPERATURE)
            )
            self.last_cost += float(getattr(generate, "last_cost", 0.0) or 0.0)
        rouge = RougeEval()
        pair_scores = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                measured = rouge.score(question, contexts, samples[j], reference=samples[i])
                if "rougeL" in measured:
                    pair_scores.append(measured["rougeL"])
        if not pair_scores:
            return {"agreement": 0.0}
        return {"agreement": sum(pair_scores) / len(pair_scores)}
