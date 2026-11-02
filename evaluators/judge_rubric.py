"""Суддя за рубрикою: окремий виклик на кожен критерій."""

from __future__ import annotations

import config
from evaluators.base import Evaluator, clamp_likert, fill_prompt, tracked
from llm import complete_json, format_context_block, load_prompt

RUBRIC_FILES = {
    "faithfulness": "judge_rubric_faithfulness.txt",
    "completeness": "judge_rubric_completeness.txt",
    "source_alignment": "judge_rubric_source_alignment.txt",
    "decision_usefulness": "judge_rubric_decision_usefulness.txt",
}


class JudgeRubric(Evaluator):
    name = "judge_rubric"
    llm_judge = True
    criteria = list(RUBRIC_FILES)

    def __init__(self, model: str | None = None, provider: str = "anthropic", name: str | None = None):
        super().__init__()
        self.provider = provider
        self.model = model or (config.JUDGE_MODEL if provider == "anthropic" else config.JUDGE_MODEL_OPENAI)
        self.name = name or ("judge_rubric" if provider == "anthropic" else f"judge_rubric_{self.model}")

    @tracked
    def score(self, question, contexts, answer, reference=None) -> dict:
        scores = {}
        context = format_context_block(contexts)
        for criterion, filename in RUBRIC_FILES.items():
            prompt = fill_prompt(
                load_prompt(filename),
                QUESTION=question,
                CONTEXT=context,
                ANSWER=answer or "",
                REFERENCE=reference or "(еталон не надано)",
            )
            data, cost = complete_json(prompt, model=self.model, provider=self.provider)
            self.last_cost += cost
            scores[criterion] = clamp_likert(data.get("score"))
        return scores
