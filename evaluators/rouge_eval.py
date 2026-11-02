"""ROUGE-1/2/L із токенізацією для української та російської."""

from __future__ import annotations

import re

from evaluators.base import Evaluator, tracked

_TOKEN = re.compile(
    r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+(?:['’ʼ-][0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+)*",
    flags=re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _prepare(text: str) -> str:
    return " ".join(tokenize(text))


class RougeEval(Evaluator):
    name = "rouge"
    needs_reference = True
    criteria = ["rouge1", "rouge2", "rougeL"]

    @tracked
    def score(self, question, contexts, answer, reference=None) -> dict:
        if not reference:
            return {}
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=False,
        )
        raw = scorer.score(_prepare(reference), _prepare(answer))
        return {
            "rouge1": raw["rouge1"].fmeasure,
            "rouge2": raw["rouge2"].fmeasure,
            "rougeL": raw["rougeL"].fmeasure,
        }
