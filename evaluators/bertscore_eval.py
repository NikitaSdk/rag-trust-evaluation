"""Семантична близькість відповіді до еталона: BERTScore, xlm-roberta-large."""

from __future__ import annotations

import config
from evaluators.base import Evaluator, tracked


class BertScoreEval(Evaluator):
    name = "bertscore"
    needs_reference = True
    criteria = ["precision", "recall", "f1"]

    @tracked
    def score(self, question, contexts, answer, reference=None) -> dict:
        if not reference or not (answer or "").strip():
            return {}
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(
            [answer],
            [reference],
            model_type=config.BERTSCORE_MODEL,
            verbose=False,
        )
        return {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
        }
