"""Методи автоматичної оцінки. Експеримент іде циклом по default_evaluators()."""

from __future__ import annotations

import config
from evaluators.bertscore_eval import BertScoreEval
from evaluators.decision_eval import DecisionEval
from evaluators.judge_pairwise import PairwiseEval
from evaluators.judge_plain import JudgePlain
from evaluators.judge_rubric import JudgeRubric
from evaluators.nli_eval import NliEval
from evaluators.ragas_eval import RagasEval
from evaluators.rouge_eval import RougeEval
from evaluators.selfcheck_eval import SelfCheckEval


def default_evaluators(
    include_second_judge: bool = True,
    include_ragas: bool = True,
    include_selfcheck: bool = False,
) -> list:
    methods = [
        RougeEval(),
        BertScoreEval(),
        NliEval(),
        JudgePlain(),
        JudgeRubric(),
        DecisionEval(),
        PairwiseEval(),
    ]
    if include_second_judge:
        methods.append(
            JudgeRubric(
                model=config.JUDGE_MODEL_OPENAI,
                provider="openai",
                name="judge_rubric_gpt4o",
            )
        )
        methods.append(
            JudgePlain(
                model=config.JUDGE_MODEL_OPENAI,
                provider="openai",
                name="judge_plain_gpt4o",
            )
        )
    if include_ragas:
        methods.append(RagasEval())
    if include_selfcheck:
        methods.append(SelfCheckEval())
    return methods
