"""Обгортка RAGAS. Підтримує і evaluate(), і метрики collections з ascore()."""

from __future__ import annotations

import asyncio
import inspect

import config
from evaluators.base import Evaluator, tracked


def _score_value(result) -> float:
    if hasattr(result, "value"):
        return float(result.value)
    return float(result)


async def _ascore(metric, **candidates):
    signature = inspect.signature(metric.ascore)
    accepted = set(signature.parameters)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        payload = candidates
    else:
        payload = {key: value for key, value in candidates.items() if key in accepted}
    return _score_value(await metric.ascore(**payload))


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("RAGAS викликано з уже запущеного event loop")


def _build_llm():
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    try:
        from ragas.llms import llm_factory

        try:
            return llm_factory(config.GENERATOR_MODEL, provider="openai", client=client)
        except TypeError:
            return llm_factory(config.GENERATOR_MODEL, client=client)
    except Exception:
        pass
    try:
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        return LangchainLLMWrapper(ChatOpenAI(model=config.GENERATOR_MODEL, temperature=0))
    except Exception as error:
        raise RuntimeError(f"Не вдалося зібрати LLM для RAGAS: {error}") from error


def _metrics_v04(llm):
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


def _score_v04(question, contexts, answer, reference) -> dict:
    llm = _build_llm()
    metrics = _metrics_v04(llm)
    texts = [chunk["text"] for chunk in contexts]
    common = {
        "user_input": question,
        "question": question,
        "response": answer,
        "answer": answer,
        "retrieved_contexts": texts,
        "contexts": texts,
        "reference": reference or "",
        "ground_truth": reference or "",
    }
    scores = {}
    for name, metric in metrics.items():
        if name == "context_recall" and not reference:
            continue
        if name == "context_precision" and not reference:
            continue
        scores[name] = _run(_ascore(metric, **common))
    return scores


def _score_legacy(question, contexts, answer, reference) -> dict:
    from datasets import Dataset
    from ragas import evaluate

    row = {
        "question": question,
        "answer": answer,
        "contexts": [chunk["text"] for chunk in contexts],
    }
    metrics = []
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError:
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

        faithfulness = Faithfulness()
        answer_relevancy = AnswerRelevancy()
        context_precision = ContextPrecision()
        context_recall = ContextRecall()
    metrics.extend([faithfulness, answer_relevancy, context_precision])
    if reference:
        row["ground_truth"] = reference
        metrics.append(context_recall)
    result = evaluate(Dataset.from_dict({key: [value] for key, value in row.items()}), metrics=metrics)
    data = result.to_pandas().iloc[0].to_dict() if hasattr(result, "to_pandas") else dict(result)
    scores = {}
    for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if key in data and data[key] is not None:
            scores[key] = float(data[key])
    return scores


class RagasEval(Evaluator):
    name = "ragas"
    criteria = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    @tracked
    def score(self, question, contexts, answer, reference=None) -> dict:
        if not contexts or not (answer or "").strip():
            return {}
        if not config.OPENAI_API_KEY:
            raise RuntimeError("Для RAGAS потрібен OPENAI_API_KEY")
        try:
            scores = _score_v04(question, contexts, answer, reference)
        except Exception:
            scores = _score_legacy(question, contexts, answer, reference)
        return {key: value for key, value in scores.items() if value == value}
