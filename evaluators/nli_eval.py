"""Частка тверджень відповіді, які контекст підтверджує моделлю NLI."""

from __future__ import annotations

import re

import config
from evaluators.base import Evaluator, fill_prompt, tracked
from llm import complete_json, load_prompt

_model = None
_tokenizer = None
_SENTENCE = re.compile(r"(?<=[\.\!\?…])\s+")


def split_claims(answer: str) -> tuple[list[str], float]:
    text = (answer or "").strip()
    if not text:
        return [], 0.0
    try:
        prompt = fill_prompt(load_prompt("claim_split.txt"), ANSWER=text)
        data, cost = complete_json(prompt, provider="openai", model=config.GENERATOR_MODEL)
        claims = data.get("claims") or data.get("statements") or data.get("твердження") or []
        clean = [str(item).strip() for item in claims if str(item).strip()]
        if clean:
            return clean, cost
    except Exception:
        pass
    parts = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    return parts or [text], 0.0


def _nli():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL)
        _model.eval()
        _model._torch = torch
    return _tokenizer, _model


def entailment_probability(premise: str, hypothesis: str) -> float:
    tokenizer, model = _nli()
    torch = model._torch
    encoded = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)
    labels = {index: name.lower() for index, name in model.config.id2label.items()}
    for index, name in labels.items():
        if "entail" in name:
            return float(probs[int(index)])
    return float(probs[-1])


class NliEval(Evaluator):
    name = "nli"
    criteria = ["support_rate", "mean_entailment"]

    @tracked
    def score(self, question, contexts, answer, reference=None) -> dict:
        claims, cost = split_claims(answer)
        self.last_cost += cost
        if not claims:
            return {"support_rate": 0.0, "mean_entailment": 0.0}
        chunks = [c.get("text", "") for c in (contexts or []) if c.get("text")]
        if not chunks:
            return {"support_rate": 0.0, "mean_entailment": 0.0}
        supported = 0
        probabilities = []
        for claim in claims:
            best = max(entailment_probability(chunk, claim) for chunk in chunks)
            probabilities.append(best)
            if best >= config.NLI_ENTAILMENT_THRESHOLD:
                supported += 1
        return {
            "support_rate": supported / len(claims),
            "mean_entailment": sum(probabilities) / len(probabilities),
        }
