"""Виклики генератора (OpenAI) і суддів (Anthropic або OpenAI) з примусовим JSON."""

from __future__ import annotations

import json
import re

import config

_openai = None
_anthropic = None


def _openai_client():
    global _openai
    if _openai is None:
        from openai import OpenAI

        if not config.OPENAI_API_KEY:
            raise RuntimeError("Не задано OPENAI_API_KEY у файлі .env")
        _openai = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai


def _anthropic_client():
    global _anthropic
    if _anthropic is None:
        import anthropic

        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("Не задано ANTHROPIC_API_KEY у файлі .env")
        _anthropic = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic


def load_prompt(name: str) -> str:
    path = config.PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def parse_json(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Очікувався JSON-об'єкт")
    return data


def build_prompt(question: str, contexts: list[dict] | None) -> str:
    if not contexts:
        return question
    blocks = []
    for i, chunk in enumerate(contexts):
        text = str(chunk.get("text", ""))[: config.MAX_CHUNK_IN_PROMPT]
        source = chunk.get("source", "невідомо")
        blocks.append(f"[{i + 1}] ({source}) {text}")
    template = load_prompt("generate.txt")
    return (
        template.replace("{{CONTEXT}}", "\n\n".join(blocks)).replace("{{QUESTION}}", question)
    )


def generate(question: str, contexts: list[dict] | None = None, temperature: float | None = None) -> str:
    """Відповідь моделі. Без контексту питання йде як є, з контекстом — у промпт із джерелами."""
    prompt = build_prompt(question, contexts)
    temp = config.TEMPERATURE if temperature is None else temperature
    response = _openai_client().chat.completions.create(
        model=config.GENERATOR_MODEL,
        temperature=temp,
        max_tokens=config.MAX_ANSWER_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    generate.last_tokens = prompt_tokens + completion_tokens
    generate.last_cost = config.estimate_cost(
        config.GENERATOR_MODEL, prompt_tokens, completion_tokens
    )
    return response.choices[0].message.content or ""


generate.last_tokens = 0
generate.last_cost = 0.0


def complete_json(prompt: str, *, model: str | None = None, provider: str = "anthropic") -> tuple[dict, float]:
    """Суддя з температурою 0. Повертає розібраний JSON і оцінену вартість виклику."""
    text, cost = _complete_text(prompt, model=model, provider=provider, force_json=True)
    try:
        return parse_json(text), cost
    except (json.JSONDecodeError, ValueError):
        repair = (
            "Попередня відповідь не є валідним JSON. Поверни лише один JSON-об'єкт, "
            "без markdown і без тексту довкола.\n\n"
            f"Попередня відповідь:\n{text}"
        )
        text2, cost2 = _complete_text(repair, model=model, provider=provider, force_json=True)
        return parse_json(text2), cost + cost2


def _complete_text(prompt: str, *, model: str | None, provider: str, force_json: bool) -> tuple[str, float]:
    provider = provider.lower()
    if provider == "anthropic":
        model = model or config.JUDGE_MODEL
        message = _anthropic_client().messages.create(
            model=model,
            max_tokens=1200,
            temperature=0,
            system="Поверни лише один JSON-об'єкт. Жодного тексту поза JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        usage = message.usage
        cost = config.estimate_cost(
            model,
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )
        return "\n".join(parts).strip(), cost

    if provider == "openai":
        model = model or config.JUDGE_MODEL_OPENAI
        kwargs = {
            "model": model,
            "temperature": 0,
            "max_tokens": 1200,
            "messages": [
                {"role": "system", "content": "Поверни лише один JSON-об'єкт."},
                {"role": "user", "content": prompt},
            ],
        }
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = _openai_client().chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            response = _openai_client().chat.completions.create(**kwargs)
        usage = response.usage
        cost = config.estimate_cost(
            model,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
        return (response.choices[0].message.content or "").strip(), cost

    raise ValueError(f"Невідомий провайдер судді: {provider}")


def format_context_block(contexts: list[dict] | None) -> str:
    if not contexts:
        return "(контекст відсутній)"
    return "\n\n".join(
        f"[{i + 1}] ({c.get('source', 'невідомо')}) {c.get('text', '')}"
        for i, c in enumerate(contexts)
    )
