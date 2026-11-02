"""Ембеддинги multilingual-e5: passage для документів, query для питання."""

from __future__ import annotations

import config

_function = None


def get_embedding_function():
    global _function
    if _function is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        _function = SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
    return _function


def _l2(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _embed(texts: list[str]) -> list[list[float]]:
    raw = get_embedding_function()(texts)
    return [_l2(list(map(float, vector))) for vector in raw]


def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [text if text.startswith("passage: ") else f"passage: {text}" for text in texts]
    return _embed(prefixed)


def embed_query(question: str) -> list[float]:
    text = question if question.startswith("query: ") else f"query: {question}"
    return _embed([text])[0]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
