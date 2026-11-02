"""Три джерела контексту: локальна база, вебпошук і їхнє поєднання."""

from __future__ import annotations

import re

import config
from embeddings import cosine, embed_passages, embed_query
from ingest import get_collection


def _clip_k(k: int | None) -> int:
    return config.TOP_K if k is None else k


def _source_label(meta: dict) -> str:
    name = str(meta.get("source", "невідомо"))
    page = meta.get("page")
    if page:
        return f"{name}, с. {page}"
    return name


def local_search(q: str, k: int | None = None) -> list[dict]:
    k = _clip_k(k)
    collection = get_collection()
    count = collection.count()
    if count == 0 or k <= 0:
        return []
    k = min(k, count)
    result = collection.query(
        query_embeddings=[embed_query(q)],
        n_results=k,
        include=["documents", "metadatas"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    found = []
    for text, meta in zip(documents, metadatas):
        meta = meta or {}
        found.append({"text": text or "", "source": _source_label(meta), "origin": "local"})
    return found


def web_search(q: str, k: int | None = None) -> list[dict]:
    k = _clip_k(k)
    if not config.TAVILY_API_KEY:
        raise RuntimeError("Не задано TAVILY_API_KEY у файлі .env")
    from tavily import TavilyClient

    payload = TavilyClient(api_key=config.TAVILY_API_KEY).search(q, max_results=k)
    found = []
    for item in payload.get("results") or []:
        text = (item.get("content") or "").strip()
        if not text:
            continue
        found.append(
            {
                "text": text,
                "source": item.get("url") or item.get("title") or "web",
                "origin": "web",
            }
        )
    return found


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def deduplicate(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for chunk in chunks:
        key = _norm(chunk.get("text", ""))[:240]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def rerank(question: str, chunks: list[dict], k: int) -> list[dict]:
    if not chunks or k <= 0:
        return []
    if len(chunks) == 1:
        return chunks[:k]
    query = embed_query(question)
    vectors = embed_passages([chunk["text"] for chunk in chunks])
    order = sorted(
        range(len(chunks)),
        key=lambda index: cosine(query, vectors[index]),
        reverse=True,
    )
    return [chunks[index] for index in order[:k]]


def combined_search(q: str, k: int | None = None) -> list[dict]:
    k = _clip_k(k)
    local: list[dict] = []
    web: list[dict] = []
    try:
        local = local_search(q, k)
    except Exception as error:
        print(f"Локальний пошук недоступний: {error}")
    try:
        web = web_search(q, k)
    except Exception as error:
        print(f"Вебпошук недоступний: {error}")
    return rerank(q, deduplicate(local + web), k)
