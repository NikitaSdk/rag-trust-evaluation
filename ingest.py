"""Документи з data/docs → чанки → Chroma з метаданими джерела."""

from __future__ import annotations

import re
from pathlib import Path

import config
from embeddings import embed_passages

_client = None


def get_client():
    global _client
    if _client is None:
        import chromadb

        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection():
    return get_client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def read_document(path: Path) -> list[tuple[int | None, str]]:
    """Список (сторінка або None, текст). PDF ріжеться посторінково."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append((index, page.extract_text() or ""))
        return pages
    if suffix == ".docx":
        import docx

        document = docx.Document(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return [(None, text)]
    if suffix in {".txt", ".md"}:
        raw = path.read_bytes()
        text = None
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        return [(None, text if text is not None else raw.decode("utf-8", errors="replace"))]
    raise ValueError(f"Непідтримуваний тип файлу: {path.name}")


def _splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def _safe_id(filename: str, page: int | None, index: int) -> str:
    stem = re.sub(r"\s+", "_", filename)
    page_part = "p" if page is None else f"p{page}"
    return f"{stem}-{page_part}-{index}"


def ingest_file(path: Path) -> int:
    path = Path(path)
    pages = read_document(path)
    splitter = _splitter()
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    for page, text in pages:
        chunks = [chunk.strip() for chunk in splitter.split_text(text or "") if chunk.strip()]
        for index, chunk in enumerate(chunks):
            meta = {"source": path.name}
            if page is not None:
                meta["page"] = int(page)
            documents.append(chunk)
            metadatas.append(meta)
            ids.append(_safe_id(path.name, page, index))
    collection = get_collection()
    _delete_source(collection, path.name)
    if not documents:
        return 0
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embed_passages(documents),
    )
    return len(documents)


def _delete_source(collection, filename: str) -> None:
    existing = collection.get(where={"source": filename})
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def ingest_dir(path: Path | None = None) -> int:
    folder = Path(path or config.DOCS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    total = 0
    files = [
        item
        for item in sorted(folder.iterdir())
        if item.is_file() and item.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
    ]
    if not files:
        raise FileNotFoundError(f"У папці {folder} немає pdf, docx або txt")
    for item in files:
        count = ingest_file(item)
        print(f"{item.name}: {count} чанків")
        total += count
    return total


if __name__ == "__main__":
    config.configure_stdio()
    print(f"Записано чанків: {ingest_dir()}")
