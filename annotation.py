"""Вивантаження відповідей для ручної розмітки і завантаження заповненого CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import config
from db import load_answers, save_annotation

FIELDS = ["faithfulness", "completeness", "usefulness", "trust", "hallucination"]
HEADER = ["id", "question", "answer", "contexts", *FIELDS, *[f"{field}_2" for field in FIELDS]]


def _contexts_text(contexts: list[dict]) -> str:
    return "\n\n".join(
        f"({chunk.get('source', '')}) {chunk.get('text', '')}" for chunk in contexts or []
    )


def export_for_annotation(path: Path | None = None, n: int = 100, seed: int = 42) -> Path:
    answers = load_answers()
    if len(answers) < n:
        print(f"У базі {len(answers)} відповідей, це менше за {n}. Вивантажую всі.")
    rng = random.Random(seed)
    rng.shuffle(answers)
    chosen = answers[:n] if len(answers) > n else answers
    destination = Path(path or config.ANNOTATION_DIR / "to_label.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for item in chosen:
            writer.writerow(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "contexts": _contexts_text(item["contexts"]),
                    **{field: "" for field in HEADER if field not in {"id", "question", "answer", "contexts"}},
                }
            )
    print(f"Записано {len(chosen)} рядків у {destination}")
    print("Режим і автоматичні оцінки приховані. Другий розмітник заповнює колонки з суфіксом _2.")
    return destination


def _number(value: str):
    text = (value or "").strip().replace(",", ".")
    if not text:
        return None
    return float(text)


def import_annotations(path: Path) -> int:
    saved = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            answer_id = int(row["id"])
            first = {field: _number(row.get(field, "")) for field in FIELDS}
            second = {field: _number(row.get(f"{field}_2", "")) for field in FIELDS}
            if any(value is not None for value in first.values()):
                save_annotation(answer_id, "expert1", first)
                saved += 1
            if any(value is not None for value in second.values()):
                save_annotation(answer_id, "expert2", second)
                saved += 1
    print(f"Імпортовано записів розмітки: {saved}")
    return saved


def main() -> None:
    config.configure_stdio()
    parser = argparse.ArgumentParser(description="Ручна розмітка відповідей")
    parser.add_argument("command", choices=["export", "import"])
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "export":
        export_for_annotation(args.path, args.n, args.seed)
    else:
        if args.path is None:
            raise SystemExit("Для import потрібен --path до заповненого CSV")
        import_annotations(args.path)


if __name__ == "__main__":
    main()
