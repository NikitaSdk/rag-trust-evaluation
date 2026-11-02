"""Масовий прогін: кожне питання × режим, далі кожен метод оцінки.

Після кожного кроку результат уже в SQLite. Повторює виклик у разі помилки
і пропускає те, що вже пораховано.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import config
from db import find_answer, log_error, mark_method_done, method_done, save_answer, save_score, upsert_question
from evaluators import default_evaluators
from pipeline import run


def call_with_retry(action, attempts: int = 3):
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:
            if attempt == attempts:
                raise
            print(f"    спроба {attempt} не вдалася: {error}")
            time.sleep(delay)
            delay *= 2


def load_golden(path=None) -> list[dict]:
    file = path or config.GOLDEN_PATH
    data = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Порожній або зіпсований набір: {file}")
    return data


def ensure_answer(item: dict, mode: str, attempts: int, force: bool) -> tuple[dict, int]:
    if not force:
        existing = find_answer(item["question"], mode)
        if existing:
            upsert_question(item["question"], item.get("reference"), item.get("id"), item.get("topic"))
            print(f"  відповідь вже є, id={existing['id']}")
            return existing, int(existing["id"])

    def generate():
        return run(item["question"], mode)

    result = call_with_retry(generate, attempts)
    answer_id = save_answer(
        result,
        reference=item.get("reference"),
        external_id=item.get("id"),
        topic=item.get("topic"),
        replace=force,
    )
    print(f"  збережено відповідь id={answer_id}, {result['latency']:.1f} с, {result['tokens']} токенів")
    return result, answer_id


def main() -> None:
    config.configure_stdio()
    parser = argparse.ArgumentParser(description="Прогін еталонного набору через RAG і методи оцінки")
    parser.add_argument("--modes", nargs="*", default=list(config.MODES))
    parser.add_argument("--methods", nargs="*", default=None, help="Імена методів, інакше всі")
    parser.add_argument("--limit", type=int, default=0, help="Скільки перших питань узяти")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--judge-repeats", type=int, default=1, help="Повтори для суддів, щоб оцінити розкид")
    parser.add_argument("--force", action="store_true", help="Перегенерувати відповіді і стерти їхні оцінки")
    parser.add_argument("--no-second-judge", action="store_true")
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    golden = load_golden()
    if args.limit:
        golden = golden[: args.limit]
    evaluators = default_evaluators(
        include_second_judge=not args.no_second_judge,
        include_ragas=not args.no_ragas,
        include_selfcheck=args.selfcheck,
    )
    if args.methods:
        wanted = set(args.methods)
        evaluators = [item for item in evaluators if item.name in wanted]
        missing = wanted - {item.name for item in evaluators}
        if missing:
            raise SystemExit(f"Невідомі методи: {', '.join(sorted(missing))}")

    print(f"Питань: {len(golden)}. Режимів: {len(args.modes)}. Методів: {len(evaluators)}.")
    for item in golden:
        print(f"\n[{item.get('id', '?')}] {item['question']}")
        for mode in args.modes:
            print(f" режим {mode}")
            try:
                result, answer_id = ensure_answer(item, mode, args.attempts, args.force)
            except Exception as error:
                log_error(item["question"], mode, "generate", traceback.format_exc())
                print(f"  генерацію пропущено: {error}")
                continue
            contexts = result.get("contexts") or []
            answer = result.get("answer") or ""
            for evaluator in evaluators:
                repeats = args.judge_repeats if evaluator.llm_judge else 1
                for run_index in range(repeats):
                    label = f"{evaluator.name}#{run_index}"
                    if not args.force and method_done(answer_id, evaluator.name, run_index):
                        print(f"    {label}: вже пораховано")
                        continue
                    print(f"    {label}")
                    try:
                        scores = call_with_retry(
                            lambda ev=evaluator: ev.score(
                                item["question"], contexts, answer, item.get("reference")
                            ),
                            args.attempts,
                        )
                    except Exception as error:
                        log_error(item["question"], mode, evaluator.name, traceback.format_exc())
                        print(f"      помилка: {error}")
                        continue
                    count = len(scores) or 1
                    for criterion, value in scores.items():
                        save_score(
                            answer_id,
                            evaluator.name,
                            criterion,
                            float(value),
                            cost=evaluator.last_cost / count,
                            latency=evaluator.last_latency / count,
                            run_index=run_index,
                        )
                    mark_method_done(answer_id, evaluator.name, run_index)
                    print(f"      {scores}")


if __name__ == "__main__":
    main()
