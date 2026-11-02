"""Контрольні приклади: добра відповідь, вигаданий факт, відповідь не по питанню.

Якщо метод ставить їм однакові оцінки за основним критерієм, він зламаний.
Запуск: python -m evaluators.sanity --method rouge
"""

from __future__ import annotations

import argparse

from evaluators import default_evaluators

CONTEXT = [
    {
        "text": (
            "Спрощена закупівля без відкритих торгів дозволена, якщо очікувана вартість "
            "не перевищує 200 000 грн. Для неї потрібно щонайменше три комерційні пропозиції. "
            "Договір із постачальником, який має податковий борг, укладати заборонено."
        ),
        "source": "polityka_zakupivel.txt",
        "origin": "local",
    }
]

QUESTION = "Чи можна провести закупівлю на 150 000 грн без відкритих торгів і що для цього потрібно?"
REFERENCE = (
    "Так, 150 000 грн не перевищує поріг 200 000 грн, тому відкриті торги не потрібні. "
    "Потрібно зібрати щонайменше три комерційні пропозиції і перевірити, що в постачальника немає податкового боргу."
)

CASES = [
    {
        "id": "good",
        "answer": REFERENCE,
    },
    {
        "id": "hallucinated",
        "answer": (
            "Закупівлю на 150 000 грн можна провести лише після погодження міністерства. "
            "Поріг без торгів становить 2 000 000 грн, а комерційні пропозиції не потрібні."
        ),
    },
    {
        "id": "offtopic",
        "answer": "Завтра в місті очікується дощ, вітер до 15 м/с і температура близько 8 градусів.",
    },
    {
        "id": "partial",
        "answer": "Без відкритих торгів можна закупити що завгодно до 50 000 грн. Пропозиції постачальників не згадуються.",
    },
    {
        "id": "refusal_wrong",
        "answer": "У контексті немає жодних правил закупівель, тому відповісти неможливо.",
    },
]

PRIMARY = {
    "rouge": "rougeL",
    "bertscore": "f1",
    "nli": "support_rate",
    "judge_plain": "score",
    "judge_rubric": "faithfulness",
    "judge_rubric_gpt4o": "faithfulness",
    "judge_plain_gpt4o": "score",
    "pairwise": "win_rate",
    "ragas": "faithfulness",
    "decision": "usefulness",
    "selfcheck": "agreement",
}


def run_sanity(method_names: list[str] | None = None) -> None:
    selected = default_evaluators(include_selfcheck=False)
    if method_names:
        wanted = set(method_names)
        selected = [item for item in selected if item.name in wanted]
        if not selected:
            known = ", ".join(item.name for item in default_evaluators())
            raise SystemExit(f"Немає таких методів. Відомі: {known}")
    failed = False
    for evaluator in selected:
        print(f"\n== {evaluator.name} ==")
        measured = {}
        try:
            for case in CASES:
                scores = evaluator.score(QUESTION, CONTEXT, case["answer"], REFERENCE)
                measured[case["id"]] = scores
                print(f"  {case['id']}: {scores}")
        except Exception as error:
            print(f"  ПРОПУЩЕНО: {error}")
            continue
        key = PRIMARY.get(evaluator.name)
        if not key:
            continue
        good = measured.get("good", {}).get(key)
        hallucinated = measured.get("hallucinated", {}).get(key)
        offtopic = measured.get("offtopic", {}).get(key)
        if good is None or hallucinated is None or offtopic is None:
            print("  НЕМАЄ основного критерію, порядок не перевірено")
            continue
        if abs(good - hallucinated) < 1e-6 or abs(good - offtopic) < 1e-6:
            print(f"  ЗЛАМАНО: {key} не розрізняє добру відповідь і погані")
            failed = True
            continue
        if not (good > hallucinated and good > offtopic):
            print(f"  ЗЛАМАНО: {key} доброї відповіді ({good}) не вищий за помилкові")
            failed = True
            continue
        print(f"  OK: {key} розрізняє добру відповідь, вигадку й відповідь не по питанню")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    from config import configure_stdio

    configure_stdio()
    parser = argparse.ArgumentParser(description="Перевірка методів на контрольних відповідях")
    parser.add_argument("--method", action="append", dest="methods")
    args = parser.parse_args()
    run_sanity(args.methods)
