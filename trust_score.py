"""Зведений показник довіри: логістична регресія на кращих метриках або зважена евристика."""

from __future__ import annotations

import json

import config
from evaluators.decision_eval import DecisionEval
from evaluators.judge_rubric import JudgeRubric
from evaluators.nli_eval import NliEval
from evaluators.ragas_eval import RagasEval

LIKERT_PREFIXES = ("judge_", "decision__")
LEVELS = ("низький", "середній", "високий")

# Ваги евристики, якщо модель ще не навчена. Сума не обов'язково 1: ваги нормуються за наявними ознаками.
HEURISTIC_WEIGHTS = {
    "nli__support_rate": 0.30,
    "judge_rubric__faithfulness": 0.20,
    "judge_rubric__source_alignment": 0.15,
    "judge_rubric__completeness": 0.10,
    "judge_rubric__decision_usefulness": 0.10,
    "decision__usefulness": 0.10,
    "ragas__faithfulness": 0.05,
}


def _scale(name: str, value: float) -> float:
    if name.startswith("judge_") or name.startswith("decision__usefulness"):
        return (float(value) - 1.0) / 4.0
    return max(0.0, min(1.0, float(value)))


def _level(score: float, high: float, medium: float) -> str:
    if score >= high:
        return "високий"
    if score >= medium:
        return "середній"
    return "низький"


def compute_features(question: str, contexts, answer: str, include_ragas: bool = False) -> dict:
    """Рахує ознаки без еталона. RAGAS вмикається окремо: він дорожчий і залежить від версії пакета."""
    features = {}
    for evaluator, prefix in (
        (NliEval(), "nli"),
        (JudgeRubric(), "judge_rubric"),
        (DecisionEval(), "decision"),
    ):
        scores = evaluator.score(question, contexts, answer, reference=None)
        for criterion, value in scores.items():
            features[f"{prefix}__{criterion}"] = float(value)
    if include_ragas:
        try:
            scores = RagasEval().score(question, contexts, answer, reference=None)
            for criterion, value in scores.items():
                features[f"ragas__{criterion}"] = float(value)
        except Exception as error:
            features["ragas__error"] = str(error)
    return features


def _heuristic(features: dict, high: float, medium: float) -> dict:
    weighted = 0.0
    used = 0.0
    breakdown = {}
    for name, weight in HEURISTIC_WEIGHTS.items():
        if name not in features:
            continue
        scaled = _scale(name, features[name])
        breakdown[name] = features[name]
        weighted += weight * scaled
        used += weight
    score = weighted / used if used else 0.0
    return {
        "level": _level(score, high, medium),
        "score": score,
        "breakdown": breakdown,
        "model": "heuristic",
    }


def _load_bundle():
    if not config.TRUST_MODEL_PATH.exists():
        return None
    import joblib

    return joblib.load(config.TRUST_MODEL_PATH)


def trust(question: str, contexts, answer: str, features: dict | None = None, include_ragas: bool = False) -> dict:
    bundle = _load_bundle()
    high = bundle["high"] if bundle else config.TRUST_HIGH
    medium = bundle["medium"] if bundle else config.TRUST_MEDIUM
    if features is None:
        features = compute_features(question, contexts, answer, include_ragas=include_ragas)
    clean = {key: value for key, value in features.items() if isinstance(value, (int, float))}
    if bundle is None:
        return _heuristic(clean, high, medium)
    names = bundle["features"]
    if any(name not in clean for name in names):
        result = _heuristic(clean, high, medium)
        result["model"] = "heuristic_missing_features"
        return result
    import numpy as np

    vector = np.array([[_scale(name, clean[name]) for name in names]])
    probability = float(bundle["model"].predict_proba(vector)[0, 1])
    return {
        "level": _level(probability, high, medium),
        "score": probability,
        "breakdown": {name: clean[name] for name in names},
        "model": "logreg",
    }


def train(test_size: float = 0.3, random_state: int = 42) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from db import to_dataframe

    frame = to_dataframe()
    if "expert_trust" not in frame.columns:
        raise RuntimeError("Немає експертної оцінки trust. Спочатку імпортуйте розмітку.")
    features = [name for name in config.TRUST_FEATURES if name in frame.columns]
    if len(features) < 2:
        raise RuntimeError(f"Замало ознак для навчання. Є колонки: {features}")
    data = frame.dropna(subset=features + ["expert_trust"]).copy()
    data["y"] = (data["expert_trust"] >= config.EXPERT_TRUST_CUTOFF).astype(int)
    if data["y"].nunique() < 2:
        raise RuntimeError("В експертній розмітці лише один клас довіри, регресію навчити не можна.")
    x = np.column_stack([data[name].map(lambda value, name=name: _scale(name, value)) for name in features])
    y = data["y"].to_numpy()
    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    model_auc = float(roc_auc_score(y_test, proba))
    singles = []
    for index, name in enumerate(features):
        try:
            auc = float(roc_auc_score(y_test, x_test[:, index]))
        except ValueError:
            continue
        singles.append({"feature": name, "auc": auc})
    singles.sort(key=lambda item: item["auc"], reverse=True)
    best = singles[0] if singles else None
    high, medium = _choose_thresholds(y_test, proba)
    bundle = {"model": model, "features": features, "high": high, "medium": medium}
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(bundle, config.TRUST_MODEL_PATH)
    report = {
        "model_auc": model_auc,
        "best_single": best,
        "features": features,
        "thresholds": {"high": high, "medium": medium},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate": float(y.mean()),
    }
    config.TRUST_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _choose_thresholds(y_true, proba) -> tuple[float, float]:
    """Високий поріг тримає точність класу «довіряти» близько 0.8, якщо даних вистачає."""
    import numpy as np

    high = config.TRUST_HIGH
    best_precision = 0.0
    for threshold in np.linspace(0.5, 0.9, 9):
        predicted = proba >= threshold
        if predicted.sum() == 0:
            continue
        precision = float(y_true[predicted].mean())
        if precision >= 0.8 and threshold <= high:
            high = float(threshold)
            best_precision = precision
        elif precision > best_precision and precision >= 0.8:
            high = float(threshold)
            best_precision = precision
    medium = min(config.TRUST_MEDIUM, high - 0.15)
    medium = max(0.2, medium)
    return float(high), float(medium)


if __name__ == "__main__":
    config.configure_stdio()
    print(json.dumps(train(), ensure_ascii=False, indent=2))
