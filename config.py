"""Імена моделей, параметри чанкування, шляхи та орієнтовні ціни."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-5")
JUDGE_MODEL_OPENAI = os.getenv("JUDGE_MODEL_OPENAI", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
NLI_MODEL = os.getenv(
    "NLI_MODEL",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
)
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "xlm-roberta-large")

TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "800"))
MAX_CHUNK_IN_PROMPT = int(os.getenv("MAX_CHUNK_IN_PROMPT", "2500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
SELFCHECK_SAMPLES = int(os.getenv("SELFCHECK_SAMPLES", "5"))
SELFCHECK_TEMPERATURE = float(os.getenv("SELFCHECK_TEMPERATURE", "0.7"))
NLI_ENTAILMENT_THRESHOLD = float(os.getenv("NLI_ENTAILMENT_THRESHOLD", "0.5"))

# Ознаки зведеного показника довіри. Усі без еталонної відповіді, щоб показник
# працював і в інтерфейсі на новому питанні. Після аналізу списку можна змінити.
TRUST_FEATURES = [
    "nli__support_rate",
    "judge_rubric__faithfulness",
    "judge_rubric__completeness",
    "judge_rubric__source_alignment",
    "judge_rubric__decision_usefulness",
    "ragas__faithfulness",
    "decision__usefulness",
]
TRUST_HIGH = float(os.getenv("TRUST_HIGH", "0.75"))
TRUST_MEDIUM = float(os.getenv("TRUST_MEDIUM", "0.45"))
EXPERT_TRUST_CUTOFF = float(os.getenv("EXPERT_TRUST_CUTOFF", "4"))

# USD за 1 млн токенів: (вхід, вихід). Орієнтир для порівняння вартості методів.
PRICES_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
}

DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "experiments.sqlite"
GOLDEN_PATH = DATA_DIR / "golden_set.json"
PROMPTS_DIR = ROOT / "prompts"
ANNOTATION_DIR = DATA_DIR / "annotations"
FIGURES_DIR = DATA_DIR / "figures"
TRUST_MODEL_PATH = DATA_DIR / "trust_model.joblib"
TRUST_REPORT_PATH = DATA_DIR / "trust_report.json"
COLLECTION_NAME = "docs"

MODES = ("none", "local", "web", "combined")


def configure_stdio() -> None:
    """Windows-консоль часто не utf-8, а звіти й довідка українською."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def price_per_token(model: str) -> tuple[float, float]:
    if model in PRICES_PER_MILLION:
        inp, out = PRICES_PER_MILLION[model]
        return inp / 1_000_000, out / 1_000_000
    name = model.lower()
    if "mini" in name or "haiku" in name:
        inp, out = PRICES_PER_MILLION["gpt-4o-mini"]
    elif "opus" in name:
        inp, out = 15.0, 75.0
    elif "claude" in name or "sonnet" in name:
        inp, out = PRICES_PER_MILLION["claude-sonnet-4-5"]
    elif "gpt-4o" in name:
        inp, out = PRICES_PER_MILLION["gpt-4o"]
    else:
        return 0.0, 0.0
    return inp / 1_000_000, out / 1_000_000


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = price_per_token(model)
    return prompt_tokens * inp + completion_tokens * out


