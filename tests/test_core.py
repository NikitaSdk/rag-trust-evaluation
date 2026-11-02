"""Перевірки без ключів API: база, промпт, режими, реранкінг і ROUGE."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from db import load_answers, method_done, save_annotation, save_answer, save_score, to_dataframe
from evaluators.judge_pairwise import _winner_score
from evaluators.rouge_eval import RougeEval, tokenize
from llm import build_prompt, parse_json
from retrievers import combined_search, deduplicate


class ParseTests(unittest.TestCase):
    def test_json_fence(self):
        data = parse_json('```json\n{"score": 4, "reasoning": "ок"}\n```')
        self.assertEqual(data["score"], 4)

    def test_json_with_preamble(self):
        data = parse_json('Ось результат: {"winner": "A"}')
        self.assertEqual(data["winner"], "A")


class PromptTests(unittest.TestCase):
    def test_context_is_numbered(self):
        prompt = build_prompt(
            "Скільки?",
            [{"text": "Поріг 200 000 грн.", "source": "політика"}],
        )
        self.assertIn("[1] (політика)", prompt)
        self.assertIn("Скільки?", prompt)
        self.assertIn("ЛИШЕ", prompt)

    def test_no_context_is_bare_question(self):
        self.assertEqual(build_prompt("Питання як є", None), "Питання як є")

    def test_prompt_files_have_placeholders(self):
        needed = {
            "generate.txt": ["{{CONTEXT}}", "{{QUESTION}}"],
            "claim_split.txt": ["{{ANSWER}}"],
            "judge_plain.txt": ["{{QUESTION}}", "{{ANSWER}}"],
            "judge_pairwise.txt": ["{{ANSWER_A}}", "{{ANSWER_B}}"],
            "decision_usefulness.txt": ["{{ANSWER}}", "missing"],
        }
        for name, parts in needed.items():
            text = (config.PROMPTS_DIR / name).read_text(encoding="utf-8")
            for part in parts:
                self.assertIn(part, text, name)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = Path(self.tmp.name) / "exp.sqlite"
        self.patch = patch("config.DB_PATH", self.path)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_roundtrip_and_skip(self):
        answer_id = save_answer(
            {
                "question": "Який поріг?",
                "mode": "local",
                "answer": "200 000 грн",
                "contexts": [{"text": "поріг", "source": "a", "origin": "local"}],
                "latency": 1.2,
                "tokens": 30,
                "cost": 0.001,
            },
            reference="200 000 грн",
        )
        again = save_answer(
            {"question": "Який поріг?", "mode": "local", "answer": "інша", "contexts": [], "latency": 0, "tokens": 0},
            reference="200 000 грн",
        )
        self.assertEqual(answer_id, again)
        save_score(answer_id, "rouge", "rougeL", 0.8, cost=0.0, latency=0.1)
        self.assertFalse(method_done(answer_id, "rouge", 0))
        from db import mark_method_done

        mark_method_done(answer_id, "rouge", 0)
        self.assertTrue(method_done(answer_id, "rouge", 0))
        save_annotation(answer_id, "expert1", {"faithfulness": 5, "trust": 4, "hallucination": 0})
        loaded = load_answers()
        self.assertEqual(loaded[0]["answer"], "200 000 грн")
        self.assertEqual(loaded[0]["annotations"]["expert1"]["trust"], 4)
        frame = to_dataframe()
        self.assertEqual(int(frame.loc[0, "rouge__rougeL"] * 10), 8)
        self.assertEqual(frame.loc[0, "expert_trust"], 4)


class PipelineTests(unittest.TestCase):
    def test_modes_dispatch(self):
        seen = {}

        def fake_generate(question, contexts=None, temperature=None):
            seen["contexts"] = contexts
            fake_generate.last_tokens = 11
            fake_generate.last_cost = 0.0
            return "відповідь"

        local = [{"text": "локальний", "source": "doc", "origin": "local"}]
        with patch("retrievers.local_search", return_value=local) as local_search, patch(
            "retrievers.web_search", return_value=[]
        ), patch("retrievers.combined_search", return_value=[]), patch("pipeline.generate", fake_generate):
            from pipeline import run

            none = run("питання", "none")
            self.assertIsNone(seen["contexts"])
            self.assertEqual(none["contexts"], [])
            local_result = run("питання", "local")
            local_search.assert_called_once()
            self.assertEqual(local_result["contexts"], local)
            self.assertEqual(local_result["tokens"], 11)

    def test_unknown_mode(self):
        from pipeline import run

        with self.assertRaises(ValueError):
            run("питання", "cloud")


class RetrieverTests(unittest.TestCase):
    def test_dedup(self):
        chunks = [
            {"text": "Той самий текст", "source": "a", "origin": "local"},
            {"text": "той   самий текст", "source": "b", "origin": "web"},
            {"text": "Інший", "source": "c", "origin": "web"},
        ]
        self.assertEqual(len(deduplicate(chunks)), 2)

    def test_rerank_keeps_closer_chunk(self):
        local = [{"text": "далекий", "source": "a", "origin": "local"}]
        web = [{"text": "близький", "source": "b", "origin": "web"}]
        with patch("retrievers.local_search", return_value=local), patch(
            "retrievers.web_search", return_value=web
        ), patch("retrievers.embed_query", return_value=[1.0, 0.0]), patch(
            "retrievers.embed_passages",
            return_value=[[0.0, 1.0], [1.0, 0.0]],
        ):
            ranked = combined_search("питання", k=1)
        self.assertEqual(ranked[0]["text"], "близький")


class RougeTests(unittest.TestCase):
    def test_tokenizer_keeps_ukrainian(self):
        tokens = tokenize("Освітленість нижча за 10 лк, є звіт.")
        self.assertIn("освітленість", tokens)
        self.assertIn("лк", tokens)

    def test_good_answer_beats_hallucination_and_offtopic(self):
        evaluator = RougeEval()
        reference = "Спрощена закупівля без відкритих торгів дозволена до 200 000 грн."
        good = evaluator.score("", [], reference, reference)["rougeL"]
        bad = evaluator.score(
            "",
            [],
            "Потрібне погодження міністерства, а поріг без торгів становить 2 000 000 грн.",
            reference,
        )["rougeL"]
        offtopic = evaluator.score("", [], "Завтра очікується дощ і сильний вітер.", reference)["rougeL"]
        self.assertGreater(good, bad)
        self.assertGreater(good, offtopic)
        self.assertGreater(bad, offtopic)


class GoldenSetTests(unittest.TestCase):
    def test_covers_annotation_sample(self):
        items = json.loads(config.GOLDEN_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(items) * len(config.MODES), 100)
        for item in items:
            self.assertTrue(item["question"].strip())
            self.assertTrue(item["reference"].strip())


if __name__ == "__main__":
    unittest.main()
