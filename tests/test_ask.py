import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from grow_faq.config import load_settings
from grow_faq.query.answer import _count_sentences
from grow_faq.query.ask import AskStageError, run_ask
from grow_faq.query.catalog import load_catalog
from grow_faq.query.pii import PII_MESSAGE
from grow_faq.query.retrieve import RetrievedChunk, StaleModelError


class FakeLLM:
    def __init__(self, draft=None, error=None):
        self.draft = draft
        self.error = error
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.draft


class AskTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        corpus = root / "corpus.json"
        corpus.write_text(
            json.dumps({"ok": True, "last_updated": "2026-11-26T17:34:15Z"}),
            encoding="utf-8",
        )
        self.settings = replace(
            load_settings(),
            traces_dir=root / "traces",
            corpus_file=corpus,
        )
        self.catalog = load_catalog(self.settings)
        self.date = "2026-11-26"
        self.chunk = RetrievedChunk(
            id="hdfc-large-cap-expense-ratio",
            text="Expense ratio of HDFC Large Cap Fund Direct Growth: 0.99%",
            score=0.81,
            source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
            scheme_or_amc="HDFC Large Cap Fund Direct Growth",
            section="fund-details",
            field="expense_ratio",
        )

    def ask(self, question, llm, chunks):
        with patch("grow_faq.query.ask.retrieve", return_value=list(chunks)):
            return run_ask(
                question,
                settings=self.settings,
                llm=llm,
                client=object(),
            )

    def draft(self, body, url):
        return f"{body}\nSource: {url}\nLast updated from sources: {self.date}"

    def traces(self):
        return list(self.settings.traces_dir.glob("*.json"))

    def test_happy_path_uses_chunk_url_corpus_date_and_writes_trace(self) -> None:
        question = "Expense ratio of HDFC Large Cap Fund Direct Growth?"
        draft = self.draft("Expense ratio is 0.99%.", self.chunk.source_url)
        llm = FakeLLM(draft)

        result = self.ask(question, llm, [self.chunk])

        self.assertEqual(result.text, draft)
        self.assertEqual(result.citation, self.chunk.source_url)
        self.assertFalse(result.pii)
        self.assertIn(self.chunk.text, llm.calls[0][0])
        self.assertIn(self.chunk.source_url, llm.calls[0][0])
        traces = self.traces()
        self.assertEqual(len(traces), 1)
        trace = json.loads(traces[0].read_text(encoding="utf-8"))
        self.assertEqual(trace["question"], question)
        self.assertEqual(trace["policy"], "answer")
        self.assertEqual(trace["chunk_ids"], [self.chunk.id])
        self.assertEqual(trace["scores"], [0.81])
        self.assertEqual(trace["citation"], self.chunk.source_url)
        self.assertEqual(trace["embedding_model"], self.settings.embedding_model)
        self.assertEqual(trace["llm_provider"], self.settings.llm_provider)

    def test_pii_is_rejected_before_retrieval_llm_and_trace(self) -> None:
        llm = FakeLLM("unused")
        with patch(
            "grow_faq.query.ask.retrieve",
            side_effect=AssertionError("retriever must not run"),
        ):
            result = run_ask(
                "My PAN is ABCDE1234F",
                settings=self.settings,
                llm=llm,
                client=object(),
            )

        self.assertTrue(result.pii)
        self.assertEqual(result.text, PII_MESSAGE)
        self.assertEqual(result.trace_path, None)
        self.assertEqual(llm.calls, [])
        self.assertEqual(self.traces(), [])
        self.assertNotIn("ABCDE1234F", result.text)

    def test_advice_refusal_uses_education_url(self) -> None:
        llm = FakeLLM(self.draft("You should buy it.", "https://example.com/blog"))
        result = self.ask("Should I buy HDFC Flexi Cap?", llm, [self.chunk])

        self.assertEqual(result.policy, "refuse")
        self.assertEqual(result.citation, self.catalog.education_url())
        self.assertIn("I can only share sourced facts", result.text)
        self.assertNotIn("buy it", result.text)
        self.assertLessEqual(_count_sentences(result.text), 3)

    def test_returns_question_uses_factsheet_url_without_computing(self) -> None:
        llm = FakeLLM(self.draft("It had 22% returns.", "https://example.com/blog"))
        result = self.ask(
            "Which of these four had the highest return?", llm, [self.chunk]
        )

        self.assertEqual(result.policy, "factsheet")
        self.assertEqual(result.citation, self.catalog.factsheet_url())
        self.assertIn("can't calculate or compare returns", result.text)
        self.assertNotIn("22%", result.text)

    def test_bad_citation_falls_back_to_chunk_url(self) -> None:
        llm = FakeLLM(self.draft("Expense ratio is 0.99%.", "https://example.com/blog"))
        result = self.ask("Expense ratio of Large Cap?", llm, [self.chunk])

        self.assertEqual(result.citation, self.chunk.source_url)
        self.assertIn("0.99%", result.text)
        self.assertNotIn("example.com", result.text)

    def test_too_long_draft_is_replaced_by_template(self) -> None:
        llm = FakeLLM(
            self.draft("One. Two. Three. Four.", self.chunk.source_url)
        )
        result = self.ask("Expense ratio of Large Cap?", llm, [self.chunk])

        self.assertLessEqual(_count_sentences(result.text), 3)
        self.assertIn("0.99%", result.text)

    def test_missing_fact_does_not_call_llm(self) -> None:
        llm = FakeLLM("unused")
        result = self.ask(
            "Exit load of HDFC Large Cap Fund Direct Growth?", llm, [self.chunk]
        )

        self.assertEqual(llm.calls, [])
        self.assertIn("Not in the sourced pages.", result.text)
        self.assertEqual(
            result.citation,
            self.catalog.ingested_url("HDFC Large Cap Fund Direct Growth"),
        )

    def test_statement_question_without_chunk_cites_statement_page(self) -> None:
        llm = Mock()
        result = self.ask(
            "How to download a capital-gains statement?", llm, []
        )

        llm.generate.assert_not_called()
        self.assertEqual(result.citation, self.catalog.statement_url())
        self.assertIn("Not in the sourced pages.", result.text)

    def test_provider_failure_has_no_guessed_fact(self) -> None:
        llm = FakeLLM(error=RuntimeError("provider down"))
        result = self.ask("Expense ratio of Large Cap?", llm, [self.chunk])

        self.assertIn("The answer could not be generated.", result.text)
        self.assertNotIn("0.99%", result.text)
        self.assertEqual(result.citation, self.chunk.source_url)
        self.assertEqual(len(self.traces()), 1)

    def test_stale_model_stops_before_trace(self) -> None:
        with patch(
            "grow_faq.query.ask.retrieve",
            side_effect=StaleModelError("run grow-embed and grow-index"),
        ):
            with self.assertRaises(StaleModelError):
                run_ask(
                    "Expense ratio of Large Cap?",
                    settings=self.settings,
                    llm=FakeLLM("unused"),
                    client=object(),
                )
        self.assertEqual(self.traces(), [])

    def test_retrieval_failure_is_labelled_without_a_trace(self) -> None:
        with patch(
            "grow_faq.query.ask.retrieve",
            side_effect=RuntimeError("Chroma request failed with HTTP 401"),
        ):
            with self.assertRaises(AskStageError) as caught:
                run_ask(
                    "Expense ratio of Large Cap?",
                    settings=self.settings,
                    llm=FakeLLM("unused"),
                    client=object(),
                )

        self.assertEqual(caught.exception.stage, "retrieval")
        self.assertIn("HTTP 401", caught.exception.detail)
        self.assertEqual(self.traces(), [])


if __name__ == "__main__":
    unittest.main()
