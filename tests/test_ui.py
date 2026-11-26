import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from grow_faq.config import load_settings
from grow_faq.query.ask import AskResult, AskStageError
from grow_faq.query.pii import PII_MESSAGE
from grow_faq.query.retrieve import StaleModelError
from grow_faq.ui.app import app, configure

QUESTION = "Expense ratio of HDFC Large Cap Fund Direct Growth?"
URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
TEXT = f"Expense ratio is 0.99%.\nSource: {URL}\nLast updated from sources: 2026-11-26"


class UiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_settings()
        configure(self.settings)
        self.client = TestClient(app)

    def result(self, text=TEXT, citation=URL, policy="answer", pii=False):
        return AskResult(
            text=text,
            policy=policy,
            citation=citation,
            chunks=(),
            pii=pii,
            trace_path=None,
        )

    def test_page_has_required_regions_and_no_identity_fields(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.text

        self.assertIn("HDFC Mutual Fund", page)
        for name in (
            "HDFC Large Cap Fund Direct Growth",
            "HDFC Flexi Cap Direct Plan Growth",
            "HDFC ELSS Tax Saver Fund Direct Plan Growth",
            "HDFC Mid Cap Fund Direct Growth",
        ):
            self.assertIn(name, page)
        self.assertIn("Facts-only. No investment advice.", page)
        self.assertIn("Expense ratio of HDFC Large Cap Fund Direct Growth?", page)
        self.assertIn("What is the ELSS lock-in period?", page)
        self.assertIn("Exit load of HDFC Mid Cap Fund Direct Growth?", page)
        self.assertEqual(page.count('type="text"'), 1)

        lowered = page.casefold()
        for word in ("email", "phone", "aadhaar", "account number", " pan "):
            self.assertNotIn(word, lowered)

    def test_ask_endpoint_returns_same_text_as_cli(self) -> None:
        with patch("grow_faq.ui.app.run_ask", return_value=self.result()) as ask_mock:
            response = self.client.post("/ask", json={"question": QUESTION})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], TEXT)
        self.assertEqual(data["citation"], URL)
        self.assertEqual(data["policy"], "answer")
        self.assertEqual(data["trace_path"], None)
        ask_mock.assert_called_once_with(QUESTION, settings=self.settings)

    def test_pii_answer_is_forwarded_without_a_citation(self) -> None:
        with patch(
            "grow_faq.ui.app.run_ask",
            return_value=self.result(text=PII_MESSAGE, citation=None, policy="pii", pii=True),
        ):
            response = self.client.post("/ask", json={"question": "PAN ABCDE1234F"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["pii"])
        self.assertEqual(data["text"], PII_MESSAGE)
        self.assertIsNone(data["citation"])

    def test_stale_model_returns_operator_message(self) -> None:
        with patch(
            "grow_faq.ui.app.run_ask",
            side_effect=StaleModelError("run grow-embed and grow-index"),
        ):
            response = self.client.post("/ask", json={"question": QUESTION})

        self.assertEqual(response.status_code, 503)
        self.assertIn("grow-embed", response.json()["text"])
        self.assertIn("grow-index", response.json()["text"])

    def test_provider_failure_has_no_guessed_fact(self) -> None:
        with patch("grow_faq.ui.app.run_ask", side_effect=RuntimeError("timeout")):
            response = self.client.post("/ask", json={"question": QUESTION})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["text"], "The answer could not be generated.")

    def test_pipeline_error_reports_stage_without_secrets(self) -> None:
        with patch(
            "grow_faq.ui.app.run_ask",
            side_effect=AskStageError(
                "retrieval", RuntimeError("Chroma request failed with HTTP 401")
            ),
        ):
            response = self.client.post("/ask", json={"question": QUESTION})

        data = response.json()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(data["stage"], "retrieval")
        self.assertIn("HTTP 401", data["detail"])
        self.assertEqual(data["text"], "The answer could not be generated.")
        self.assertIsNone(data["citation"])


if __name__ == "__main__":
    unittest.main()
