import unittest

from grow_faq.config import load_settings
from grow_faq.query.answer import (
    _count_sentences,
    failure_answer,
    source_date,
    template_answer,
    validate_draft,
)
from grow_faq.query.catalog import load_catalog
from grow_faq.query.policy import ANSWER, FACTSHEET
from grow_faq.query.retrieve import RetrievedChunk


def make_chunk(text="Expense ratio of HDFC Large Cap Fund Direct Growth: 0.99%"):
    return RetrievedChunk(
        id="hdfc-large-cap-expense-ratio",
        text=text,
        score=0.81,
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        scheme_or_amc="HDFC Large Cap Fund Direct Growth",
        section="fund-details",
        field="expense_ratio",
    )


class AnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_settings()
        self.catalog = load_catalog(self.settings)
        self.date = "2026-11-26"
        self.allowed = frozenset({make_chunk().source_url})

    def draft(self, body, url=None):
        return (
            f"{body}\nSource: {url or make_chunk().source_url}\n"
            f"Last updated from sources: {self.date}"
        )

    def test_valid_draft_is_accepted(self) -> None:
        draft = self.draft("Expense ratio is 0.99%.")
        answer = validate_draft(
            draft,
            policy=ANSWER,
            allowed=self.allowed,
            date=self.date,
            max_sentences=3,
        )
        self.assertIsNotNone(answer)
        self.assertEqual(answer.citation, make_chunk().source_url)

    def test_bad_citation_is_rejected(self) -> None:
        draft = self.draft("Expense ratio is 0.99%.", "https://example.com/blog")
        self.assertIsNone(
            validate_draft(
                draft,
                policy=ANSWER,
                allowed=self.allowed,
                date=self.date,
                max_sentences=3,
            )
        )

    def test_too_long_draft_is_rejected(self) -> None:
        draft = self.draft("One. Two. Three. Four.")
        self.assertGreater(_count_sentences(draft), 3)
        self.assertIsNone(
            validate_draft(
                draft,
                policy=ANSWER,
                allowed=self.allowed,
                date=self.date,
                max_sentences=3,
            )
        )

    def test_wrong_or_missing_date_is_rejected(self) -> None:
        for draft in (
            self.draft("Ratio is 0.99%.").replace(self.date, "2026-11-27"),
            "Ratio is 0.99%.\nSource: " + make_chunk().source_url,
        ):
            with self.subTest(draft=draft):
                self.assertIsNone(
                    validate_draft(
                        draft,
                        policy=ANSWER,
                        allowed=self.allowed,
                        date=self.date,
                        max_sentences=3,
                    )
                )

    def test_advice_language_is_rejected_for_facts(self) -> None:
        draft = self.draft("You should buy this fund.")
        self.assertIsNone(
            validate_draft(
                draft,
                policy=ANSWER,
                allowed=self.allowed,
                date=self.date,
                max_sentences=3,
            )
        )

    def test_factsheet_draft_with_a_return_is_rejected(self) -> None:
        url = "https://www.hdfcfund.com/mutual-funds/factsheets"
        draft = self.draft("The fund returned 12.4%.", url)
        self.assertIsNone(
            validate_draft(
                draft,
                policy=FACTSHEET,
                allowed=frozenset({url}),
                date=self.date,
                max_sentences=3,
            )
        )

    def test_template_is_short_grounded_and_dated(self) -> None:
        answer = template_answer(
            ANSWER,
            (make_chunk(),),
            self.catalog,
            self.settings,
            self.date,
        )
        self.assertEqual(answer.citation, make_chunk().source_url)
        self.assertIn("0.99%", answer.text)
        self.assertIn(f"Last updated from sources: {self.date}", answer.text)
        self.assertLessEqual(_count_sentences(answer.text), 3)

    def test_failure_answer_has_no_guessed_fact(self) -> None:
        answer = failure_answer(
            ANSWER,
            (make_chunk(),),
            self.catalog,
            self.settings,
            self.date,
        )
        self.assertIn("The answer could not be generated.", answer.text)
        self.assertNotIn("0.99%", answer.text)
        self.assertEqual(answer.citation, make_chunk().source_url)
        self.assertLessEqual(_count_sentences(answer.text), 3)

    def test_source_date_comes_from_corpus_not_today(self) -> None:
        self.assertEqual(source_date(self.settings), self.date)


if __name__ == "__main__":
    unittest.main()
