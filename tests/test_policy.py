import unittest

from grow_faq.config import load_settings
from grow_faq.query.policy import (
    ANSWER,
    FACTSHEET,
    OUT_OF_SCOPE,
    REFUSE,
    asked_field,
    classify,
)
from grow_faq.query.schemes import scheme_names, single_scheme


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_settings()

    def test_advice_is_refused(self) -> None:
        self.assertEqual(classify("Should I buy HDFC Flexi Cap?"), REFUSE)

    def test_advice_phrased_as_fact_is_still_a_refusal(self) -> None:
        self.assertEqual(classify("Should I buy this for the return?"), REFUSE)

    def test_advice_phrasings_are_refused(self) -> None:
        for question in (
            "Is it a good time to buy HDFC Mid Cap?",
            "Should I sell my HDFC ELSS units?",
            "Hold or sell HDFC Large Cap?",
            "Which fund should I invest in?",
            "Which is better, Flexi Cap or Large Cap?",
            "Is HDFC Flexi Cap worth investing in?",
            "Can you recommend an HDFC fund?",
            "Help me choose between these schemes",
            "What is the best fund among these?",
        ):
            with self.subTest(question=question):
                self.assertEqual(classify(question), REFUSE)

    def test_transaction_verbs_in_factual_questions_are_answered(self) -> None:
        for question in (
            "What is the exit load if I sell within a year?",
            "How do I buy HDFC Mid Cap Fund Direct Growth?",
            "Minimum SIP to hold HDFC ELSS?",
            "Can I sell HDFC ELSS units before the lock-in ends?",
            "How do I choose the SIP date?",
        ):
            with self.subTest(question=question):
                self.assertEqual(classify(question), ANSWER)

    def test_other_fund_houses_are_out_of_scope(self) -> None:
        for question in (
            "What is the expense ratio of SBI Bluechip Fund?",
            "ICICI Prudential Bluechip exit load?",
            "Returns of Axis Midcap Fund?",
        ):
            with self.subTest(question=question):
                self.assertEqual(classify(question), OUT_OF_SCOPE)

    def test_how_to_invest_is_a_field(self) -> None:
        for question in (
            "How do I buy HDFC Mid Cap Fund Direct Growth?",
            "How to invest in HDFC Flexi Cap?",
            "Where can I invest in HDFC ELSS?",
        ):
            with self.subTest(question=question):
                self.assertEqual(asked_field(question), "how_to_invest")

    def test_returns_question_goes_to_factsheet(self) -> None:
        self.assertEqual(
            classify("Which of these four had the highest return?"), FACTSHEET
        )

    def test_factual_field_is_an_answer(self) -> None:
        self.assertEqual(
            classify("Expense ratio of HDFC Large Cap Fund Direct Growth?"), ANSWER
        )

    def test_asked_fields_are_detected(self) -> None:
        cases = {
            "Expense ratio of Large Cap?": "expense_ratio",
            "What is the exit load?": "exit_load",
            "Minimum SIP amount?": "minimum_sip",
            "ELSS lock-in period?": "lock_in",
            "Riskometer of Mid Cap?": "riskometer",
            "Benchmark of Flexi Cap?": "benchmark",
            "How to download a capital-gains statement?": "statement_download",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(asked_field(question), expected)

    def test_one_scheme_is_detected_for_filtering(self) -> None:
        question = "Expense ratio of HDFC Large Cap Fund Direct Growth?"
        self.assertEqual(
            single_scheme(question, self.settings),
            "HDFC Large Cap Fund Direct Growth",
        )

    def test_lock_in_maps_to_elss(self) -> None:
        self.assertEqual(
            single_scheme("What is the ELSS lock-in?", self.settings),
            "HDFC ELSS Tax Saver Fund Direct Plan Growth",
        )

    def test_two_schemes_disable_the_filter(self) -> None:
        question = "Large Cap and Mid Cap expense ratios"
        self.assertEqual(len(scheme_names(question, self.settings)), 2)
        self.assertIsNone(single_scheme(question, self.settings))

    def test_no_scheme_disables_the_filter(self) -> None:
        self.assertIsNone(single_scheme("What is a benchmark?", self.settings))


if __name__ == "__main__":
    unittest.main()
