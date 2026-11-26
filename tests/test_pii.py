import unittest

from grow_faq.query.pii import PII_MESSAGE, contains_pii, find_pii_kind


class PiiTests(unittest.TestCase):
    def test_identifiers_are_detected(self) -> None:
        cases = {
            "My PAN is ABCDE1234F": "PAN",
            "Aadhaar 1234 5678 9012": "Aadhaar",
            "Call me on 9876543210": "phone number",
            "Email me at first.last@example.com": "email",
            "Account 123456789012": "account number",
            "My OTP is 482913": "OTP",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(find_pii_kind(question), expected)
                self.assertTrue(contains_pii(question))

    def test_factual_questions_are_not_flagged(self) -> None:
        for question in (
            "Expense ratio of HDFC Large Cap Fund Direct Growth?",
            "What is the exit load?",
            "Minimum SIP is 500",
            "Last updated from sources: 2026-11-26",
        ):
            with self.subTest(question=question):
                self.assertFalse(contains_pii(question))

    def test_rejection_message_never_echoes_an_identifier(self) -> None:
        self.assertNotIn("1234", PII_MESSAGE)
        self.assertIn("personal identifiers", PII_MESSAGE)


if __name__ == "__main__":
    unittest.main()
