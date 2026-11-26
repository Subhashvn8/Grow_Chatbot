import unittest

from grow_faq.config import load_settings
from grow_faq.query.answer import allowed_urls
from grow_faq.query.catalog import load_catalog
from grow_faq.query.policy import ANSWER, FACTSHEET, REFUSE
from grow_faq.query.retrieve import RetrievedChunk


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_settings()
        self.catalog = load_catalog(self.settings)

    def test_catalog_has_four_groups(self) -> None:
        self.assertEqual(len(self.catalog.ingested), 5)
        self.assertGreaterEqual(len(self.catalog.factsheets), 5)
        self.assertGreaterEqual(len(self.catalog.education), 3)
        self.assertTrue(self.catalog.statements)
        self.assertTrue(all(url.startswith("https://") for url in self.catalog.all_urls))

    def test_scheme_specific_factsheet_is_selected(self) -> None:
        url = self.catalog.factsheet_url("HDFC Large Cap Fund Direct Growth")
        self.assertIn("Large%20Cap", url)
        self.assertIn(url, self.catalog.factsheet_urls)

    def test_education_and_statement_urls_are_selected(self) -> None:
        self.assertIn(self.catalog.education_url(), self.catalog.education_urls)
        self.assertIn(self.catalog.statement_url(), self.catalog.all_urls)

    def test_answer_cites_only_retrieved_chunks(self) -> None:
        chunk = RetrievedChunk(
            id="c1",
            text="Expense ratio: 0.99%",
            score=0.9,
            source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
            scheme_or_amc="HDFC Large Cap Fund Direct Growth",
            section="fund-details",
            field="expense_ratio",
        )
        allowed = allowed_urls(ANSWER, (chunk,), self.catalog)
        self.assertEqual(allowed, {chunk.source_url})
        self.assertNotIn("https://example.com/blog", allowed)
        self.assertTrue(
            allowed_urls(FACTSHEET, (), self.catalog) <= self.catalog.factsheet_urls
        )
        self.assertTrue(
            allowed_urls(REFUSE, (), self.catalog) <= self.catalog.education_urls
        )

    def test_answer_without_chunks_may_cite_ingested_pages(self) -> None:
        self.assertEqual(allowed_urls(ANSWER, (), self.catalog), self.catalog.ingested_urls)


if __name__ == "__main__":
    unittest.main()
