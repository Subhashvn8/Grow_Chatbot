import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from grow_faq.config import load_settings
from grow_faq.ingest.chunk import run_chunk


class ChunkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_settings = load_settings()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = replace(
            self.base_settings,
            documents_dir=root / "documents",
            chunks_dir=root / "chunks",
            corpus_file=root / "corpus.json",
        )
        self.settings.documents_dir.mkdir(parents=True, exist_ok=True)
        self._write_corpus(ok=True, stage="load")

    def _write_corpus(self, ok: bool, stage: str) -> None:
        self.settings.corpus_file.write_text(
            json.dumps({"ok": ok, "stage": stage, "documents": []}),
            encoding="utf-8",
        )

    def _write_doc(self, slug: str, name: str, text: str, url: str) -> None:
        (self.settings.documents_dir / f"{slug}.json").write_text(
            json.dumps(
                {
                    "url": url,
                    "scheme_or_amc": name,
                    "fetched_at": "2026-01-01T00:00:00Z",
                    "text": text,
                }
            ),
            encoding="utf-8",
        )

    def _chunks(self) -> list[dict[str, str]]:
        path = self.settings.chunks_dir / "chunks.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def test_fact_and_prose_chunks_are_stable_and_grounded(self) -> None:
        page = self.base_settings.pages[1]
        self._write_doc(
            page.slug,
            page.name,
            "\n".join(
                [
                    page.name,
                    "Equity: Large Cap: Very High Risk",
                    "Min. for SIP: ₹100",
                    "Expense ratio: 0.99%",
                    "Fund benchmark: NIFTY 100 Total Return Index",
                    "Exit load of 1% if redeemed within 1 year",
                    f"About {page.name}",
                    f"{page.name} is a facts-only overview with enough words to form one prose chunk for the chunker demo.",
                    "Investment Objective: The scheme seeks long-term capital appreciation by investing predominantly in large-cap companies.",
                ]
            ),
            page.url,
        )

        self.assertTrue(run_chunk(self.settings))
        chunks = self._chunks()
        by_id = {chunk["id"]: chunk for chunk in chunks}

        self.assertEqual(
            set(by_id[f"{page.slug}-expense_ratio"]),
            {"id", "text", "source_url", "scheme_or_amc", "section", "field", "fetched_at"},
        )
        self.assertIn("0.99%", by_id[f"{page.slug}-expense_ratio"]["text"])
        self.assertIn(page.name, by_id[f"{page.slug}-expense_ratio"]["text"])
        self.assertIn("₹100", by_id[f"{page.slug}-minimum_sip"]["text"])
        self.assertIn("1% if redeemed", by_id[f"{page.slug}-exit_load"]["text"])
        self.assertIn("NIFTY 100", by_id[f"{page.slug}-benchmark"]["text"])
        self.assertEqual(by_id[f"{page.slug}-riskometer"]["text"].count("Very High Risk"), 1)
        self.assertIn("overview", [chunk["field"] for chunk in chunks])

        for chunk in chunks:
            self.assertEqual(chunk["scheme_or_amc"], page.name)
            self.assertEqual(chunk["source_url"], page.url)
            self.assertEqual(chunk["fetched_at"], "2026-01-01T00:00:00Z")

    def test_rerun_replaces_same_ids_with_new_values(self) -> None:
        page = self.base_settings.pages[1]
        text = "\n".join([page.name, "Expense ratio: 0.99%"])
        self._write_doc(page.slug, page.name, text, page.url)

        self.assertTrue(run_chunk(self.settings))
        first = {chunk["id"]: chunk["text"] for chunk in self._chunks()}
        first_id = f"{page.slug}-expense_ratio"
        self.assertIn(first_id, first)

        self._write_doc(
            page.slug,
            page.name,
            "\n".join([page.name, "Expense ratio: 1.00%"]),
            page.url,
        )
        self.assertTrue(run_chunk(self.settings))
        second = {chunk["id"]: chunk["text"] for chunk in self._chunks()}

        self.assertEqual(set(first), set(second))
        self.assertIn("1.00%", second[first_id])
        self.assertNotIn("0.99%", second[first_id])

    def test_missing_fact_is_skipped(self) -> None:
        page = self.base_settings.pages[3]
        self._write_doc(page.slug, page.name, "\n".join([page.name, "Expense ratio: 1.18%"]), page.url)

        self.assertTrue(run_chunk(self.settings))
        ids = {chunk["id"] for chunk in self._chunks()}
        self.assertIn(f"{page.slug}-expense_ratio", ids)
        self.assertNotIn(f"{page.slug}-exit_load", ids)

    def test_amc_scheme_list_does_not_become_chunks(self) -> None:
        amc = self.base_settings.pages[0]
        self._write_doc(
            amc.slug,
            amc.name,
            "\n".join(
                [
                    amc.name,
                    "HDFC Asset Management Company Ltd., or HDFC Mutual Fund, is an asset manager with a long public history.",
                    "List of HDFC Mutual Fund in India",
                    "HDFC ELSS Tax Saver Fund",
                    "HDFC Mid Cap Fund Direct Growth",
                ]
            ),
            amc.url,
        )

        self.assertTrue(run_chunk(self.settings))
        text = "\n".join(chunk["text"] for chunk in self._chunks())
        self.assertNotIn("HDFC ELSS Tax Saver Fund", text)
        self.assertNotIn("HDFC Mid Cap Fund Direct Growth", text)

    def test_long_prose_is_split_with_overlap(self) -> None:
        page = self.base_settings.pages[1]
        self.settings = replace(self.settings, max_prose_words=10, overlap_words=2)
        words = " ".join(f"w{index}" for index in range(30))
        self._write_doc(page.slug, page.name, "\n".join(["About test", words]), page.url)

        self.assertTrue(run_chunk(self.settings))
        parts = [chunk for chunk in self._chunks() if chunk["field"] == "overview"]
        self.assertGreater(len(parts), 1)
        first_body = parts[0]["text"].split("\n", 1)[1].split()
        second_body = parts[1]["text"].split("\n", 1)[1].split()
        self.assertEqual(first_body[-2:], second_body[:2])

    def test_failed_load_corpus_blocks_chunking(self) -> None:
        self._write_corpus(ok=False, stage="load")
        self.assertFalse(run_chunk(self.settings))
        self.assertFalse((self.settings.chunks_dir / "chunks.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
