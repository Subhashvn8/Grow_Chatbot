import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from grow_faq.config import load_settings
from grow_faq.ingest.load import run_load, extract_text


def make_html(name: str) -> str:
    return f"""<html><body>
<div class="cookie-banner">COOKIE_MARKER</div>
<nav>NAV_MARKER</nav>
<footer>FOOTER_MARKER</footer>
<script id="__NEXT_DATA__">{{"hidden":"SCRIPT_MARKER"}}</script>
<div class="layout-main">
  <header><h1>{name}</h1></header>
  <h3>Fund details</h3>
  <div class="fundDetails_gap4"><span>Expense ratio</span><span>0.99%</span></div>
  <p>{name} is a short but usable facts-only test document for loading.</p>
</div>
</body></html>"""


class LoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_settings = load_settings()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = replace(
            self.base_settings,
            raw_dir=root / "raw",
            documents_dir=root / "documents",
            corpus_file=root / "corpus.json",
            min_document_chars=30,
        )
        self.settings.raw_dir.mkdir(parents=True, exist_ok=True)
        self.fetched_times: list[str] = []

    def _write_raw(self, html_by_slug: dict[str, str]) -> None:
        self.fetched_times = []
        for index, page in enumerate(self.settings.pages):
            fetched_at = f"2026-01-01T00:00:0{index}Z"
            self.fetched_times.append(fetched_at)
            (self.settings.raw_dir / f"{page.slug}.html").write_text(
                html_by_slug.get(page.slug, make_html(page.name)), encoding="utf-8"
            )
            (self.settings.raw_dir / f"{page.slug}.meta.json").write_text(
                json.dumps(
                    {
                        "url": page.url,
                        "scheme_or_amc": page.name,
                        "fetched_at": fetched_at,
                        "http_status": 200,
                    }
                ),
                encoding="utf-8",
            )

    def test_extract_text_drops_chrome_and_embedded_json(self) -> None:
        text = extract_text(make_html("HDFC Test Fund"))
        self.assertIn("HDFC Test Fund", text)
        self.assertIn("Expense ratio: 0.99%", text)
        for marker in ("COOKIE_MARKER", "NAV_MARKER", "FOOTER_MARKER", "SCRIPT_MARKER"):
            self.assertNotIn(marker, text)

    def test_load_writes_documents_and_successful_corpus(self) -> None:
        self._write_raw({})
        ok = run_load(self.settings)

        self.assertTrue(ok)
        page = self.settings.pages[0]
        document_path = self.settings.documents_dir / f"{page.slug}.json"
        document = json.loads(document_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(document), {"url", "scheme_or_amc", "fetched_at", "text"}
        )
        self.assertEqual(document["url"], page.url)
        self.assertEqual(document["scheme_or_amc"], page.name)
        self.assertEqual(document["fetched_at"], self.fetched_times[0])
        self.assertIn("Expense ratio: 0.99%", document["text"])

        corpus = json.loads(self.settings.corpus_file.read_text(encoding="utf-8"))
        self.assertTrue(corpus["ok"])
        self.assertEqual(corpus["stage"], "load")
        self.assertEqual(corpus["last_updated"], self.fetched_times[0])
        self.assertEqual(len(corpus["documents"]), 5)
        self.assertTrue(all(item["ok"] for item in corpus["documents"]))

    def test_empty_document_fails_and_removes_stale_file(self) -> None:
        page = self.settings.pages[0]
        self.settings = replace(self.settings, min_document_chars=200)
        self._write_raw({page.slug: "<html><body><div class='layout-main'><h1>Short</h1></div></body></html>"})

        stale_path = self.settings.documents_dir / f"{page.slug}.json"
        self.settings.documents_dir.mkdir(parents=True, exist_ok=True)
        stale_path.write_text('{"stale": true}', encoding="utf-8")

        ok = run_load(self.settings)

        self.assertFalse(ok)
        self.assertFalse(stale_path.exists())
        corpus = json.loads(self.settings.corpus_file.read_text(encoding="utf-8"))
        self.assertFalse(corpus["ok"])
        self.assertIsNone(corpus["last_updated"])
        failed = [item for item in corpus["documents"] if item["slug"] == page.slug]
        self.assertFalse(failed[0]["ok"])
        self.assertIn("too short", failed[0]["reason"])

    def test_failed_scrape_corpus_blocks_loading(self) -> None:
        original = {"ok": False, "stage": "scrape", "documents": []}
        self.settings.corpus_file.write_text(json.dumps(original), encoding="utf-8")

        ok = run_load(self.settings)

        self.assertFalse(ok)
        self.assertEqual(list(self.settings.documents_dir.glob("*.json")), [])
        self.assertEqual(
            json.loads(self.settings.corpus_file.read_text(encoding="utf-8")), original
        )


if __name__ == "__main__":
    unittest.main()
