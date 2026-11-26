import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from grow_faq.config import load_settings
from grow_faq.ingest import scrape


class ScrapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_settings = load_settings()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = replace(
            self.base_settings,
            raw_dir=root / "raw",
            corpus_file=root / "corpus.json",
        )
        self.calls: list[str] = []

    def _fake_get(self, statuses: dict[str, int]):
        slugs = {page.url: page.slug for page in self.settings.pages}

        def fake_get(_client: httpx.Client, url: str, **_kwargs: object) -> httpx.Response:
            self.calls.append(url)
            slug = slugs[url]
            return httpx.Response(
                statuses[url],
                content=f"<html>{slug}</html>".encode("utf-8"),
                request=httpx.Request("GET", url),
            )

        return fake_get

    def test_all_200_writes_raw_artifacts_and_successful_corpus(self) -> None:
        statuses = {page.url: 200 for page in self.settings.pages}
        fetched_times = [
            "2026-01-01T00:00:05Z",
            "2026-01-01T00:00:04Z",
            "2026-01-01T00:00:03Z",
            "2026-01-01T00:00:02Z",
            "2026-01-01T00:00:01Z",
        ]

        with (
            patch.object(httpx.Client, "get", new=self._fake_get(statuses)),
            patch.object(scrape, "_utc_now", side_effect=["2026-01-01T00:00:00Z", *fetched_times]),
        ):
            ok = scrape.run_scrape(self.settings)

        self.assertTrue(ok)
        self.assertEqual(self.calls, [page.url for page in self.settings.pages])

        first = self.settings.pages[0]
        html_path = self.settings.raw_dir / f"{first.slug}.html"
        meta_path = self.settings.raw_dir / f"{first.slug}.meta.json"
        self.assertEqual(html_path.read_bytes(), f"<html>{first.slug}</html>".encode("utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(meta), {"url", "scheme_or_amc", "fetched_at", "http_status"}
        )
        self.assertEqual(meta["url"], first.url)
        self.assertEqual(meta["scheme_or_amc"], first.name)
        self.assertEqual(meta["http_status"], 200)

        corpus = json.loads(self.settings.corpus_file.read_text(encoding="utf-8"))
        self.assertTrue(corpus["ok"])
        self.assertEqual(corpus["stage"], "scrape")
        self.assertEqual(corpus["last_updated"], fetched_times[-1])
        self.assertEqual(len(corpus["documents"]), 5)
        self.assertTrue(all(document["ok"] for document in corpus["documents"]))

    def test_one_non_200_marks_corpus_not_ok(self) -> None:
        failing_page = self.settings.pages[-1]
        statuses = {
            page.url: (503 if page.url == failing_page.url else 200)
            for page in self.settings.pages
        }

        with patch.object(httpx.Client, "get", new=self._fake_get(statuses)):
            ok = scrape.run_scrape(self.settings)

        self.assertFalse(ok)
        self.assertEqual(len(self.calls), 5)

        corpus = json.loads(self.settings.corpus_file.read_text(encoding="utf-8"))
        self.assertFalse(corpus["ok"])
        self.assertIsNone(corpus["last_updated"])
        failed = [d for d in corpus["documents"] if d["slug"] == failing_page.slug]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["http_status"], 503)
        self.assertFalse(failed[0]["ok"])

    def test_network_error_is_recorded_as_status_zero(self) -> None:
        failing_page = self.settings.pages[0]

        def fake_get(_client: httpx.Client, url: str, **_kwargs: object) -> httpx.Response:
            self.calls.append(url)
            if url == failing_page.url:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(
                200,
                content=b"<html>ok</html>",
                request=httpx.Request("GET", url),
            )

        with patch.object(httpx.Client, "get", new=fake_get):
            ok = scrape.run_scrape(self.settings)

        self.assertFalse(ok)
        meta = json.loads(
            (self.settings.raw_dir / f"{failing_page.slug}.meta.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(meta["http_status"], 0)
        self.assertEqual(
            (self.settings.raw_dir / f"{failing_page.slug}.html").read_bytes(), b""
        )


if __name__ == "__main__":
    unittest.main()
