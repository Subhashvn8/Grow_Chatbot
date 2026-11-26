import unittest
from unittest.mock import patch

from grow_faq.config import load_settings
from grow_faq.ingest.ingest import run_ingest


def stage_result(name: str, result: bool, calls: list[str]):
    def stage(settings):
        calls.append(name)
        return result

    return stage


class IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_settings()

    def test_failed_scrape_stops_pipeline(self) -> None:
        calls: list[str] = []
        with (
            patch(
                "grow_faq.ingest.ingest.scrape_stage.run_scrape",
                side_effect=stage_result("scrape", False, calls),
            ),
            patch(
                "grow_faq.ingest.ingest.load_stage.run_load",
                side_effect=stage_result("load", True, calls),
            ) as load_mock,
            patch(
                "grow_faq.ingest.ingest.chunk_stage.run_chunk",
                side_effect=stage_result("chunk", True, calls),
            ) as chunk_mock,
            patch(
                "grow_faq.ingest.ingest.embed_stage.run_embed",
                side_effect=stage_result("embed", True, calls),
            ) as embed_mock,
            patch(
                "grow_faq.ingest.ingest.index_stage.run_index",
                side_effect=stage_result("index", True, calls),
            ) as index_mock,
        ):
            ok = run_ingest(self.settings)

        self.assertFalse(ok)
        self.assertEqual(calls, ["scrape"])
        load_mock.assert_not_called()
        chunk_mock.assert_not_called()
        embed_mock.assert_not_called()
        index_mock.assert_not_called()

    def test_failed_load_stops_pipeline(self) -> None:
        calls: list[str] = []
        with (
            patch(
                "grow_faq.ingest.ingest.scrape_stage.run_scrape",
                side_effect=stage_result("scrape", True, calls),
            ),
            patch(
                "grow_faq.ingest.ingest.load_stage.run_load",
                side_effect=stage_result("load", False, calls),
            ),
            patch(
                "grow_faq.ingest.ingest.chunk_stage.run_chunk",
                side_effect=stage_result("chunk", True, calls),
            ) as chunk_mock,
            patch(
                "grow_faq.ingest.ingest.embed_stage.run_embed",
                side_effect=stage_result("embed", True, calls),
            ) as embed_mock,
            patch(
                "grow_faq.ingest.ingest.index_stage.run_index",
                side_effect=stage_result("index", True, calls),
            ) as index_mock,
        ):
            ok = run_ingest(self.settings)

        self.assertFalse(ok)
        self.assertEqual(calls, ["scrape", "load"])
        chunk_mock.assert_not_called()
        embed_mock.assert_not_called()
        index_mock.assert_not_called()

    def test_success_runs_stages_in_order_and_passes_client_to_index(self) -> None:
        calls: list[str] = []
        fake_client = object()
        seen_clients: list[object] = []

        def index_stage(settings, client=None):
            calls.append("index")
            seen_clients.append(client)
            return True

        with (
            patch(
                "grow_faq.ingest.ingest.scrape_stage.run_scrape",
                side_effect=stage_result("scrape", True, calls),
            ),
            patch(
                "grow_faq.ingest.ingest.load_stage.run_load",
                side_effect=stage_result("load", True, calls),
            ),
            patch(
                "grow_faq.ingest.ingest.chunk_stage.run_chunk",
                side_effect=stage_result("chunk", True, calls),
            ),
            patch(
                "grow_faq.ingest.ingest.embed_stage.run_embed",
                side_effect=stage_result("embed", True, calls),
            ),
            patch(
                "grow_faq.ingest.ingest.index_stage.run_index",
                side_effect=index_stage,
            ),
        ):
            ok = run_ingest(self.settings, client=fake_client)

        self.assertTrue(ok)
        self.assertEqual(calls, ["scrape", "load", "chunk", "embed", "index"])
        self.assertEqual(seen_clients, [fake_client])


if __name__ == "__main__":
    unittest.main()
