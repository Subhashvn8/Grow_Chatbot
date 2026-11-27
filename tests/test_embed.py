import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from grow_faq.config import load_settings
from grow_faq.embedding import PROJECT_ROOT, model_cache_dir, normalize_vectors
from grow_faq.ingest.embed import run_embed


class EmbedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_settings = load_settings()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = replace(
            self.base_settings,
            chunks_dir=root / "chunks",
            embeddings_dir=root / "embeddings",
        )
        self.settings.chunks_dir.mkdir(parents=True, exist_ok=True)

    def _write_chunks(self, records: list[dict[str, object]]) -> None:
        path = self.settings.chunks_dir / "chunks.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def _embeddings(self) -> list[dict[str, object]]:
        path = self.settings.embeddings_dir / "embeddings.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def test_normalize_vectors_produces_unit_length(self) -> None:
        normalized = normalize_vectors([[3.0, 4.0], [0.0, 0.0]])
        self.assertAlmostEqual(normalized[0][0], 0.6)
        self.assertAlmostEqual(normalized[0][1], 0.8)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in normalized[0])), 1.0)
        self.assertEqual(normalized[1], [0.0, 0.0])

    def test_embed_writes_one_normalized_record_per_chunk_in_order(self) -> None:
        chunks = [
            {"id": "large-cap-expense", "text": "Expense ratio: 0.99%"},
            {"id": "large-cap-benchmark", "text": "Benchmark: NIFTY 100"},
        ]
        self._write_chunks(chunks)
        calls: list[tuple[str, list[str], int]] = []

        def fake_encode(model: str, texts: list[str], batch_size: int) -> list[list[float]]:
            calls.append((model, list(texts), batch_size))
            return [[3.0, 4.0], [1.0, 0.0]]

        with patch("grow_faq.ingest.embed.encode_texts", side_effect=fake_encode):
            ok = run_embed(self.settings)

        self.assertTrue(ok)
        self.assertEqual(
            calls,
            [
                (
                    self.settings.embedding_model,
                    ["Expense ratio: 0.99%", "Benchmark: NIFTY 100"],
                    self.settings.embedding_batch_size,
                )
            ],
        )
        records = self._embeddings()
        self.assertEqual([record["id"] for record in records], [chunk["id"] for chunk in chunks])
        for record in records:
            self.assertEqual(record["model"], self.settings.embedding_model)
            self.assertEqual(record["dimension"], 2)
            vector = record["vector"]
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)
        self.assertAlmostEqual(records[0]["vector"][0], 0.6)

    def test_duplicate_chunk_ids_fail_before_embedding(self) -> None:
        self._write_chunks(
            [
                {"id": "same", "text": "one"},
                {"id": "same", "text": "two"},
            ]
        )
        with patch("grow_faq.ingest.embed.encode_texts") as encode:
            ok = run_embed(self.settings)

        self.assertFalse(ok)
        encode.assert_not_called()
        self.assertFalse((self.settings.embeddings_dir / "embeddings.jsonl").exists())

    def test_vector_count_mismatch_fails(self) -> None:
        self._write_chunks([{"id": "one", "text": "one"}, {"id": "two", "text": "two"}])
        with patch(
            "grow_faq.ingest.embed.encode_texts",
            return_value=[[1.0, 0.0]],
        ):
            ok = run_embed(self.settings)

        self.assertFalse(ok)
        self.assertFalse((self.settings.embeddings_dir / "embeddings.jsonl").exists())


class ModelCacheDirTests(unittest.TestCase):
    def test_default_cache_outside_render(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(model_cache_dir())

    def test_render_caches_inside_the_project(self) -> None:
        with patch.dict("os.environ", {"RENDER": "true"}, clear=True):
            self.assertEqual(
                model_cache_dir(), str(PROJECT_ROOT / ".cache" / "models")
            )

    def test_explicit_cache_dir_wins(self) -> None:
        with patch.dict(
            "os.environ", {"RENDER": "true", "MODEL_CACHE_DIR": "/tmp/m"}, clear=True
        ):
            self.assertEqual(model_cache_dir(), "/tmp/m")


if __name__ == "__main__":
    unittest.main()
