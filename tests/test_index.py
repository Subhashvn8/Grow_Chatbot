import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from grow_faq.config import load_settings
from grow_faq.ingest.index import run_index
from grow_faq.vector_store import open_collection


class FakeCollection:
    def __init__(self, metadata=None, ids=()):
        self.metadata = metadata
        self.ids = list(ids)
        self.deleted_ids = None
        self.upserted = None

    def get(self, include=None):
        return {"ids": list(self.ids)}

    def delete(self, ids):
        self.deleted_ids = list(ids)
        self.ids = [item for item in self.ids if item not in set(ids)]

    def upsert(self, ids, documents, embeddings, metadatas):
        self.upserted = {
            "ids": list(ids),
            "documents": list(documents),
            "embeddings": [list(vector) for vector in embeddings],
            "metadatas": [dict(metadata) for metadata in metadatas],
        }


class FakeClient:
    def __init__(self, metadata=None, ids=()):
        self.collection = FakeCollection(metadata=metadata, ids=ids)
        self.deleted_names = []
        self.created = []

    def get_or_create_collection(self, name, metadata=None):
        self.created.append(dict(metadata or {}))
        if self.collection is None:
            self.collection = FakeCollection(metadata=metadata)
        return self.collection

    def delete_collection(self, name):
        self.deleted_names.append(name)
        self.collection = None


class IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_settings = load_settings()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = replace(
            self.base_settings,
            chunks_dir=root / "chunks",
            embeddings_dir=root / "embeddings",
            vector_persist_path=root / "chroma",
            vector_collection="hdfc_mf_facts",
        )
        self.settings.chunks_dir.mkdir(parents=True)
        self.settings.embeddings_dir.mkdir(parents=True)

    def _write_inputs(self, chunks, embeddings) -> None:
        (self.settings.chunks_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(item) for item in chunks) + "\n", encoding="utf-8"
        )
        (self.settings.embeddings_dir / "embeddings.jsonl").write_text(
            "\n".join(json.dumps(item) for item in embeddings) + "\n", encoding="utf-8"
        )

    def _chunk(self, chunk_id: str, text: str) -> dict[str, object]:
        return {
            "id": chunk_id,
            "text": text,
            "source_url": f"https://groww.in/{chunk_id}",
            "scheme_or_amc": "HDFC Large Cap Fund Direct Growth",
            "section": "fund-details",
            "field": "expense_ratio",
            "fetched_at": "2026-01-01T00:00:00Z",
        }

    def _embedding(self, chunk_id: str, vector: list[float]) -> dict[str, object]:
        return {
            "id": chunk_id,
            "model": self.settings.embedding_model,
            "dimension": len(vector),
            "vector": vector,
        }

    def test_index_upserts_current_chunks_and_deletes_stale_ids(self) -> None:
        self._write_inputs(
            [
                self._chunk("c1", "Expense ratio of scheme A: 0.99%"),
                self._chunk("c2", "Expense ratio of scheme B: 1.18%"),
            ],
            [self._embedding("c1", [0.6, 0.8]), self._embedding("c2", [1.0, 0.0])],
        )
        collection = FakeCollection(ids=["stale", "c1"])

        with patch("grow_faq.ingest.index.open_collection", return_value=collection):
            ok = run_index(self.settings)

        self.assertTrue(ok)
        self.assertEqual(collection.deleted_ids, ["stale"])
        self.assertIsNotNone(collection.upserted)
        upserted = collection.upserted
        assert upserted is not None
        self.assertEqual(upserted["ids"], ["c1", "c2"])
        self.assertEqual(upserted["embeddings"], [[0.6, 0.8], [1.0, 0.0]])
        self.assertEqual(
            upserted["documents"],
            [
                "Expense ratio of scheme A: 0.99%",
                "Expense ratio of scheme B: 1.18%",
            ],
        )
        for metadata in upserted["metadatas"]:
            self.assertEqual(
                set(metadata),
                {
                    "source_url",
                    "scheme_or_amc",
                    "section",
                    "field",
                    "fetched_at",
                    "embedding_model",
                },
            )
            self.assertTrue(all(isinstance(value, str) for value in metadata.values()))
            self.assertEqual(metadata["embedding_model"], self.settings.embedding_model)

    def test_mismatched_embedding_model_blocks_indexing(self) -> None:
        embedding = self._embedding("c1", [1.0, 0.0])
        embedding["model"] = "a/different-model"
        self._write_inputs([self._chunk("c1", "A fact")], [embedding])

        with patch("grow_faq.ingest.index.open_collection") as open_collection_mock:
            ok = run_index(self.settings)

        self.assertFalse(ok)
        open_collection_mock.assert_not_called()

    def test_chunk_and_embedding_ids_must_match_exactly(self) -> None:
        self._write_inputs(
            [self._chunk("c1", "A fact"), self._chunk("c2", "Another fact")],
            [self._embedding("c1", [1.0, 0.0]), self._embedding("c3", [0.0, 1.0])],
        )

        with patch("grow_faq.ingest.index.open_collection") as open_collection_mock:
            ok = run_index(self.settings)

        self.assertFalse(ok)
        open_collection_mock.assert_not_called()

    def test_invalid_vector_length_blocks_indexing(self) -> None:
        self._write_inputs(
            [self._chunk("c1", "A fact")],
            [
                {
                    "id": "c1",
                    "model": self.settings.embedding_model,
                    "dimension": 3,
                    "vector": [1.0, 0.0],
                }
            ],
        )

        with patch("grow_faq.ingest.index.open_collection") as open_collection_mock:
            ok = run_index(self.settings)

        self.assertFalse(ok)
        open_collection_mock.assert_not_called()

    def test_collection_is_rebuilt_when_stored_model_changes(self) -> None:
        client = FakeClient(
            metadata={
                "embedding_model": "an/old-model",
                "hnsw:space": "cosine",
            }
        )

        collection = open_collection(self.settings, client=client)

        self.assertEqual(client.deleted_names, ["hdfc_mf_facts"])
        self.assertEqual(
            collection.metadata,
            {
                "embedding_model": self.settings.embedding_model,
                "hnsw:space": "cosine",
            },
        )


if __name__ == "__main__":
    unittest.main()
