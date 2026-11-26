import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from grow_faq.config import load_settings
from grow_faq.embedding import encode_texts
from grow_faq.query.answer import Answer
from grow_faq.query.trace import write_trace
from grow_faq.vector_http import (
    ChromaHttpClient,
    ChromaHttpError,
    CollectionNotFoundError,
)
from grow_faq.vector_store import new_client


class ServerlessTests(unittest.TestCase):
    def test_environment_overrides_are_applied(self) -> None:
        values = {
            "LLM_MODEL": "test-llm",
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "CHROMA_HOST": "chroma.example.com",
            "CHROMA_PORT": "443",
            "CHROMA_SSL": "true",
            "CHROMA_COLLECTION": "remote_facts",
            "CHROMA_TENANT": "tenant-1",
            "CHROMA_DATABASE": "faq",
            "CHROMA_API_KEY": "client-secret",
            "TOP_K": "2",
            "TRACE_ENABLED": "false",
        }
        with patch.dict(os.environ, values):
            settings = load_settings()

        self.assertEqual(settings.llm_model, "test-llm")
        self.assertEqual(settings.embedding_provider, "openai")
        self.assertEqual(settings.embedding_model, "text-embedding-3-small")
        self.assertEqual(settings.vector_host, "chroma.example.com")
        self.assertEqual(settings.vector_port, 443)
        self.assertTrue(settings.vector_ssl)
        self.assertEqual(settings.vector_collection, "remote_facts")
        self.assertEqual(settings.vector_tenant, "tenant-1")
        self.assertEqual(settings.vector_database, "faq")
        self.assertEqual(settings.vector_api_key, "client-secret")
        self.assertEqual(settings.top_k, 2)
        self.assertFalse(settings.trace_enabled)

    def test_remote_chroma_client_does_not_import_chromadb(self) -> None:
        values = {
            "CHROMA_HOST": "chroma.example.com",
            "CHROMA_TENANT": "tenant-1",
            "CHROMA_DATABASE": "faq",
            "CHROMA_API_KEY": "client-secret",
        }
        with patch.dict(os.environ, values):
            settings = load_settings()

        with patch("grow_faq.vector_store._import_chromadb") as import_mock:
            client = new_client(settings)

        import_mock.assert_not_called()
        self.assertIsInstance(client, ChromaHttpClient)
        self.assertEqual(client.base_url, "https://chroma.example.com")
        self.assertEqual(client.tenant, "tenant-1")
        self.assertEqual(client.database, "faq")
        self.assertEqual(client.api_key, "client-secret")

    def test_hosted_embeddings_require_a_remote_vector_store(self) -> None:
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai"}):
            settings = load_settings()

        with self.assertRaises(RuntimeError) as caught:
            new_client(settings)

        self.assertIn("CHROMA_HOST", str(caught.exception))

    def test_vercel_requires_a_remote_chroma_host(self) -> None:
        values = {"VERCEL": "1", "CHROMA_HOST": ""}
        with patch.dict(os.environ, values):
            settings = load_settings()
            with self.assertRaises(RuntimeError) as caught:
                new_client(settings)

        self.assertIn("CHROMA_HOST is required on Vercel", str(caught.exception))

    def test_chroma_cloud_requires_explicit_credentials(self) -> None:
        values = {
            "CHROMA_HOST": "api.trychroma.com",
            "CHROMA_TENANT": "tenant-1",
            "CHROMA_DATABASE": "faq",
        }
        with patch.dict(os.environ, values):
            settings = load_settings()

        with self.assertRaises(ChromaHttpError) as caught:
            ChromaHttpClient(settings)

        self.assertIn("CHROMA_API_KEY", str(caught.exception))

    def test_http_collection_uses_chroma_v2_endpoints(self) -> None:
        values = {
            "CHROMA_HOST": "https://chroma.example.com",
            "CHROMA_TENANT": "tenant-1",
            "CHROMA_DATABASE": "faq",
        }
        with patch.dict(os.environ, values):
            settings = load_settings()
        client = ChromaHttpClient(settings)

        collection_data = {
            "id": "col-1",
            "name": "facts",
            "metadata": {"embedding_model": "test-embedding"},
        }
        with patch.object(client, "_request", return_value=collection_data) as request:
            collection = client.get_collection("facts")

        request.assert_called_once_with(
            "GET",
            "/api/v2/tenants/tenant-1/databases/faq/collections/facts",
        )
        self.assertEqual(collection.collection_id, "col-1")
        self.assertEqual(collection.metadata, {"embedding_model": "test-embedding"})

        query_result = {
            "ids": [["chunk-1"]],
            "documents": [["Expense ratio is 0.42%."]],
            "metadatas": [[{"source_url": "https://groww.in/example"}]],
            "distances": [[0.05]],
        }
        with patch.object(client, "_request", return_value=query_result) as request:
            result = collection.query(
                query_embeddings=[[0.1, 0.2]],
                n_results=4,
                include=["documents", "metadatas", "distances"],
                where={"scheme_or_amc": "HDFC Flexi Cap Fund Direct Growth"},
            )

        request.assert_called_once_with(
            "POST",
            "/api/v2/tenants/tenant-1/databases/faq/collections/col-1/query",
            {
                "query_embeddings": [[0.1, 0.2]],
                "n_results": 4,
                "include": ["documents", "metadatas", "distances"],
                "where": {"scheme_or_amc": "HDFC Flexi Cap Fund Direct Growth"},
            },
        )
        self.assertEqual(result, query_result)

    def test_missing_remote_collection_is_distinguishable(self) -> None:
        values = {"CHROMA_HOST": "chroma.example.com"}
        with patch.dict(os.environ, values):
            settings = load_settings()
        client = ChromaHttpClient(settings)

        error = HTTPError("url", 404, "Not Found", {}, None)
        with patch("grow_faq.vector_http.urlopen", side_effect=error):
            with self.assertRaises(CollectionNotFoundError):
                client._request("GET", "/missing")

    def test_hosted_embeddings_are_normalized_and_reordered(self) -> None:
        values = {
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
        }
        data = [
            SimpleNamespace(index=1, embedding=[0.0, 2.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]
        with patch.dict(os.environ, values):
            with patch("grow_faq.embedding._openai_client") as client_mock:
                client_mock.return_value.embeddings.create.return_value.data = data
                vectors = encode_texts("text-embedding-3-small", ["first", "second"], 2)

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        client_mock.return_value.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["first", "second"],
        )

    def test_disabled_tracing_writes_nothing(self) -> None:
        settings = replace(load_settings(), trace_enabled=False)
        answer = Answer(text="Answer.", citation=None, policy="answer")

        with patch("grow_faq.query.trace.Path.mkdir") as mkdir_mock:
            path = write_trace(settings, question="Question?", policy="answer", chunks=(), answer=answer)

        self.assertIsNone(path)
        mkdir_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
