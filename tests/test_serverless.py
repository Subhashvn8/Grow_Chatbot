import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from grow_faq.config import load_settings
from grow_faq.embedding import encode_texts
from grow_faq.query.answer import Answer
from grow_faq.query.trace import write_trace
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

    def test_remote_chroma_client_is_used_when_host_is_set(self) -> None:
        values = {
            "CHROMA_HOST": "chroma.example.com",
            "CHROMA_API_KEY": "client-secret",
        }
        with patch.dict(os.environ, values):
            settings = load_settings()

        with patch("grow_faq.vector_store._import_chromadb") as import_mock:
            chromadb = import_mock.return_value
            client = new_client(settings)

        kwargs = chromadb.HttpClient.call_args.kwargs
        self.assertIs(client, chromadb.HttpClient.return_value)
        self.assertEqual(kwargs["host"], "chroma.example.com")
        self.assertEqual(kwargs["port"], 443)
        self.assertTrue(kwargs["ssl"])
        self.assertEqual(kwargs["headers"], {"X-Chroma-Token": "client-secret"})
        chromadb.PersistentClient.assert_not_called()

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
