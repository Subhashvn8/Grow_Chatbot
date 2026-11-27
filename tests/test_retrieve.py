import unittest
from unittest.mock import patch

from grow_faq.config import load_settings
from grow_faq.query.retrieve import StaleModelError, retrieve


class FakeCollection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def response(ids, texts=None, fields=None, distances=None):
    texts = texts or [f"text for {item}" for item in ids]
    fields = fields or ["expense_ratio"] * len(ids)
    distances = distances or [0.2] * len(ids)
    return {
        "ids": [ids],
        "documents": [texts],
        "metadatas": [
            [
                {
                    "source_url": f"https://groww.in/mutual-funds/{item}",
                    "scheme_or_amc": "HDFC Large Cap Fund Direct Growth",
                    "section": "fund-details",
                    "field": field,
                }
                for item, field in zip(ids, fields)
            ]
        ],
        "distances": [distances],
    }


class RetrieveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = load_settings()

    def retrieve_with(self, collection, question="question"):
        with (
            patch(
                "grow_faq.query.retrieve.collection_embedding_model",
                return_value=self.settings.embedding_model,
            ),
            patch("grow_faq.query.retrieve.existing_collection", return_value=collection),
            patch(
                "grow_faq.query.retrieve.encode_texts",
                return_value=[[0.1, 0.2, 0.3]],
            ) as encode_mock,
        ):
            chunks = retrieve(question, self.settings, client=object())
        return chunks, collection, encode_mock

    def test_one_scheme_applies_metadata_filter(self) -> None:
        collection = FakeCollection([response(["chunk-a", "chunk-b"])])
        question = "Expense ratio of HDFC Large Cap Fund Direct Growth?"
        with (
            patch(
                "grow_faq.query.retrieve.collection_embedding_model",
                return_value=self.settings.embedding_model,
            ),
            patch("grow_faq.query.retrieve.existing_collection", return_value=collection),
            patch("grow_faq.query.retrieve.encode_texts", return_value=[[0.1]]),
        ):
            chunks = retrieve(question, self.settings, client=object())

        self.assertEqual([chunk.id for chunk in chunks], ["chunk-a", "chunk-b"])
        self.assertEqual(chunks[0].score, 0.8)
        self.assertEqual(chunks[0].field, "expense_ratio")
        self.assertEqual(
            collection.calls[0]["where"],
            {"scheme_or_amc": "HDFC Large Cap Fund Direct Growth"},
        )
        self.assertEqual(collection.calls[0]["n_results"], self.settings.top_k)

    def test_how_to_invest_searches_the_amc_page(self) -> None:
        collection = FakeCollection([response(["chunk-a"])])
        question = "How do I buy HDFC Mid Cap Fund Direct Growth?"
        with (
            patch(
                "grow_faq.query.retrieve.collection_embedding_model",
                return_value=self.settings.embedding_model,
            ),
            patch("grow_faq.query.retrieve.existing_collection", return_value=collection),
            patch("grow_faq.query.retrieve.encode_texts", return_value=[[0.1]]),
        ):
            retrieve(question, self.settings, client=object())

        self.assertEqual(
            collection.calls[0]["where"],
            {"scheme_or_amc": self.settings.pages[0].name},
        )

    def test_two_schemes_do_not_apply_a_filter(self) -> None:
        collection = FakeCollection([response(["chunk-a"])])
        with (
            patch(
                "grow_faq.query.retrieve.collection_embedding_model",
                return_value=self.settings.embedding_model,
            ),
            patch("grow_faq.query.retrieve.existing_collection", return_value=collection),
            patch("grow_faq.query.retrieve.encode_texts", return_value=[[0.1]]),
        ):
            retrieve("Compare Large Cap and Mid Cap", self.settings, client=object())

        self.assertNotIn("where", collection.calls[0])

    def test_empty_filter_result_falls_back_to_whole_collection(self) -> None:
        collection = FakeCollection([response([]), response(["chunk-b"])])
        chunks, _, _ = self.retrieve_with(
            collection, "Expense ratio of HDFC Large Cap Fund Direct Growth?"
        )
        self.assertEqual([chunk.id for chunk in chunks], ["chunk-b"])
        self.assertEqual(len(collection.calls), 2)
        self.assertIn("where", collection.calls[0])
        self.assertNotIn("where", collection.calls[1])

    def test_stale_model_refuses_to_query(self) -> None:
        with (
            patch(
                "grow_faq.query.retrieve.collection_embedding_model",
                return_value="some-other-model",
            ),
            patch(
                "grow_faq.query.retrieve.encode_texts",
                return_value=[[0.1]],
            ) as encode_mock,
        ):
            with self.assertRaises(StaleModelError) as caught:
                retrieve("question", self.settings, client=object())

        message = str(caught.exception)
        self.assertIn("grow-embed", message)
        self.assertIn("grow-index", message)
        encode_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
