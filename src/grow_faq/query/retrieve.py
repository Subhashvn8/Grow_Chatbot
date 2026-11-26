"""Embed a question and retrieve grounded chunks from the persistent index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grow_faq.config import Settings
from grow_faq.embedding import encode_texts
from grow_faq.query.schemes import single_scheme
from grow_faq.vector_store import (
    collection_embedding_model,
    existing_collection,
    new_client,
)


class StaleModelError(RuntimeError):
    """The index was built with a different embedding model (or is missing)."""


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    score: float
    source_url: str
    scheme_or_amc: str
    section: str
    field: str

    def as_dict(self) -> dict[str, str | float]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "source_url": self.source_url,
            "scheme_or_amc": self.scheme_or_amc,
            "section": self.section,
            "field": self.field,
        }


def check_model(settings: Settings, client: Any | None = None) -> None:
    """Refuse to query an index built with another model."""
    stored = collection_embedding_model(settings, client=client)
    if stored == settings.embedding_model:
        return
    if stored is None:
        raise StaleModelError(
            "The vector index is missing. Run grow-embed and grow-index before "
            "asking questions."
        )
    raise StaleModelError(
        f"The vector index uses embedding model {stored!r}, but config expects "
        f"{settings.embedding_model!r}. Run grow-embed and grow-index before "
        "asking questions."
    )


def distance_to_score(distance: float, metric: str) -> float:
    """Convert a Chroma distance into a similarity-like score for traces."""
    if metric == "cosine":
        return 1.0 - distance
    if metric == "ip":
        return -distance
    return 1.0 / (1.0 + max(distance, 0.0))


def _query(
    collection: Any,
    vector: list[float],
    top_k: int,
    where: dict[str, str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "query_embeddings": [vector],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    return collection.query(**kwargs)


def _chunks(result: dict[str, Any], metric: str) -> list[RetrievedChunk]:
    ids_by_query = result.get("ids") or []
    if not ids_by_query or not ids_by_query[0]:
        return []

    def rows(key: str) -> list[Any]:
        values = result.get(key) or []
        return values[0] if values and isinstance(values[0], list) else []

    ids = ids_by_query[0]
    documents = rows("documents")
    metadatas = rows("metadatas")
    distances = rows("distances")
    chunks: list[RetrievedChunk] = []

    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        document = documents[index] if index < len(documents) else ""
        distance = distances[index] if index < len(distances) else 0.0
        try:
            score = distance_to_score(float(distance), metric)
        except (TypeError, ValueError):
            score = 0.0
        chunks.append(
            RetrievedChunk(
                id=str(chunk_id),
                text=str(document),
                score=score,
                source_url=str(metadata.get("source_url", "")),
                scheme_or_amc=str(metadata.get("scheme_or_amc", "")),
                section=str(metadata.get("section", "")),
                field=str(metadata.get("field", "")),
            )
        )
    return chunks


def retrieve(
    question: str,
    settings: Settings,
    client: Any | None = None,
) -> list[RetrievedChunk]:
    """Return ordered, grounded chunks for ``question``."""
    client = client if client is not None else new_client(settings)
    check_model(settings, client=client)
    collection = existing_collection(settings, client=client)
    if collection is None:
        raise StaleModelError("The vector collection is missing. Run grow-index.")

    vector = encode_texts(
        settings.embedding_model,
        [question],
        settings.embedding_batch_size,
    )[0]

    scheme = single_scheme(question, settings)
    where = {"scheme_or_amc": scheme} if scheme else None
    chunks = _chunks(_query(collection, vector, settings.top_k, where), settings.vector_metric)
    if where and not chunks:
        chunks = _chunks(_query(collection, vector, settings.top_k), settings.vector_metric)
    return chunks
