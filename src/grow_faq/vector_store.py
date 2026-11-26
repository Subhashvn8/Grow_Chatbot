"""Chroma collection access shared by indexing and retrieval.

Local runs use the persistent client at ``vector_db.persist_path``. Setting
``CHROMA_HOST`` switches retrieval to the lightweight Chroma v2 REST client, so
serverless deployments do not need the heavy ``chromadb`` package. Collection
metadata records the embedding model and distance space so an old collection
cannot be queried with vectors from a different model.
"""

from __future__ import annotations

from typing import Any

from grow_faq.config import Settings

COLLECTION_MODEL_KEY = "embedding_model"
COLLECTION_SPACE_KEY = "hnsw:space"
METRIC_TO_SPACE = {
    "cosine": "cosine",
    "l2": "l2",
    "ip": "ip",
}


def desired_metadata(settings: Settings) -> dict[str, str]:
    """Metadata required on a collection created for these settings."""
    return {
        COLLECTION_SPACE_KEY: METRIC_TO_SPACE[settings.vector_metric],
        COLLECTION_MODEL_KEY: settings.embedding_model,
    }


def _import_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is required for this stage. Install the project "
            "dependencies with 'pip install -e .'."
        ) from exc
    return chromadb


def new_client(settings: Settings) -> Any:
    """Open a remote HTTP client when configured, otherwise local persistent."""
    if settings.vector_host:
        from grow_faq.vector_http import ChromaHttpClient

        return ChromaHttpClient(settings)

    if settings.embedding_provider == "openai":
        raise RuntimeError(
            "CHROMA_HOST is required when EMBEDDING_PROVIDER=openai. Set it to "
            "the remote Chroma server, or use local embeddings and chromadb."
        )

    chromadb = _import_chromadb()
    chroma_settings = chromadb.config.Settings(anonymized_telemetry=False)
    return chromadb.PersistentClient(
        path=str(settings.vector_persist_path),
        settings=chroma_settings,
    )


def _collection_metadata(collection: Any) -> dict[str, Any]:
    return dict(collection.metadata or {})


def open_collection(settings: Settings, client: Any | None = None) -> Any:
    """Return the configured collection, rebuilding it if its config changed."""
    client = client if client is not None else new_client(settings)
    metadata = desired_metadata(settings)

    collection = client.get_or_create_collection(
        name=settings.vector_collection,
        metadata=metadata,
    )
    existing = _collection_metadata(collection)
    if any(existing.get(key) != value for key, value in metadata.items()):
        client.delete_collection(name=settings.vector_collection)
        collection = client.get_or_create_collection(
            name=settings.vector_collection,
            metadata=metadata,
        )
    return collection


def existing_collection(settings: Settings, client: Any | None = None) -> Any | None:
    """Return the collection if it exists, without creating or rebuilding it."""
    from grow_faq.vector_http import CollectionNotFoundError

    client = client if client is not None else new_client(settings)
    try:
        return client.get_collection(name=settings.vector_collection)
    except CollectionNotFoundError:
        return None
    except Exception:
        if getattr(client, "is_http", False):
            raise
        return None


def collection_embedding_model(settings: Settings, client: Any | None = None) -> str | None:
    """Return the model stored on the existing collection, if any."""
    collection = existing_collection(settings, client=client)
    if collection is None:
        return None
    value = _collection_metadata(collection).get(COLLECTION_MODEL_KEY)
    return str(value) if value is not None else None
