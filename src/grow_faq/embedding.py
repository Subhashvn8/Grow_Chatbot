"""Shared text encoding for ingestion and querying.

Local runs use sentence-transformers. Serverless runs can select a hosted
OpenAI embeddings client with ``EMBEDDING_PROVIDER=openai``, avoiding a multi-gigabyte
``torch`` bundle. Both paths return unit-length vectors.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def model_cache_dir() -> str | None:
    """Where local embedding models are cached.

    ``MODEL_CACHE_DIR`` wins. On Render (``RENDER`` is set during build and
    runtime) the default is inside the project directory, because only that
    directory is carried from the build to the running service, so a model
    downloaded at build time is not fetched again on the first question.
    Elsewhere the sentence-transformers default cache is used.
    """
    explicit = (os.getenv("MODEL_CACHE_DIR") or "").strip()
    if explicit:
        return explicit
    if os.getenv("RENDER"):
        return str(PROJECT_ROOT / ".cache" / "models")
    return None


@lru_cache(maxsize=4)
def load_encoder(model_name: str) -> Any:
    """Load one sentence-transformer model per process."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local embedding. "
            "Set EMBEDDING_PROVIDER=openai for serverless deployments."
        ) from exc
    return SentenceTransformer(model_name, cache_folder=model_cache_dir())


@lru_cache(maxsize=4)
def _openai_client(api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai is required for hosted embeddings. "
            "Install the Vercel dependencies with "
            "'pip install -r requirements-vercel.txt'."
        ) from exc
    return OpenAI(api_key=api_key)


def _openai_vectors(model_name: str, texts: Sequence[str]) -> list[list[float]]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for hosted embeddings")
    response = _openai_client(api_key).embeddings.create(
        model=model_name,
        input=list(texts),
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    return [[float(value) for value in item.embedding] for item in ordered]


def normalize_vectors(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return L2-normalized copies of the given vectors."""
    normalized: list[list[float]] = []
    for vector in vectors:
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        normalized.append([value / norm for value in values] if norm else values)
    return normalized


def encode_texts(
    model_name: str,
    texts: Sequence[str],
    batch_size: int,
) -> list[list[float]]:
    """Encode text with the configured model and return unit-length vectors."""
    if not texts:
        return []
    if (os.getenv("EMBEDDING_PROVIDER") or "local").strip().lower() == "openai":
        return normalize_vectors(_openai_vectors(model_name, texts))
    model = load_encoder(model_name)
    raw = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if getattr(raw, "ndim", None) == 1:
        raw = [raw]
    return normalize_vectors([[float(value) for value in row] for row in raw])


def main() -> int:
    """Download the configured local embedding model (Render build step)."""
    from grow_faq.config import load_settings

    settings = load_settings()
    load_encoder(settings.embedding_model)
    print(
        f"embedding model ready: {settings.embedding_model} "
        f"(cache: {model_cache_dir() or 'default'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
