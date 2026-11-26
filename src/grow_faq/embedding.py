"""Shared sentence-transformer encoding for ingestion and querying.

Both ``grow-embed`` and the later retriever call these functions with the model
name from config, so the same vectors are produced on both paths.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4)
def load_encoder(model_name: str) -> Any:
    """Load one sentence-transformer model per process."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for embedding. "
            "Install the project dependencies with 'pip install -e .'."
        ) from exc
    return SentenceTransformer(model_name)


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
