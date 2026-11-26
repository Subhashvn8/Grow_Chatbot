"""Write one trace per successful non-PII question."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from grow_faq.config import Settings
from grow_faq.query.answer import Answer
from grow_faq.query.retrieve import RetrievedChunk


def write_trace(
    settings: Settings,
    *,
    question: str,
    policy: str,
    chunks: tuple[RetrievedChunk, ...],
    answer: Answer,
) -> Path:
    """Persist retrieval evidence and the final answer under ``paths.traces_dir``."""
    settings.traces_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    path = settings.traces_dir / f"{stamp}.json"

    trace = {
        "question": question,
        "policy": policy,
        "chunk_ids": [chunk.id for chunk in chunks],
        "scores": [chunk.score for chunk in chunks],
        "source_urls": [chunk.source_url for chunk in chunks],
        "answer": answer.text,
        "citation": answer.citation,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }
    path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
