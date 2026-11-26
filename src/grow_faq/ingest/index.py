"""Phase 6: upsert chunks and their vectors into persistent Chroma.

This stage reads only ``chunks.jsonl`` and ``embeddings.jsonl``. It does not
fetch, embed, or call an LLM. Upserting on chunk id makes re-runs replace stale
facts, and ids no longer present in the chunk file are deleted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from grow_faq.config import Settings, load_settings
from grow_faq.vector_store import open_collection

CHUNK_FIELDS = (
    "id",
    "text",
    "source_url",
    "scheme_or_amc",
    "section",
    "field",
    "fetched_at",
)
EMBEDDING_FIELDS = ("id", "model", "dimension", "vector")


def _read_jsonl(path: Path, required: tuple[str, ...], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")

    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{label} line {line_number} is not a JSON object")

            missing = [field for field in required if field not in record]
            if missing:
                raise ValueError(
                    f"{label} line {line_number} is missing {', '.join(missing)}"
                )

            record_id = record["id"]
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{label} line {line_number} has no valid id")
            if record_id in ids:
                raise ValueError(f"duplicate {label} id: {record_id}")
            ids.add(record_id)
            records.append(record)

    if not records:
        raise ValueError(f"{label} file is empty: {path}")
    return records


def _validate_inputs(
    chunks: list[dict[str, Any]],
    embeddings: list[dict[str, Any]],
    expected_model: str,
) -> int:
    chunk_ids = [str(chunk["id"]) for chunk in chunks]
    embedding_ids = [str(record["id"]) for record in embeddings]
    if chunk_ids != embedding_ids:
        raise ValueError("chunk ids and embedding ids do not match exactly")

    dimension: int | None = None
    for record in embeddings:
        record_id = str(record["id"])
        model = record["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"embedding {record_id!r} has no model name")
        if model != expected_model:
            raise ValueError(
                f"embedding model {model!r} does not match config {expected_model!r}; "
                "re-run grow-embed before indexing"
            )

        declared = record["dimension"]
        if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
            raise ValueError(f"embedding {record_id!r} has an invalid dimension")

        vector = record["vector"]
        if not isinstance(vector, list) or len(vector) != declared:
            raise ValueError(
                f"embedding {record_id!r} vector length does not match its dimension"
            )
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"embedding {record_id!r} has a non-numeric vector value")
            if not math.isfinite(float(value)):
                raise ValueError(f"embedding {record_id!r} has a non-finite vector value")

        if dimension is None:
            dimension = declared
        elif declared != dimension:
            raise ValueError("embedding dimensions are inconsistent")

    return dimension or 0


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _metadata(chunk: dict[str, Any], model: str) -> dict[str, str]:
    """Chroma metadata must contain only string values."""
    return {
        "source_url": _string(chunk.get("source_url")),
        "scheme_or_amc": _string(chunk.get("scheme_or_amc")),
        "section": _string(chunk.get("section")),
        "field": _string(chunk.get("field")),
        "fetched_at": _string(chunk.get("fetched_at")),
        "embedding_model": model,
    }


def run_index(settings: Settings, client: Any | None = None) -> bool:
    """Upsert all chunks with their vectors and prune removed ids."""
    try:
        chunks = _read_jsonl(
            settings.chunks_dir / "chunks.jsonl", CHUNK_FIELDS, "chunks"
        )
        embeddings = _read_jsonl(
            settings.embeddings_dir / "embeddings.jsonl", EMBEDDING_FIELDS, "embeddings"
        )
        dimension = _validate_inputs(chunks, embeddings, settings.embedding_model)
    except (OSError, ValueError) as exc:
        print(f"[fail] could not read index inputs: {exc}", file=sys.stderr)
        return False

    ids = [str(chunk["id"]) for chunk in chunks]
    documents = [str(chunk["text"]) for chunk in chunks]
    vectors = [list(record["vector"]) for record in embeddings]
    metadatas = [_metadata(chunk, settings.embedding_model) for chunk in chunks]

    try:
        collection = open_collection(settings, client=client)
        existing_ids = {str(item) for item in collection.get(include=[])["ids"]}
        current_ids = set(ids)
        stale_ids = sorted(existing_ids - current_ids)
        if stale_ids:
            collection.delete(ids=stale_ids)
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )
    except Exception as exc:
        print(f"[fail] indexing into {settings.vector_collection}: {exc}", file=sys.stderr)
        return False

    print(
        f"index: {settings.vector_collection} @ {settings.vector_persist_path} "
        f"({len(ids)} records, dimension {dimension}, {len(stale_ids)} removed)"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--env", type=Path, default=None, help="path to .env")
    args = parser.parse_args(argv)

    settings = load_settings(config_path=args.config, env_path=args.env)
    return 0 if run_index(settings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
