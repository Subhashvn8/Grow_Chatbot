"""Phase 5: embed chunk text with the configured sentence-transformer.

This stage reads ``chunks.jsonl`` and writes ``embeddings.jsonl``. It does not
call Chroma and does not make any LLM request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from grow_faq.config import Settings, load_settings
from grow_faq.embedding import encode_texts, normalize_vectors


def _read_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(chunk, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            chunk_id = chunk.get("id")
            text = chunk.get("text")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"line {line_number} has no chunk id")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"chunk {chunk_id!r} has no text")
            if chunk_id in ids:
                raise ValueError(f"duplicate chunk id: {chunk_id}")
            ids.add(chunk_id)
            chunks.append(chunk)

    if not chunks:
        raise ValueError("chunks.jsonl is empty")
    return chunks


def _write_embeddings(
    path: Path,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    model_name: str,
) -> None:
    dimension = len(vectors[0])
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk, vector in zip(chunks, vectors):
            record = {
                "id": chunk["id"],
                "model": model_name,
                "dimension": dimension,
                "vector": vector,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_embed(settings: Settings) -> bool:
    """Encode every chunk and persist one embedding record per chunk."""
    chunks_path = settings.chunks_dir / "chunks.jsonl"
    try:
        chunks = _read_chunks(chunks_path)
    except (OSError, ValueError) as exc:
        print(f"[fail] could not read {chunks_path}: {exc}", file=sys.stderr)
        return False

    try:
        vectors = normalize_vectors(
            encode_texts(
                settings.embedding_model,
                [str(chunk["text"]) for chunk in chunks],
                settings.embedding_batch_size,
            )
        )
    except Exception as exc:
        print(f"[fail] embedding with {settings.embedding_model}: {exc}", file=sys.stderr)
        return False

    if len(vectors) != len(chunks):
        print(
            f"[fail] expected {len(chunks)} vectors, got {len(vectors)}",
            file=sys.stderr,
        )
        return False

    dimension = len(vectors[0]) if vectors else 0
    if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
        print("[fail] embedding dimensions are inconsistent", file=sys.stderr)
        return False

    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.embeddings_dir / "embeddings.jsonl"
    _write_embeddings(output_path, chunks, vectors, settings.embedding_model)
    print(
        f"embeddings: {output_path} "
        f"({len(vectors)} vectors, dimension {dimension}, model {settings.embedding_model})"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--env", type=Path, default=None, help="path to .env")
    args = parser.parse_args(argv)

    settings = load_settings(config_path=args.config, env_path=args.env)
    return 0 if run_embed(settings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
