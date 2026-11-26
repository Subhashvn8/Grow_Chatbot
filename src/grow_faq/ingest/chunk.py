"""Phase 4: section-and-field chunks from loaded documents.

This stage reads ``data/documents/`` only. It never fetches, embeds, or calls an
LLM. Every chunk belongs to exactly one page's ``scheme_or_amc``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from grow_faq.config import Settings, load_settings

MAX_FACTS_PER_FIELD = 3

FIELD_META = {
    "expense_ratio": ("Expense ratio of {scheme}: {value}", "fund-details"),
    "exit_load": ("Exit load for {scheme}: {value}", "exit-load"),
    "minimum_sip": ("Minimum SIP for {scheme}: {value}", "fund-details"),
    "lock_in": ("Lock-in for {scheme}: {value}", "scheme-overview"),
    "riskometer": ("Riskometer for {scheme}: {value}", "fund-details"),
    "benchmark": ("Benchmark of {scheme}: {value}", "investment-objective"),
}

EXACT_FACT_PATTERNS = (
    ("expense_ratio", re.compile(r"^Expense ratio\s*[:\-]?\s*(.+)$", re.I)),
    (
        "minimum_sip",
        re.compile(
            r"^(?:Min\. for SIP|Minimum SIP Investment(?: is set to)?)\s*[:\-]?\s*(.+)$",
            re.I,
        ),
    ),
    ("exit_load", re.compile(r"^Exit load of\s+(.+)$", re.I)),
    ("benchmark", re.compile(r"^Fund benchmark\s*[:\-]\s*(.+)$", re.I)),
)

SEARCH_FACT_PATTERNS = (
    ("lock_in", re.compile(r"\b\d+\s*(?:Y|years?|months?)\s+Lock-in\b", re.I)),
    (
        "riskometer",
        re.compile(
            r"\b(?:Very High|High|Moderately High|Moderate|Moderately Low|Low|Very Low)\s+Risk\b",
            re.I,
        ),
    ),
)

SKIP_SECTION_TERMS = (
    "return",
    "ranking",
    "holdings",
    "calculator",
    "minimum-investments",
    "fund-management",
    "compare-similar-funds",
    "key-information",
    "why-invest",
    "list-of-hdfc",
    "tax-implication",
)

MAJOR_SECTION_TERMS = (
    "about",
    "investment-objective",
    "fund-house",
    "how-to-invest",
    "statement",
    "exit-load",
    "minimum-investments",
    "hdfc-mutual-fund",
)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_url: str
    scheme_or_amc: str
    section: str
    field: str
    fetched_at: str

    def to_record(self) -> dict[str, str]:
        return asdict(self)


def _clean_value(value: str) -> str:
    return value.strip().strip(".; ").strip()


def _canonical_value(field: str, value: str) -> str:
    value = _clean_value(value)
    if field == "riskometer":
        return " ".join(word.capitalize() for word in value.lower().split())
    return value


def _extract_facts(line: str) -> list[tuple[str, str]]:
    for field, pattern in EXACT_FACT_PATTERNS:
        match = pattern.match(line)
        if match:
            return [(field, _canonical_value(field, match.group(1)))]

    found: list[tuple[str, str]] = []
    for field, pattern in SEARCH_FACT_PATTERNS:
        match = pattern.search(line)
        if match:
            found.append((field, _canonical_value(field, match.group(0))))
    return found


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _is_heading(line: str) -> bool:
    return (
        0 < len(line) <= 90
        and ":" not in line
        and not line.startswith("Step ")
        and (line[0].isupper() or line[0].isdigit())
    )


def _is_short_label_value(line: str) -> bool:
    return ":" in line and len(line) <= 120 and not line.startswith("Step ")


def _should_skip_section(heading: str) -> bool:
    slug = _slugify(heading)
    return any(term in slug for term in SKIP_SECTION_TERMS)


def _is_major_heading(heading: str) -> bool:
    slug = _slugify(heading)
    return any(term in slug for term in MAJOR_SECTION_TERMS)


def _iter_sections(lines: list[str], scheme_name: str) -> Iterator[tuple[str, list[str]]]:
    """Group prose lines under their nearest heading, skipping UI/list sections."""
    heading = "overview"
    body: list[str] = []
    skip_section = False
    is_amc = scheme_name == "HDFC Mutual Fund"
    skipping_amc_list = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_amc:
            if re.match(
                r"^(?:List of HDFC Mutual Fund|Let's have a closer look|Now let us jump)",
                line,
                re.I,
            ):
                if body:
                    yield heading, body
                    body = []
                skipping_amc_list = True
                heading = line
                continue
            if skipping_amc_list and re.match(
                r"^(?:Key information about HDFC Mutual Fund|How to invest in HDFC Mutual Funds\?)",
                line,
                re.I,
            ):
                skipping_amc_list = False

        if _extract_facts(line):
            continue

        if _is_heading(line):
            if body:
                yield heading, body
                body = []
            if skipping_amc_list or (skip_section and not _is_major_heading(line)):
                heading = line
                continue
            heading = line
            skip_section = _should_skip_section(heading)
            continue

        if skipping_amc_list or skip_section:
            continue
        if _is_short_label_value(line) or (len(line) < 25 and not line.startswith("Step ")):
            continue
        body.append(line)

    if body and not skip_section and not skipping_amc_list:
        yield heading, body


def _prose_field(heading: str, scheme_slug: str) -> str:
    slug = _slugify(heading)
    if "statement" in slug:
        return "statement_download"
    if "how-to-invest" in slug:
        return "how_to_invest"
    if slug.startswith("about") or slug in {"investment-objective", scheme_slug, "hdfc-mutual-fund"}:
        return "overview"
    return slug


def _split_prose(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    parts: list[str] = []
    step = max_words - overlap_words
    start = 0
    while start < len(words):
        parts.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
        start += step
    return parts


def _add_chunk(chunks: list[Chunk], seen_ids: set[str], chunk: Chunk) -> None:
    candidate = chunk.id
    suffix = 2
    while candidate in seen_ids:
        candidate = f"{chunk.id}-{suffix}"
        suffix += 1
    seen_ids.add(candidate)
    chunks.append(replace(chunk, id=candidate))


def _read_corpus(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n")


def run_chunk(settings: Settings) -> bool:
    """Chunk every loaded document. Return False on a failed or empty load."""
    corpus = _read_corpus(settings.corpus_file)
    if not corpus or corpus.get("stage") != "load" or not corpus.get("ok"):
        print("load corpus is not ok; run grow-load before grow-chunk", file=sys.stderr)
        return False

    document_paths = sorted(settings.documents_dir.glob("*.json"))
    if not document_paths:
        print(f"no loaded documents in {settings.documents_dir}", file=sys.stderr)
        return False

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()

    for document_path in document_paths:
        try:
            document: Any = json.loads(document_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[fail] {document_path.name}: {exc}", file=sys.stderr)
            return False
        if not isinstance(document, dict) or not all(
            key in document for key in ("url", "scheme_or_amc", "fetched_at", "text")
        ):
            print(f"[fail] {document_path.name}: missing required fields", file=sys.stderr)
            return False

        slug = document_path.stem
        scheme = str(document["scheme_or_amc"])
        source_url = str(document["url"])
        fetched_at = str(document["fetched_at"])
        lines = str(document["text"]).splitlines()
        is_amc = scheme == "HDFC Mutual Fund"

        values: dict[str, list[str]] = {}
        if not is_amc:
            for line in lines:
                for field, value in _extract_facts(line):
                    existing = values.setdefault(field, [])
                    if value not in existing:
                        existing.append(value)

        for field, (template, section) in FIELD_META.items():
            field_values = values.get(field, [])[:MAX_FACTS_PER_FIELD]
            for index, value in enumerate(field_values):
                suffix = "" if len(field_values) == 1 else f"-{index + 1}"
                _add_chunk(
                    chunks,
                    seen_ids,
                    Chunk(
                        id=f"{slug}-{field}{suffix}",
                        text=template.format(scheme=scheme, value=value),
                        source_url=source_url,
                        scheme_or_amc=scheme,
                        section=section,
                        field=field,
                        fetched_at=fetched_at,
                    ),
                )

        for heading, body_lines in _iter_sections(lines, scheme):
            field = _prose_field(heading, slug)
            section = _slugify(heading)
            parts = _split_prose(
                "\n".join(body_lines),
                settings.max_prose_words,
                settings.overlap_words,
            )
            for index, part in enumerate(parts):
                suffix = "" if len(parts) == 1 else f"-{index + 1}"
                _add_chunk(
                    chunks,
                    seen_ids,
                    Chunk(
                        id=f"{slug}-{field}{suffix}",
                        text=f"{scheme}: {heading}\n{part}",
                        source_url=source_url,
                        scheme_or_amc=scheme,
                        section=section,
                        field=field,
                        fetched_at=fetched_at,
                    ),
                )

    if not chunks:
        print("no chunks produced from loaded documents", file=sys.stderr)
        return False

    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.chunks_dir / "chunks.jsonl"
    _write_jsonl(output_path, chunks)
    print(f"chunks: {output_path} ({len(chunks)} chunks)")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--env", type=Path, default=None, help="path to .env")
    args = parser.parse_args(argv)

    settings = load_settings(config_path=args.config, env_path=args.env)
    return 0 if run_chunk(settings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
