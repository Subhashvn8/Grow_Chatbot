"""Strict prompt, output check, and deterministic templates for answers.

The LLM only sees retrieved chunks. Its draft is accepted only when it has at
most ``answer.max_sentences`` sentences, exactly one allowed URL, the corpus
date line, and no advice language. Otherwise a template replaces it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from grow_faq.config import Settings
from grow_faq.query.catalog import URL_RE, SourceCatalog
from grow_faq.query.policy import ANSWER, FACTSHEET, REFUSE
from grow_faq.query.retrieve import RetrievedChunk
from grow_faq.query.schemes import single_scheme

DATE_PREFIX = "Last updated from sources:"

POLICY_INSTRUCTIONS = {
    ANSWER: (
        "Answer only the asked field. Cite the source_url of the chunk that "
        "contains the fact."
    ),
    FACTSHEET: (
        "Do not calculate, compare, or invent returns. Cite one official "
        "factsheet URL from the allowed list."
    ),
    REFUSE: (
        "Do not give investment advice or a recommendation, even if the user "
        "rephrases it. Cite one education URL from the allowed list."
    ),
}

AFFIRMATIVE_ADVICE_RE = re.compile(
    r"\b(?:you should|i recommend|i advise|buy this|sell this|hold this|pick this|go for)\b",
    re.IGNORECASE,
)
# Advice phrasing in a factual draft. Bare "buy"/"sell"/"hold" are allowed
# because facts use them ("an exit load applies if you sell within a year").
ADVICE_LEAK_RE = re.compile(
    r"\b(?:recommend(?:s|ed|ing|ation|ations)?|advise|advisable|advice|suggest"
    r"|good (?:time|investment|choice|option)|worth (?:it|buying|investing|holding)"
    r"|consider (?:buying|selling|investing|holding|switching)"
    r"|(?:buy|sell|hold|invest) now|better (?:fund|choice|option|investment))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Answer:
    text: str
    citation: str | None
    policy: str


def source_date(settings: Settings) -> str:
    """Return the corpus's last successful fetch date, never today's date."""
    try:
        data = json.loads(settings.corpus_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return "unknown"
    if not isinstance(data, dict):
        return "unknown"
    value = data.get("last_updated")
    if isinstance(value, str):
        match = re.match(r"(\d{4}-\d{2}-\d{2})", value)
        if match:
            return match.group(1)
    return "unknown"


def allowed_urls(
    policy: str,
    chunks: tuple[RetrievedChunk, ...],
    catalog: SourceCatalog,
) -> frozenset[str]:
    """URLs the draft may cite for this policy class."""
    if policy == ANSWER:
        urls = {chunk.source_url for chunk in chunks if chunk.source_url}
        return frozenset(urls or catalog.ingested_urls)
    if policy == FACTSHEET:
        return catalog.factsheet_urls
    if policy == REFUSE:
        return catalog.education_urls
    return catalog.all_urls


def build_prompt(
    question: str,
    policy: str,
    chunks: tuple[RetrievedChunk, ...],
    allowed: frozenset[str],
    date: str,
    settings: Settings,
) -> tuple[str, str]:
    """Return the system and user messages for the configured LLM."""
    chunk_lines = "\n".join(
        f"- id={chunk.id} | field={chunk.field or 'none'} | "
        f"scheme={chunk.scheme_or_amc} | section={chunk.section} | "
        f"url={chunk.source_url} | text={chunk.text}"
        for chunk in chunks
    ) or "(no chunks retrieved)"

    system = (
        "You are a facts-only assistant for HDFC mutual funds.\n"
        f"Policy class: {policy}\n"
        "Use only the retrieved chunks below for scheme facts and numbers. "
        "Do not use memory or outside sources.\n"
        "If the asked fact is not in the chunks, say 'Not in the sourced pages.'\n"
        f"{POLICY_INSTRUCTIONS.get(policy, POLICY_INSTRUCTIONS[ANSWER])}\n"
        f"Write at most {settings.max_sentences} sentences in total, including "
        "the final date line.\n"
        "Include exactly one URL, chosen from the allowed URLs below.\n"
        f"End with exactly this line: {DATE_PREFIX} {date}\n"
        "Do not echo personal identifiers.\n"
        f"Retrieved chunks:\n{chunk_lines}\n"
        f"Allowed URLs: {', '.join(sorted(allowed))}"
    )
    user = f"Question: {question}"
    return system, user


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?)]}>\"'")


def _extract_urls(text: str) -> list[str]:
    return [_clean_url(url) for url in URL_RE.findall(text)]


def _count_sentences(text: str) -> int:
    masked = URL_RE.sub("URL_TOKEN", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", masked.strip())
    return len([part for part in parts if part.strip()])


def _has_advice_leak(text: str, policy: str) -> bool:
    if policy == REFUSE:
        return bool(AFFIRMATIVE_ADVICE_RE.search(text))
    return bool(ADVICE_LEAK_RE.search(text)) or bool(
        re.search(r"\byou should\b", text, re.IGNORECASE)
    )


def validate_draft(
    draft: str,
    *,
    policy: str,
    allowed: frozenset[str],
    date: str,
    max_sentences: int,
) -> Answer | None:
    """Accept a draft only if every output guardrail passes."""
    if not isinstance(draft, str) or not draft.strip():
        return None
    text = draft.strip()
    urls = _extract_urls(text)
    if len(urls) != 1 or urls[0] not in allowed:
        return None
    if not re.search(
        rf"{re.escape(DATE_PREFIX)}\s*{re.escape(date)}\.?\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    ):
        return None
    if _count_sentences(text) > max_sentences:
        return None
    if policy == FACTSHEET and re.search(r"\b\d+(?:\.\d+)?\s*%", text):
        return None
    if _has_advice_leak(text, policy):
        return None
    return Answer(text=text, citation=urls[0], policy=policy)


def _first_sentence(text: str) -> str:
    body = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    body = URL_RE.sub("", body).strip()
    if len(body) > 240:
        body = body[:240].rstrip() + "…"
    return body


def _template_url(
    policy: str,
    chunks: tuple[RetrievedChunk, ...],
    catalog: SourceCatalog,
    settings: Settings,
    scheme: str | None,
    field: str | None = None,
) -> str:
    if policy == ANSWER:
        if chunks and chunks[0].source_url:
            return chunks[0].source_url
        if field == "statement_download" and catalog.statement_url():
            return catalog.statement_url()
        return catalog.ingested_url(scheme) or catalog.ingested_url()
    if policy == FACTSHEET:
        return catalog.factsheet_url(scheme) or catalog.factsheet_url()
    return catalog.education_url()


def template_answer(
    policy: str,
    chunks: tuple[RetrievedChunk, ...],
    catalog: SourceCatalog,
    settings: Settings,
    date: str,
    *,
    scheme: str | None = None,
    field: str | None = None,
) -> Answer:
    """Short, grounded fallback used when a draft fails a guardrail."""
    url = _template_url(policy, chunks, catalog, settings, scheme, field)
    if policy == REFUSE:
        body, label = "I can only share sourced facts, not investment guidance.", "Education source"
    elif policy == FACTSHEET:
        body, label = "I can't calculate or compare returns.", "Official factsheet"
    elif chunks:
        body, label = _first_sentence(chunks[0].text), "Source"
    else:
        body, label = "Not in the sourced pages.", "Source"
    text = f"{body}\n{label}: {url}\n{DATE_PREFIX} {date}"
    return Answer(text=text, citation=url or None, policy=policy)


def failure_answer(
    policy: str,
    chunks: tuple[RetrievedChunk, ...],
    catalog: SourceCatalog,
    settings: Settings,
    date: str,
    *,
    scheme: str | None = None,
    field: str | None = None,
) -> Answer:
    """Provider failure message with a real citation but no guessed fact."""
    url = _template_url(policy, chunks, catalog, settings, scheme, field)
    text = f"The answer could not be generated.\nSource: {url}\n{DATE_PREFIX} {date}"
    return Answer(text=text, citation=url or None, policy=policy)


def missing_answer(
    policy: str,
    chunks: tuple[RetrievedChunk, ...],
    catalog: SourceCatalog,
    settings: Settings,
    date: str,
    *,
    scheme: str | None = None,
    field: str | None = None,
) -> Answer:
    """A field with no sourced chunk gets a clear refusal and a scheme page."""
    url = _template_url(policy, chunks, catalog, settings, scheme, field)
    text = f"Not in the sourced pages.\nSource: {url}\n{DATE_PREFIX} {date}"
    return Answer(text=text, citation=url or None, policy=policy)


def generate_answer(
    question: str,
    policy: str,
    chunks: tuple[RetrievedChunk, ...],
    settings: Settings,
    llm: object,
    catalog: SourceCatalog,
    *,
    scheme: str | None = None,
    field: str | None = None,
    date: str | None = None,
) -> Answer:
    """Generate, validate, and fall back to a deterministic template."""
    date = date or source_date(settings)
    scheme = scheme if scheme is not None else single_scheme(question, settings)
    allowed = allowed_urls(policy, chunks, catalog)
    system, user = build_prompt(question, policy, chunks, allowed, date, settings)

    try:
        draft = llm.generate(system, user)  # type: ignore[attr-defined]
    except Exception:
        return failure_answer(policy, chunks, catalog, settings, date, scheme=scheme, field=field)

    validated = validate_draft(
        draft,
        policy=policy,
        allowed=allowed,
        date=date,
        max_sentences=settings.max_sentences,
    )
    if validated is not None:
        return validated
    return template_answer(policy, chunks, catalog, settings, date, scheme=scheme, field=field)
