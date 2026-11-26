"""Phase 7: the shared question-answering path used by CLI and UI.

Order: PII gate, policy, retrieval, generation, output check, trace. This
module never fetches the web; all facts come from the persistent index.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grow_faq.config import Settings, load_settings
from grow_faq.llm.adapters import LLM, build_llm
from grow_faq.query.answer import (
    Answer,
    failure_answer,
    generate_answer,
    missing_answer,
    source_date,
)
from grow_faq.query.catalog import SourceCatalog, load_catalog
from grow_faq.query.pii import PII_MESSAGE, contains_pii
from grow_faq.query.policy import ANSWER, PII, asked_field, classify
from grow_faq.query.retrieve import RetrievedChunk, StaleModelError, retrieve
from grow_faq.query.schemes import single_scheme
from grow_faq.query.trace import write_trace


@dataclass(frozen=True)
class AskResult:
    text: str
    policy: str
    citation: str | None
    chunks: tuple[RetrievedChunk, ...]
    pii: bool
    trace_path: Path | None


_SENSITIVE_TEXT = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b\s*[:=]\s*\S+"
)


def safe_error_detail(exc: BaseException) -> str:
    """Return a short, non-secret explanation for operator diagnostics."""
    text = str(exc).strip() or exc.__class__.__name__
    text = _SENSITIVE_TEXT.sub(r"\1=<redacted>", text)
    return f"{exc.__class__.__name__}: {text[:240]}"


class AskStageError(RuntimeError):
    """An error before answer generation, labelled by pipeline stage."""

    def __init__(self, stage: str, error: BaseException) -> None:
        self.stage = stage
        self.detail = safe_error_detail(error)
        super().__init__(f"{stage} failed: {self.detail}")


def _prioritize_field(
    chunks: tuple[RetrievedChunk, ...], field: str | None
) -> tuple[RetrievedChunk, ...]:
    """Put chunks for the asked field first without changing other order."""
    if not field:
        return chunks
    return tuple(sorted(chunks, key=lambda chunk: chunk.field != field))


def run_ask(
    question: str,
    settings: Settings | None = None,
    *,
    llm: LLM | None = None,
    client: Any | None = None,
    catalog: SourceCatalog | None = None,
) -> AskResult:
    """Answer a question, or reject it before any trace or LLM call."""
    settings = settings or load_settings()

    if not question or not question.strip():
        return AskResult(
            text="Please ask a factual question about HDFC mutual funds.",
            policy=ANSWER,
            citation=None,
            chunks=(),
            pii=False,
            trace_path=None,
        )

    # Deterministic gate: nothing below this line may see or store the question
    # when an identifier is present.
    if contains_pii(question):
        return AskResult(
            text=PII_MESSAGE,
            policy=PII,
            citation=None,
            chunks=(),
            pii=True,
            trace_path=None,
        )

    try:
        policy = classify(question)
        field = asked_field(question) if policy == ANSWER else None
        scheme = single_scheme(question, settings)
    except Exception as exc:
        raise AskStageError("policy", exc) from exc

    try:
        chunks = _prioritize_field(
            tuple(retrieve(question, settings, client=client)), field
        )
    except StaleModelError:
        raise
    except Exception as exc:
        raise AskStageError("retrieval", exc) from exc

    try:
        catalog = catalog or load_catalog(settings)
        date = source_date(settings)
    except Exception as exc:
        raise AskStageError("catalog", exc) from exc

    if policy == ANSWER and field and not any(chunk.field == field for chunk in chunks):
        answer = missing_answer(
            policy,
            chunks,
            catalog,
            settings,
            date,
            scheme=scheme,
            field=field,
        )
    else:
        try:
            llm = llm or build_llm(settings)
            answer = generate_answer(
                question,
                policy,
                chunks,
                settings,
                llm,
                catalog,
                scheme=scheme,
                field=field,
                date=date,
            )
        except Exception:
            answer = failure_answer(
                policy,
                chunks,
                catalog,
                settings,
                date,
                scheme=scheme,
                field=field,
            )

    trace_path: Path | None = None
    try:
        trace_path = write_trace(
            settings,
            question=question,
            policy=policy,
            chunks=chunks,
            answer=answer,
        )
    except OSError as exc:
        print(f"[warn] could not write trace: {exc}", file=sys.stderr)
    except Exception as exc:
        raise AskStageError("trace", exc) from exc

    return AskResult(
        text=answer.text,
        policy=policy,
        citation=answer.citation,
        chunks=chunks,
        pii=False,
        trace_path=trace_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="question to answer")
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--env", type=Path, default=None, help="path to .env")
    args = parser.parse_args(argv)

    settings = load_settings(config_path=args.config, env_path=args.env)
    try:
        result = run_ask(" ".join(args.question), settings=settings)
    except StaleModelError as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1

    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
