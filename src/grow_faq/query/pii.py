"""Deterministic PII gate. This runs before retrieval, LLM, and tracing."""

from __future__ import annotations

import re

PII_MESSAGE = (
    "I don't take personal identifiers. Please ask a factual question about "
    "HDFC mutual funds."
)

PII_PATTERNS = (
    (
        "email",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("account number", re.compile(r"\b\d{12,18}\b")),
    ("Aadhaar", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("phone number", re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)")),
    (
        "OTP",
        re.compile(
            r"\b(?:otp|one[- ]?time password|verification code)\b"
            r"(?:\s+(?:is|was|for|to))?[\s:=-]{0,12}\d{4,8}\b"
            r"|\b\d{4,8}\b[\s:=-]{0,12}(?:\s+(?:is|was))?\s*"
            r"\b(?:otp|one[- ]?time password|verification code)\b",
            re.IGNORECASE,
        ),
    ),
)


def find_pii_kind(question: str) -> str | None:
    """Return the identifier category found, or None. Never returns the value."""
    for kind, pattern in PII_PATTERNS:
        if pattern.search(question):
            return kind
    return None


def contains_pii(question: str) -> bool:
    return find_pii_kind(question) is not None
