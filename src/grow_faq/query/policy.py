"""Pre-generation question policy.

This is deterministic code. PII is handled before this module runs, so the
classes here are only ``answer``, ``factsheet``, and ``refuse``.
"""

from __future__ import annotations

import re

ANSWER = "answer"
FACTSHEET = "factsheet"
REFUSE = "refuse"
PII = "pii"

# Advice wins over returns when both are present: "Should I buy this for the
# return?" is a refusal, not a factsheet question.
#
# Only advice *phrasing* is matched. Bare verbs such as "buy", "sell" or "hold"
# also appear in factual questions ("exit load if I sell within a year?",
# "how do I buy ...?"), so they are refused only inside a request for a
# judgement.
ADVICE_RE = re.compile(
    r"\b(?:should i|shall i|should we|do you think|would you recommend"
    r"|can you recommend|recommend(?:s|ed|ing|ation|ations)?|advise|advice"
    r"|suggest(?:ion)?|invest now|good investment|safe investment"
    r"|worth (?:it|buying|investing|holding|keeping)"
    r"|(?:good|right|best) time to (?:buy|sell|invest|enter|exit)"
    r"|(?:buy|sell|hold|keep|switch|exit|redeem)\s+or\s+(?:not|buy|sell|hold|keep|switch|exit|redeem)"
    r"|(?:help me|which (?:one|fund|scheme) to) (?:pick|choose|select)"
    r"|best (?:fund|scheme|option|choice)s?)\b"
    r"|\bwhich (?:fund|scheme|one) (?:should|is (?:best|better)|would)\b"
    r"|\bwhich is (?:better|best)\b"
    r"|\bis (?:it|this|this fund|now) (?:a )?good\b",
    re.IGNORECASE,
)

FACTSHEET_RE = re.compile(
    r"\b(?:returns?|performance|cagr|outperform(?:ed|ing)?|trailing return"
    r"|highest return|top performing|which did better)\b",
    re.IGNORECASE,
)

FIELD_PATTERNS = (
    ("minimum_sip", re.compile(r"\b(?:minimum|min)\s+sip\b|\bsip\s+(?:amount|minimum)\b", re.I)),
    ("expense_ratio", re.compile(r"\bexpense\s+ratios?\b", re.I)),
    ("exit_load", re.compile(r"\bexit\s+loads?\b", re.I)),
    ("lock_in", re.compile(r"\block[\s-]?in\b", re.I)),
    ("riskometer", re.compile(r"\briskometer\b|\brisk\s+level\b", re.I)),
    ("benchmark", re.compile(r"\bbenchmark\b", re.I)),
    ("statement_download", re.compile(r"\b(?:statement|capital[\s-]?gains?)\b", re.I)),
)


def classify(question: str) -> str:
    """Return the answer policy class for a non-PII question."""
    if ADVICE_RE.search(question):
        return REFUSE
    if FACTSHEET_RE.search(question):
        return FACTSHEET
    return ANSWER


def asked_field(question: str) -> str | None:
    """Return the normalized field a question asks about, if recognizable."""
    for name, pattern in FIELD_PATTERNS:
        if pattern.search(question):
            return name
    return None
