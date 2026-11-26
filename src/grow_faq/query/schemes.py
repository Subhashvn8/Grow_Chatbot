"""Scheme-name detection shared by retrieval, citation, and templates.

A question that names exactly one scheme can be filtered by ``scheme_or_amc``.
A question that names none (or more than one) searches the whole collection.
"""

from __future__ import annotations

import re

from grow_faq.config import Settings

# Several aliases can resolve to the same scheme (ELSS, Tax Saver, lock-in).
ALIAS_KEYS = (
    ("large cap", "large cap"),
    ("flexi cap", "flexi cap"),
    ("elss", "elss"),
    ("tax saver", "elss"),
    ("lock in", "elss"),
    ("mid cap", "mid cap"),
    ("hdfc mutual fund", "amc"),
)

KEY_TOKENS = {
    "amc": ("hdfc mutual fund",),
    "large cap": ("large cap",),
    "flexi cap": ("flexi cap",),
    "elss": ("elss",),
    "mid cap": ("mid cap",),
}


def normalize(text: str) -> str:
    """Lowercase text and turn separators into spaces for phrase matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def scheme_keys(text: str) -> tuple[str, ...]:
    """Return the distinct scheme keys named in ``text``."""
    normalized = normalize(text)
    keys: list[str] = []
    for alias, key in ALIAS_KEYS:
        if alias in normalized and key not in keys:
            keys.append(key)
    return tuple(keys)


def _name_for_key(settings: Settings, key: str) -> str | None:
    tokens = KEY_TOKENS.get(key, (key,))
    for page in settings.pages:
        name = normalize(page.name)
        if all(token in name for token in tokens):
            return page.name
    return None


def scheme_names(text: str, settings: Settings) -> tuple[str, ...]:
    """Return configured scheme names named in ``text``, in config order."""
    names: list[str] = []
    for key in scheme_keys(text):
        name = _name_for_key(settings, key)
        if name is not None and name not in names:
            names.append(name)
    return tuple(names)


def single_scheme(text: str, settings: Settings) -> str | None:
    """Return one named scheme, or None when zero or several are named."""
    names = scheme_names(text, settings)
    return names[0] if len(names) == 1 else None
