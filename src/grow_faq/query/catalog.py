"""Read the hand-curated source catalog as a citation allowlist.

``Docs/sources.md`` is never written by the loader. Query-time code reads it to
decide which URLs a draft answer may cite and to pick catalog URLs for refusal,
factsheet, and fallback templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from grow_faq.config import Settings
from grow_faq.query.schemes import scheme_keys

URL_RE = re.compile(r"https?://[^\s<>\"']+")

GROUP_HEADINGS = {
    "ingested pages": "ingested",
    "factsheets": "factsheets",
    "education": "education",
    "statements": "statements",
}


@dataclass(frozen=True)
class SourceEntry:
    label: str
    url: str


@dataclass(frozen=True)
class SourceCatalog:
    ingested: tuple[SourceEntry, ...]
    factsheets: tuple[SourceEntry, ...]
    education: tuple[SourceEntry, ...]
    statements: tuple[SourceEntry, ...]

    @property
    def all_urls(self) -> frozenset[str]:
        return frozenset(
            entry.url
            for group in (self.ingested, self.factsheets, self.education, self.statements)
            for entry in group
        )

    @property
    def ingested_urls(self) -> frozenset[str]:
        return frozenset(entry.url for entry in self.ingested)

    @property
    def factsheet_urls(self) -> frozenset[str]:
        return frozenset(entry.url for entry in self.factsheets)

    @property
    def education_urls(self) -> frozenset[str]:
        return frozenset(entry.url for entry in self.education)

    def _match(self, group: tuple[SourceEntry, ...], scheme: str | None) -> SourceEntry | None:
        if scheme:
            keys = set(scheme_keys(scheme))
            for entry in group:
                if keys.intersection(scheme_keys(entry.label)):
                    return entry
        return group[0] if group else None

    def ingested_url(self, scheme: str | None = None) -> str:
        entry = self._match(self.ingested, scheme)
        return entry.url if entry else ""

    def factsheet_url(self, scheme: str | None = None) -> str:
        entry = self._match(self.factsheets, scheme)
        return entry.url if entry else ""

    def education_url(self) -> str:
        return self.education[0].url if self.education else ""

    def statement_url(self, capital_gains: bool = False) -> str:
        """Statement page; the capital-gains guide when that is asked for."""
        if capital_gains:
            for entry in self.statements:
                if "capital" in entry.label.casefold():
                    return entry.url
        return self.statements[0].url if self.statements else ""


def load_catalog(settings: Settings) -> SourceCatalog:
    """Parse the curated Markdown catalog into four URL groups."""
    path = settings.sources_catalog
    if not path.is_file():
        raise ValueError(f"source catalog not found: {path}")

    groups: dict[str, list[SourceEntry]] = {
        name: [] for name in GROUP_HEADINGS.values()
    }
    current: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = GROUP_HEADINGS.get(heading.group(1).strip().casefold())
            continue
        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        if current is None or not bullet:
            continue
        text = bullet.group(1).strip()
        match = URL_RE.search(text)
        if match is None:
            continue
        url = match.group(0).rstrip(".,;:!?)]}>\"'")
        label = text[: match.start()].strip().rstrip(":–-").strip()
        groups[current].append(SourceEntry(label=label or url, url=url))

    catalog = SourceCatalog(
        ingested=tuple(groups["ingested"]),
        factsheets=tuple(groups["factsheets"]),
        education=tuple(groups["education"]),
        statements=tuple(groups["statements"]),
    )
    if not catalog.ingested or not catalog.factsheets or not catalog.education:
        raise ValueError(f"source catalog is missing required groups: {path}")
    return catalog
