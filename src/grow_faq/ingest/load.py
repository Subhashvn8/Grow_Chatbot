"""Phase 3: turn saved raw HTML into one loaded document per page.

This stage never fetches and never chunks. It reads only ``data/raw/`` and writes
``data/documents/<slug>.json`` plus an updated ``data/corpus.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Tag

from grow_faq.config import PageConfig, Settings, load_settings

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
PROSE_TAGS = {"p", "div", "li"}
DROPPABLE_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "canvas",
    "nav",
    "footer",
}
DROP_HINTS = (
    "cookie",
    "consent",
    "banner",
    "modal",
    "popup",
    "rodal",
    "overlay",
    "dropdown",
    "megamenu",
    "navbar",
    "breadcrumb",
    "advert",
    "promo",
)
ROW_HINTS = (
    "row",
    "funddetails",
    "benchmarkrow",
    "exitloadrow",
    "gap4",
    "keyvalue",
    "termblock",
    "readmore",
)


@dataclass(frozen=True)
class DocumentResult:
    slug: str
    url: str
    scheme_or_amc: str
    fetched_at: str
    http_status: int
    text: str
    ok: bool
    reason: str | None = None

    def document(self) -> dict[str, str]:
        """Exactly the four fields required for a loaded document."""
        return {
            "url": self.url,
            "scheme_or_amc": self.scheme_or_amc,
            "fetched_at": self.fetched_at,
            "text": self.text,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _class_blob(tag: Tag) -> str:
    classes = tag.get("class", [])
    if isinstance(classes, str):
        classes = [classes]
    return " ".join([str(tag.get("id") or ""), *(str(item) for item in classes)]).lower()


def _is_droppable(tag: Tag) -> bool:
    return tag.name in DROPPABLE_TAGS or any(hint in _class_blob(tag) for hint in DROP_HINTS)


def _prune(node: Tag) -> None:
    for child in list(node.find_all(recursive=False)):
        if not isinstance(child, Tag):
            continue
        if _is_droppable(child):
            child.decompose()
        else:
            _prune(child)


def _content_root(soup: BeautifulSoup) -> Tag:
    root = soup.select_one("div.layout-main") or soup.body or soup
    _prune(root)
    return root


def _row_parts(tag: Tag) -> list[str] | None:
    parts: list[str] = []
    for child in tag.find_all(recursive=False):
        text = _normalized(child.get_text(" ", strip=True))
        if text:
            parts.append(text)
    return parts if 2 <= len(parts) <= 3 else None


def _is_row(tag: Tag) -> bool:
    if tag.name not in {"div", "li", "tr"}:
        return False
    blob = _class_blob(tag)
    return any(hint in blob for hint in ROW_HINTS) and _row_parts(tag) is not None


def _is_prose(tag: Tag) -> bool:
    if tag.name not in PROSE_TAGS:
        return False
    text = _normalized(tag.get_text(" ", strip=True))
    if not 25 <= len(text) <= 1200:
        return False
    children = [
        _normalized(child.get_text(" ", strip=True))
        for child in tag.find_all(recursive=False)
    ]
    return not any(len(child_text) >= 25 for child_text in children)


def _is_descendant(tag: Tag, ancestor: Tag) -> bool:
    return any(parent is ancestor for parent in tag.parents)


def _tag_line(tag: Tag) -> str:
    if tag.name in HEADING_TAGS:
        return _normalized(tag.get_text(" ", strip=True))
    parts = _row_parts(tag)
    if parts:
        return ": ".join(parts)
    return _normalized(tag.get_text(" ", strip=True))


def _dedupe(lines: list[str]) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        if len(line) < 2 or line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return "\n".join(kept)


def extract_text(html: str) -> str:
    """Extract headings, paragraphs, and label/value rows from visible page copy."""
    soup = BeautifulSoup(html, "html.parser")
    root = _content_root(soup)

    selected: list[Tag] = []
    for tag in root.find_all(True):
        if tag.name in HEADING_TAGS or _is_row(tag) or _is_prose(tag):
            selected.append(tag)

    rows = [tag for tag in selected if _is_row(tag)]
    lines: list[str] = []
    for tag in selected:
        if any(tag is not row and _is_descendant(tag, row) for row in rows):
            continue
        line = _tag_line(tag)
        if line:
            lines.append(line)

    return _dedupe(lines)


def _read_meta(path: Path, page: PageConfig) -> tuple[str, str, int, str | None]:
    if not path.is_file():
        return page.url, "", 0, "raw sidecar missing"
    try:
        meta: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return page.url, "", 0, f"could not read raw sidecar: {exc}"
    if not isinstance(meta, dict):
        return page.url, "", 0, "raw sidecar is not a JSON object"

    url = str(meta.get("url") or page.url)
    fetched_at = str(meta.get("fetched_at") or "")
    status = meta.get("http_status")
    if isinstance(status, bool) or not isinstance(status, int):
        status = 0

    if url != page.url:
        return url, fetched_at, status, "raw sidecar URL does not match config"
    if status != 200:
        return url, fetched_at, status, "raw page was not fetched with HTTP 200"
    if not fetched_at:
        return url, fetched_at, status, "raw sidecar has no fetched_at"
    return url, fetched_at, status, None


def _read_corpus(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _corpus_payload(results: list[DocumentResult], checked_at: str, ok: bool) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    for result in results:
        document: dict[str, Any] = {
            "slug": result.slug,
            "url": result.url,
            "scheme_or_amc": result.scheme_or_amc,
            "fetched_at": result.fetched_at,
            "http_status": result.http_status,
            "ok": result.ok,
            "text_chars": len(result.text.strip()),
        }
        if result.reason:
            document["reason"] = result.reason
        documents.append(document)

    successful_dates = sorted(
        result.fetched_at for result in results if result.ok and result.fetched_at
    )
    return {
        "ok": ok,
        "stage": "load",
        "checked_at": checked_at,
        "last_updated": successful_dates[0] if ok and successful_dates else None,
        "documents": documents,
    }


def run_load(settings: Settings) -> bool:
    """Load all five saved pages. Return True only if every document is usable."""
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.corpus_file.parent.mkdir(parents=True, exist_ok=True)

    previous = _read_corpus(settings.corpus_file)
    if previous and previous.get("stage") == "scrape" and not previous.get("ok"):
        print("scrape corpus is not ok; run grow-scrape before grow-load", file=sys.stderr)
        return False

    checked_at = _utc_now()
    results: list[DocumentResult] = []

    for page in settings.pages:
        html_path = settings.raw_dir / f"{page.slug}.html"
        meta_path = settings.raw_dir / f"{page.slug}.meta.json"
        url, fetched_at, status, reason = _read_meta(meta_path, page)
        text = ""

        if reason is None:
            try:
                text = extract_text(html_path.read_text(encoding="utf-8", errors="replace"))
            except OSError as exc:
                reason = f"could not read raw HTML: {exc}"

        if reason is None and len(text.strip()) < settings.min_document_chars:
            reason = f"visible text too short ({len(text.strip())} chars)"

        ok = reason is None
        document_path = settings.documents_dir / f"{page.slug}.json"
        if ok:
            _write_json(document_path, DocumentResult(
                page.slug, url, page.name, fetched_at, status, text, True
            ).document())
            print(f"[ok] {page.slug}: {len(text.strip())} chars")
        else:
            if document_path.exists():
                document_path.unlink()
            print(f"[fail] {page.slug}: {reason}", file=sys.stderr)
            text = ""

        results.append(
            DocumentResult(
                slug=page.slug,
                url=url,
                scheme_or_amc=page.name,
                fetched_at=fetched_at,
                http_status=status,
                text=text,
                ok=ok,
                reason=reason,
            )
        )

    ok = all(result.ok for result in results)
    _write_json(settings.corpus_file, _corpus_payload(results, checked_at, ok))
    print(f"corpus: {settings.corpus_file} (ok={str(ok).lower()})")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--env", type=Path, default=None, help="path to .env")
    args = parser.parse_args(argv)

    settings = load_settings(config_path=args.config, env_path=args.env)
    return 0 if run_load(settings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
