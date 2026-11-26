"""Phase 2: save the five allowlisted public pages.

This stage only GETs the URLs from config. It never parses ``<a href>`` values,
never calls a Groww API, and never fetches a page discovered inside the HTML.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from grow_faq.config import PageConfig, Settings, load_settings

SUCCESS_STATUS = 200
NETWORK_ERROR_STATUS = 0


@dataclass(frozen=True)
class FetchResult:
    """One raw-page fetch and the metadata written beside it."""

    slug: str
    url: str
    scheme_or_amc: str
    fetched_at: str
    http_status: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.http_status == SUCCESS_STATUS

    def meta(self) -> dict[str, object]:
        """Exactly the sidecar fields required by the implementation plan."""
        return {
            "url": self.url,
            "scheme_or_amc": self.scheme_or_amc,
            "fetched_at": self.fetched_at,
            "http_status": self.http_status,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fetch(client: httpx.Client, page: PageConfig) -> tuple[bytes, FetchResult]:
    fetched_at = _utc_now()
    try:
        response = client.get(page.url)
    except httpx.HTTPError as exc:
        return b"", FetchResult(
            slug=page.slug,
            url=page.url,
            scheme_or_amc=page.name,
            fetched_at=fetched_at,
            http_status=NETWORK_ERROR_STATUS,
            error=f"{type(exc).__name__}: {exc}",
        )
    return response.content, FetchResult(
        slug=page.slug,
        url=page.url,
        scheme_or_amc=page.name,
        fetched_at=fetched_at,
        http_status=response.status_code,
    )


def _corpus_payload(
    results: list[FetchResult],
    checked_at: str,
    ok: bool,
) -> dict[str, object]:
    documents: list[dict[str, object]] = []
    for result in results:
        document: dict[str, object] = {
            "slug": result.slug,
            "url": result.url,
            "scheme_or_amc": result.scheme_or_amc,
            "fetched_at": result.fetched_at,
            "http_status": result.http_status,
            "ok": result.ok,
        }
        if result.error:
            document["error"] = result.error
        documents.append(document)

    successful_dates = sorted(result.fetched_at for result in results if result.ok)
    return {
        "ok": ok,
        "stage": "scrape",
        "checked_at": checked_at,
        "last_updated": successful_dates[0] if ok and successful_dates else None,
        "documents": documents,
    }


def run_scrape(settings: Settings) -> bool:
    """Fetch every configured page once. Return True only if all returned 200."""
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.corpus_file.parent.mkdir(parents=True, exist_ok=True)
    checked_at = _utc_now()
    results: list[FetchResult] = []

    with httpx.Client(
        timeout=settings.scrape_timeout_seconds,
        follow_redirects=False,
        headers={
            "User-Agent": settings.scrape_user_agent,
            "Accept": "text/html,application/xhtml+xml",
        },
    ) as client:
        for page in settings.pages:
            body, result = _fetch(client, page)
            (settings.raw_dir / f"{page.slug}.html").write_bytes(body)
            _write_json(settings.raw_dir / f"{page.slug}.meta.json", result.meta())
            results.append(result)

            if result.ok:
                print(f"[ok] {page.slug}: HTTP {result.http_status}")
            else:
                detail = f" ({result.error})" if result.error else ""
                print(
                    f"[fail] {page.slug}: HTTP {result.http_status}{detail}",
                    file=sys.stderr,
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
    return 0 if run_scrape(settings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
