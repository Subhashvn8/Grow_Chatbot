"""Load and validate the single non-secret config file.

This is the only module that reads ``config.yaml``. API keys are read from the
environment (optionally via ``.env``) and are never written to config or traces.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

SUPPORTED_PROVIDERS = ("groq", "gemini", "openai", "anthropic", "mistral")
PROVIDER_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}
EXPECTED_PAGE_COUNT = 5
VALID_METRICS = {"cosine", "l2", "ip"}
ALLOWED_HOSTS = {"groww.in", "www.groww.in"}


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed, or out of scope."""


@dataclass(frozen=True)
class PageConfig:
    """One of the five allowlisted Groww pages."""

    name: str
    url: str
    slug: str


@dataclass(frozen=True)
class Settings:
    """Validated settings shared by ingestion, query, and UI modules."""

    root: Path
    config_path: Path
    scrape_timeout_seconds: float
    scrape_user_agent: str
    min_document_chars: int
    pages: tuple[PageConfig, ...]
    embedding_model: str
    embedding_batch_size: int
    vector_persist_path: Path
    vector_collection: str
    vector_metric: str
    max_prose_words: int
    overlap_words: int
    top_k: int
    llm_provider: str
    llm_model: str
    llm_timeout_seconds: float
    max_sentences: int
    sources_catalog: Path
    raw_dir: Path
    documents_dir: Path
    chunks_dir: Path
    embeddings_dir: Path
    traces_dir: Path
    corpus_file: Path
    api_keys: Mapping[str, str | None] = field(repr=False)

    @property
    def api_key(self) -> str | None:
        """API key for the provider selected in config.yaml, if set."""
        return self.api_keys.get(self.llm_provider)

    def require_api_key(self) -> str:
        """Return the selected provider's key or fail with the env var name."""
        key = self.api_key
        if not key:
            env_var = PROVIDER_ENV_VARS[self.llm_provider]
            raise ConfigError(
                f"Missing API key for provider '{self.llm_provider}'. "
                f"Set {env_var} in {self.root / '.env'} or the environment."
            )
        return key

    def describe(self) -> str:
        """Non-secret summary, useful for ``python -m grow_faq.config``."""
        env_var = PROVIDER_ENV_VARS[self.llm_provider]
        key_state = "set" if self.api_key else f"not set ({env_var})"
        lines = [
            f"config: {self.config_path}",
            f"scrape timeout: {self.scrape_timeout_seconds}s",
            f"pages: {len(self.pages)} (allowlisted)",
            f"min document chars: {self.min_document_chars}",
            f"embedding model: {self.embedding_model} (batch {self.embedding_batch_size})",
            f"vector DB: {self.vector_collection} @ {self.vector_persist_path} ({self.vector_metric})",
            f"chunking: {self.max_prose_words} words max, {self.overlap_words} overlap",
            f"top_k: {self.top_k}",
            f"llm: {self.llm_provider}/{self.llm_model} - {key_state}",
            f"llm timeout: {self.llm_timeout_seconds}s",
            f"max sentences: {self.max_sentences}",
            f"source catalog: {self.sources_catalog}",
            f"corpus file: {self.corpus_file}",
        ]
        lines.extend(f"page: {page.slug} - {page.name}" for page in self.pages)
        return "\n".join(lines)


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _text(section: Mapping[str, Any], key: str, where: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{where}.{key}' must be a non-empty string")
    return value.strip()


def _positive_int(section: Mapping[str, Any], key: str, where: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{where}.{key}' must be a positive integer")
    return value


def _positive_number(section: Mapping[str, Any], key: str, where: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"'{where}.{key}' must be a positive number")
    return float(value)


def _path(section: Mapping[str, Any], key: str, where: str, root: Path) -> Path:
    value = _text(section, key, where)
    return (root / value).resolve()


def _slug_from_url(url: str, where: str) -> str:
    segments = [segment for segment in urlparse(url).path.split("/") if segment]
    if not segments:
        raise ConfigError(f"'{where}.url' has no path")
    slug = re.sub(r"[^a-z0-9]+", "-", segments[-1].lower()).strip("-")
    if not slug:
        raise ConfigError(f"'{where}.url' cannot produce a file slug")
    return slug


def _pages(corpus: Mapping[str, Any]) -> tuple[PageConfig, ...]:
    raw_pages = corpus.get("pages")
    if not isinstance(raw_pages, list) or len(raw_pages) != EXPECTED_PAGE_COUNT:
        raise ConfigError(
            f"'corpus.pages' must list exactly {EXPECTED_PAGE_COUNT} pages"
        )

    pages: list[PageConfig] = []
    seen_urls: set[str] = set()
    seen_names: set[str] = set()
    seen_slugs: set[str] = set()

    for index, item in enumerate(raw_pages, start=1):
        where = f"corpus.pages[{index}]"
        if not isinstance(item, Mapping):
            raise ConfigError(f"'{where}' must be a mapping")
        name = _text(item, "name", where)
        url = _text(item, "url", where)
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc.lower() not in ALLOWED_HOSTS
            or not parsed.path.startswith("/mutual-funds/")
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigError(
                f"'{where}.url' must be an https://groww.in/mutual-funds/... URL"
            )

        slug = _slug_from_url(url, where)
        for label, value, seen in (
            ("url", url, seen_urls),
            ("name", name.casefold(), seen_names),
            ("slug", slug, seen_slugs),
        ):
            if value in seen:
                raise ConfigError(f"Duplicate {label} in 'corpus.pages': {value!r}")
            seen.add(value)

        pages.append(PageConfig(name=name, url=url, slug=slug))

    return tuple(pages)


def load_settings(
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
) -> Settings:
    """Read, validate, and return settings from ``config.yaml``.

    Relative paths in the file are resolved against the config file's directory.
    """
    path = Path(config_path).resolve() if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} must contain a YAML mapping")

    root = path.parent
    load_dotenv(Path(env_path).resolve() if env_path is not None else root / ".env")

    scrape = _section(data, "scrape")
    corpus = _section(data, "corpus")
    embedding = _section(data, "embedding")
    vector_db = _section(data, "vector_db")
    chunking = _section(data, "chunking")
    retrieval = _section(data, "retrieval")
    llm = _section(data, "llm")
    answer = _section(data, "answer")
    sources = _section(data, "sources")
    paths = _section(data, "paths")

    provider = _text(llm, "provider", "llm").lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"'llm.provider' must be one of {', '.join(SUPPORTED_PROVIDERS)}"
        )

    metric = _text(vector_db, "metric", "vector_db").lower()
    if metric not in VALID_METRICS:
        raise ConfigError(f"'vector_db.metric' must be one of {', '.join(sorted(VALID_METRICS))}")

    max_prose_words = _positive_int(chunking, "max_prose_words", "chunking")
    overlap_words = _positive_int(chunking, "overlap_words", "chunking")
    if overlap_words >= max_prose_words:
        raise ConfigError("'chunking.overlap_words' must be less than 'chunking.max_prose_words'")

    api_keys = {
        name: (os.getenv(env_var) or "").strip() or None
        for name, env_var in PROVIDER_ENV_VARS.items()
    }

    return Settings(
        root=root,
        config_path=path,
        scrape_timeout_seconds=_positive_number(scrape, "timeout_seconds", "scrape"),
        scrape_user_agent=_text(scrape, "user_agent", "scrape"),
        min_document_chars=_positive_int(corpus, "min_document_chars", "corpus"),
        pages=_pages(corpus),
        embedding_model=_text(embedding, "model", "embedding"),
        embedding_batch_size=_positive_int(embedding, "batch_size", "embedding"),
        vector_persist_path=_path(vector_db, "persist_path", "vector_db", root),
        vector_collection=_text(vector_db, "collection", "vector_db"),
        vector_metric=metric,
        max_prose_words=max_prose_words,
        overlap_words=overlap_words,
        top_k=_positive_int(retrieval, "top_k", "retrieval"),
        llm_provider=provider,
        llm_model=_text(llm, "model", "llm"),
        llm_timeout_seconds=_positive_number(llm, "timeout_seconds", "llm"),
        max_sentences=_positive_int(answer, "max_sentences", "answer"),
        sources_catalog=_path(sources, "catalog", "sources", root),
        raw_dir=_path(paths, "raw_dir", "paths", root),
        documents_dir=_path(paths, "documents_dir", "paths", root),
        chunks_dir=_path(paths, "chunks_dir", "paths", root),
        embeddings_dir=_path(paths, "embeddings_dir", "paths", root),
        traces_dir=_path(paths, "traces_dir", "paths", root),
        corpus_file=_path(paths, "corpus_file", "paths", root),
        api_keys=api_keys,
    )


if __name__ == "__main__":
    print(load_settings().describe())
