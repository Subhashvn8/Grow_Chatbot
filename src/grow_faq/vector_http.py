"""Small Chroma v2 REST client used on Vercel.

The local pipeline keeps using ``chromadb``'s persistent client. Serverless
retrieval only needs collection metadata and vector search, so this module uses
the standard library to call those two endpoints. That avoids shipping
``chromadb`` and its heavy native dependencies in the function bundle.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from grow_faq.config import Settings

DEFAULT_TENANT = "default_tenant"
DEFAULT_DATABASE = "default_database"
DEFAULT_TIMEOUT_SECONDS = 20.0


class ChromaHttpError(RuntimeError):
    """A remote Chroma request failed."""


class CollectionNotFoundError(ChromaHttpError):
    """The configured remote collection does not exist."""


@dataclass(frozen=True)
class HttpCollection:
    """Collection-shaped object exposing the part of the API retrieval uses."""

    client: "ChromaHttpClient"
    collection_id: str
    name: str
    metadata: dict[str, Any]

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
        where: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return a Chroma query response for one or more query vectors."""
        body: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include,
        }
        if where is not None:
            body["where"] = where
        path = self.client._collection_path(self.collection_id, "query")
        return self.client._request("POST", path, body)

    def get(
        self,
        *,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return records matching a metadata filter (no vector search)."""
        body: dict[str, Any] = {"include": include or ["documents", "metadatas"]}
        if where is not None:
            body["where"] = where
        if limit is not None:
            body["limit"] = limit
        path = self.client._collection_path(self.collection_id, "get")
        return self.client._request("POST", path, body)


class ChromaHttpClient:
    """Client for the subset of Chroma v2 needed by ``grow-ask``."""

    is_http = True

    def __init__(self, settings: Settings) -> None:
        self.base_url = self._base_url(settings)
        self.tenant = (settings.vector_tenant or DEFAULT_TENANT).strip()
        self.database = (settings.vector_database or DEFAULT_DATABASE).strip()
        self.api_key = settings.vector_api_key

        if "trychroma.com" in urlparse(self.base_url).netloc.lower():
            missing = [
                name
                for name, value in (
                    ("CHROMA_API_KEY", self.api_key),
                    ("CHROMA_TENANT", settings.vector_tenant),
                    ("CHROMA_DATABASE", settings.vector_database),
                )
                if not value
            ]
            if missing:
                raise ChromaHttpError(
                    "Chroma Cloud requires " + ", ".join(missing)
                )

        timeout_text = (os.getenv("CHROMA_TIMEOUT_SECONDS") or "").strip()
        self.timeout = float(timeout_text) if timeout_text else DEFAULT_TIMEOUT_SECONDS
        if self.timeout <= 0:
            raise ValueError("CHROMA_TIMEOUT_SECONDS must be positive")

    @staticmethod
    def _base_url(settings: Settings) -> str:
        host = (settings.vector_host or "").strip().rstrip("/")
        if not host:
            raise ValueError("CHROMA_HOST is required for remote retrieval")

        scheme = "https" if settings.vector_ssl else "http"
        candidate = host if "://" in host else f"{scheme}://{host}"
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid CHROMA_HOST: {host!r}")
        scheme = parsed.scheme

        netloc = parsed.netloc
        try:
            has_port = parsed.port is not None
        except ValueError as exc:
            raise ValueError(f"Invalid CHROMA_HOST port: {host!r}") from exc
        default_port = 443 if scheme == "https" else 80
        if not has_port and settings.vector_port != default_port:
            netloc = f"{netloc}:{settings.vector_port}"

        return f"{scheme}://{netloc}{parsed.path.rstrip('/')}"

    def _scope_path(self) -> str:
        tenant = quote(self.tenant, safe="")
        database = quote(self.database, safe="")
        return f"/api/v2/tenants/{tenant}/databases/{database}/collections"

    def _collection_path(self, collection_id: str, suffix: str = "") -> str:
        path = f"{self._scope_path()}/{quote(collection_id, safe='')}"
        return f"{path}/{suffix}" if suffix else path

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        if self.api_key:
            headers["X-Chroma-Token"] = self.api_key

        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                raise CollectionNotFoundError("Remote collection was not found") from exc
            raise ChromaHttpError(f"Chroma request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise ChromaHttpError(f"Could not reach Chroma: {exc.reason}") from exc

        if not raw:
            return {}
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ChromaHttpError("Chroma returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ChromaHttpError("Chroma returned an unexpected response")
        return result

    def get_collection(self, name: str) -> HttpCollection:
        """Return an existing collection by name, without creating one."""
        result = self._request("GET", self._collection_path(name))
        metadata = result.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        collection_id = str(result.get("id") or name)
        return HttpCollection(
            client=self,
            collection_id=collection_id,
            name=str(result.get("name") or name),
            metadata=metadata,
        )
