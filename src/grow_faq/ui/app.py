"""One Stitch-styled page that calls the ask function and nothing else.

The page never scrapes, embeds, or follows links. It renders the exact string
``grow-ask`` prints, with the single citation turned into a clickable link.
The visual source of truth is ``chat_page.html``; see Docs/architecture.md.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from grow_faq.config import Settings, load_settings
from grow_faq.query.ask import AskStageError, run_ask, safe_error_detail
from grow_faq.query.retrieve import StaleModelError

app = FastAPI(
    title="HDFC Mutual Fund FAQ Assistant",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

EXAMPLE_QUESTIONS = (
    "Expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the ELSS lock-in period?",
    "Exit load of HDFC Mid Cap Fund Direct Growth?",
)
DISCLAIMER = "Facts-only. No investment advice."

_settings: Settings | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def configure(settings: Settings) -> None:
    """Set the validated settings used by the page and the ask endpoint."""
    global _settings
    _settings = settings


def current_settings() -> Settings:
    return _settings if _settings is not None else load_settings()


def welcome_line(settings: Settings) -> str:
    names = [page.name for page in settings.pages]
    amc, schemes = names[0], names[1:]
    return (
        f"{amc} FAQ Assistant — facts from the AMC page and four "
        f"direct-growth schemes: {', '.join(schemes)}."
    )


PAGE_TEMPLATE_PATH = Path(__file__).with_name("chat_page.html")


def page_template() -> str:
    """Read the Stitch-derived page template from the UI package."""
    return PAGE_TEMPLATE_PATH.read_text(encoding="utf-8")


def short_scheme_name(name: str) -> str:
    """'HDFC Flexi Cap Direct Plan Growth' -> 'Flexi Cap' for compact cards."""
    short = re.sub(r"^HDFC\s+", "", name)
    short = re.sub(r"\s+(?:Fund\s+)?Direct(?:\s+Plan)?\s+Growth$", "", short)
    return short.strip() or name


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the one-page UI with settings-derived labels."""
    settings = current_settings()
    page = page_template()
    # pages[0] is the AMC page; pages[1:] are the four schemes.
    for number, scheme in enumerate(settings.pages[1:5], start=1):
        page = page.replace(f"__SCHEME_{number}_SHORT__", html.escape(short_scheme_name(scheme.name)))
        page = page.replace(f"__SCHEME_{number}__", html.escape(scheme.name))
    page = (
        page
        .replace("__WELCOME__", html.escape(welcome_line(settings)))
        .replace("__EXAMPLE_1__", html.escape(EXAMPLE_QUESTIONS[0]))
        .replace("__EXAMPLE_2__", html.escape(EXAMPLE_QUESTIONS[1]))
        .replace("__EXAMPLE_3__", html.escape(EXAMPLE_QUESTIONS[2]))
        .replace("__DISCLAIMER__", html.escape(DISCLAIMER))
    )
    return HTMLResponse(page)


@app.post("/ask")
def ask(payload: AskRequest):
    """Call the same ask function the CLI uses and return its exact text."""
    settings = current_settings()
    try:
        result = run_ask(payload.question.strip(), settings=settings)
    except StaleModelError as exc:
        return JSONResponse(
            {
                "text": str(exc),
                "citation": None,
                "policy": "error",
                "pii": False,
                "error": True,
            },
            status_code=503,
        )
    except AskStageError as exc:
        return JSONResponse(
            {
                "text": "The answer could not be generated.",
                "citation": None,
                "policy": "error",
                "pii": False,
                "error": True,
                "stage": exc.stage,
                "detail": exc.detail,
            },
            status_code=500,
        )
    except Exception as exc:
        return JSONResponse(
            {
                "text": "The answer could not be generated.",
                "citation": None,
                "policy": "error",
                "pii": False,
                "error": True,
                "stage": "unknown",
                "detail": safe_error_detail(exc),
            },
            status_code=500,
        )

    return {
        "text": result.text,
        "citation": result.citation,
        "policy": result.policy,
        "pii": result.pii,
        "trace_path": str(result.trace_path) if result.trace_path else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on")
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--env", type=Path, default=None, help="path to .env")
    args = parser.parse_args(argv)

    settings = load_settings(config_path=args.config, env_path=args.env)
    configure(settings)

    try:
        import uvicorn
    except ImportError:
        print(
            "[fail] uvicorn is required for grow-ui. Install the project "
            "dependencies with 'pip install -e .'.",
            file=sys.stderr,
        )
        return 1

    print(f"UI: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
