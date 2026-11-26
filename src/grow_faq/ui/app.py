"""Phase 8: one local page that calls the ask function and nothing else.

The page never scrapes, embeds, or follows links. It renders the exact string
``grow-ask`` prints, with the single citation turned into a clickable link.
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
_SOURCE_LINE = re.compile(
    r"^(Source|Education source|Official factsheet):\s+(https?://\S+?)([.,;:]?\s*)$"
)

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


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HDFC Mutual Fund FAQ Assistant</title>
<style>
  :root { color-scheme: light; --ink:#152238; --muted:#5b6779; --line:#d8deea; --accent:#0b5dab; }
  * { box-sizing: border-box; }
  body { margin:0; background:#f4f7fb; color:var(--ink);
         font:16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 760px; margin: 40px auto; padding: 28px;
         background:#fff; border:1px solid var(--line); border-radius:14px;
         box-shadow:0 8px 24px rgba(20,40,80,.06); }
  h1 { font-size:1.35rem; margin:0 0 6px; }
  .welcome { margin:0 0 18px; color:var(--muted); }
  .examples { margin:0 0 20px; }
  .examples span { display:block; font-size:.85rem; text-transform:uppercase;
                   letter-spacing:.06em; color:var(--muted); margin-bottom:8px; }
  .example { display:inline-block; margin:0 8px 8px 0; padding:7px 11px;
             background:#eef4fc; color:var(--accent); border:1px solid #cbdcf1;
             border-radius:999px; cursor:pointer; font:inherit; font-size:.9rem; }
  .example:hover { background:#dfeaf8; }
  form { display:flex; gap:8px; }
  input[type=text] { flex:1; min-width:0; padding:11px 12px; font:inherit;
                     border:1px solid var(--line); border-radius:8px; }
  button.ask { padding:11px 18px; font:inherit; color:#fff; background:var(--accent);
               border:0; border-radius:8px; cursor:pointer; }
  button:disabled { opacity:.6; cursor:progress; }
  .disclaimer { margin:14px 0 0; padding:9px 12px; background:#fff6e5;
                border-left:4px solid #e0a500; color:#6b4e00; font-weight:600; }
  #response { margin-top:20px; padding:16px; min-height:56px;
              border:1px solid var(--line); border-radius:10px; background:#fbfcfe; }
  #response div { margin:0 0 6px; white-space:pre-wrap; }
  #response div:last-child { margin-bottom:0; }
  #response .date { color:var(--muted); font-size:.9rem; margin-top:10px; }
  #status { color:var(--muted); font-size:.9rem; }
</style>
</head>
<body>
<main>
  <h1>HDFC Mutual Fund FAQ Assistant</h1>
  <p class="welcome" id="welcome">__WELCOME__</p>

  <div class="examples">
    <span>Example questions</span>
    <button type="button" class="example" data-question="__EXAMPLE_1__">__EXAMPLE_1__</button>
    <button type="button" class="example" data-question="__EXAMPLE_2__">__EXAMPLE_2__</button>
    <button type="button" class="example" data-question="__EXAMPLE_3__">__EXAMPLE_3__</button>
  </div>

  <form id="ask-form">
    <label for="question" class="sr-only">Question</label>
    <input id="question" name="question" type="text" autocomplete="off"
           placeholder="Ask a factual question about these schemes" required>
    <button class="ask" id="submit" type="submit">Ask</button>
  </form>

  <p class="disclaimer" id="disclaimer">__DISCLAIMER__</p>
  <div id="response" aria-live="polite"></div>
  <div id="status"></div>
</main>
<script>
(function () {
  const form = document.getElementById('ask-form');
  const input = document.getElementById('question');
  const submit = document.getElementById('submit');
  const response = document.getElementById('response');
  const status = document.getElementById('status');
  const sourceLine = /^(Source|Education source|Official factsheet):\\s+(https?:\\/\\/\\S+?)([.,;:]?\\s*)$/;

  function render(text) {
    response.replaceChildren();
    String(text).split('\\n').forEach(function (line) {
      const row = document.createElement('div');
      const match = sourceLine.exec(line);
      if (match) {
        row.appendChild(document.createTextNode(match[1] + ': '));
        const link = document.createElement('a');
        link.href = match[2];
        link.textContent = match[2];
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        row.appendChild(link);
        row.appendChild(document.createTextNode(match[3]));
      } else {
        row.textContent = line;
        if (line.indexOf('Last updated from sources:') === 0) row.className = 'date';
      }
      response.appendChild(row);
    });
  }

  async function ask(question) {
    const trimmed = question.trim();
    if (!trimmed) return;
    input.value = trimmed;
    status.textContent = 'Retrieving sourced facts…';
    submit.disabled = true;
    try {
      const reply = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: trimmed})
      });
      const data = await reply.json();
      render(data.text || 'The answer could not be generated.');
    } catch (error) {
      render('The answer could not be generated.');
    } finally {
      status.textContent = '';
      submit.disabled = false;
    }
  }

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    ask(input.value);
  });
  document.querySelectorAll('.example').forEach(function (button) {
    button.addEventListener('click', function () {
      ask(button.dataset.question);
    });
  });
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the one-page UI with settings-derived labels."""
    settings = current_settings()
    page = (
        PAGE_TEMPLATE.replace("__WELCOME__", html.escape(welcome_line(settings)))
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
