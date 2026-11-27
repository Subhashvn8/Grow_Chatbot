# HDFC Mutual Fund FAQ Assistant

A local, facts-only RAG prototype. It reads five public Groww pages for HDFC
Mutual Fund and four direct-growth schemes, stores section-and-field facts in a
local Chroma collection, and answers one question at a time with one cited
source. It does not give investment advice and does not calculate returns.

## Scope

**In scope**

- HDFC Mutual Fund AMC page
- HDFC Large Cap Fund Direct Growth
- HDFC Flexi Cap Direct Plan Growth
- HDFC ELSS Tax Saver Fund Direct Plan Growth
- HDFC Mid Cap Fund Direct Growth
- Factual fields: expense ratio, exit load, minimum SIP, lock-in, riskometer,
  benchmark, and statement download
- Polite refusals for advice questions; official factsheet links for returns
  questions

**Out of scope**

- Any other Groww page, link found on a page, or private Groww API
- Buy/sell/hold recommendations, fund rankings, or computed returns
- NAV data, portfolio data, user accounts, personal identifiers, or chat history

## Setup

Python 3.11+ is required.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the Groq client used by default. For another LLM provider, set
it in `config.yaml` and install its extra: `pip install -e .[gemini]`,
`.[openai]`, `.[anthropic]`, `.[mistral]`, or `.[all]`.

Copy `.env.example` to `.env` and set the key for the provider selected in
`config.yaml` (Groq by default):

```text
GROQ_API_KEY=your_key_here
```

`.env` is gitignored. Never put a key in `config.yaml`, source code, or traces.

## Ingest

Run the stages once, or use the wrapper:

```bash
grow-scrape     # five allowlisted pages -> data/raw + data/corpus.json
grow-load       # data/raw -> data/documents
grow-chunk      # documents -> data/chunks/chunks.jsonl
grow-embed      # chunks -> data/embeddings/embeddings.jsonl
grow-index      # chunks + embeddings -> data/chroma

grow-ingest     # all five stages; stops on the first failure
```

Ingestion never calls an LLM. A failed scrape or an empty document stops the
pipeline before chunking. Re-running `grow-index` upserts by chunk id and
removes ids that are no longer in `chunks.jsonl`.

## Ask

```bash
grow-ask "Expense ratio of HDFC Large Cap Fund Direct Growth?"
grow-ask "Should I buy HDFC Flexi Cap?"
```

For the local page:

```bash
grow-ui
# open http://127.0.0.1:8000
```

The page calls the same `run_ask` function as the CLI. It does not scrape,
embed, or fetch any web page. It has one question field, three example buttons,
a permanent “Facts-only. No investment advice.” disclaimer, and a response with
one link and the corpus date line.

The chat layout is adapted from the Google Stitch export in
`Docs/design/stitch_cosmic_groww_ai_chatbot.zip`. The visual tokens and retained
source files are recorded in `Docs/architecture.md` §4.10.1 and
`Docs/implementation.md` §8.1; the rendered template is
`src/grow_faq/ui/chat_page.html`. Those two documents are the main source of
truth for the UI contract.

## Query guardrails

1. **PII gate** — PAN, Aadhaar, phone, email, account-like digit strings, and
   OTPs are rejected before retrieval, the LLM, or a trace. Nothing is echoed
   or stored.
2. **Policy** — questions are classified as `answer`, `factsheet`, or `refuse`.
3. **Retrieval** — the question is embedded with the configured model, the
   stored model name is checked, and Chroma is queried with `top_k` from config.
   One named scheme filters on `scheme_or_amc`; two schemes or none search the
   whole collection.
4. **Generation** — the LLM sees only retrieved chunks. The prompt requires at
   most three sentences, exactly one allowed URL, and the `corpus.json` date.
5. **Check** — drafts with a disallowed URL, extra URLs, more than three
   sentences, advice language, or a computed return are replaced by a grounded
   template.
6. **Trace** — non-PII turns write `data/traces/<timestamp>.json` with the
   question, policy, chunk ids, scores, citation, and final answer.

If the collection's embedding model differs from config, the assistant stops
and asks the operator to run `grow-embed` and `grow-index`. If the provider
fails, the answer is “The answer could not be generated.” — never a guessed
number.

## Configuration

`config.yaml` holds all non-secret settings: five page URLs, embedding model,
Chroma path/collection/metric, `top_k`, LLM provider and model, `answer` sentence
cap, source catalog path, and output paths. `config.py` is the only reader.

Changing the LLM provider or model does not rebuild vectors. Changing the
embedding model requires `grow-embed` and `grow-index` again. The citation
allowlist lives in `Docs/sources.md` and is edited by hand; the loader never
adds to it.

## Render deployment

Use a Python **Web Service** with these settings:

```text
Root Directory: .        # repository root
Runtime: Python 3.12
Build Command: pip install -r requirements.txt && PYTHONPATH=src python -m grow_faq.embedding
Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
```

The second half of the build command downloads the embedding model into
`.cache/models` inside the project, so the first question after a deploy does
not wait for a Hugging Face download. `requirements.txt` installs the CPU-only
PyTorch build and pins `chromadb` to the version that wrote `data/chroma`.

Set only this environment variable:

```text
GROQ_API_KEY=your_groq_key
```

Do **not** set `CHROMA_HOST`, `EMBEDDING_PROVIDER`, or `OPENAI_API_KEY` on
Render. The committed `data/chroma` index and local sentence-transformers are
used instead. Choose a plan with at least 2 GB RAM because `torch` is loaded at
runtime. Traces are written to `data/traces` and disappear on redeploy unless a
Render Disk is attached.

## Vercel deployment

Vercel uses the root `app.py` entrypoint and installs `requirements-vercel.txt`
instead of the local ingestion dependencies. Only these values must be set in
the Vercel project:

```text
GROQ_API_KEY=your_groq_key

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_openai_key

CHROMA_HOST=your_chroma_host
CHROMA_API_KEY=your_chroma_key
CHROMA_TENANT=your_tenant
CHROMA_DATABASE=your_database
```

A fill-in template is available at `.env.example`. Replace every `CHANGE_ME`
value before importing it into Vercel.

Everything else already has a default in `config.yaml` or `Settings`:
`CHROMA_COLLECTION=hdfc_mf_facts`, `CHROMA_PORT=443`, `CHROMA_SSL=true`,
`LLM_PROVIDER=groq`, `LLM_MODEL=openai/gpt-oss-20b`, and `TOP_K=4`. Override
them only if your deployment differs. `TRACE_ENABLED` defaults to `true`; set it
to `false` on Vercel to skip ephemeral trace writes, though those writes are
non-fatal.

Remote retrieval calls the Chroma v2 REST API directly, so `chromadb` is not
installed on Vercel. `CHROMA_HOST` is mandatory there; without it the app fails
with an explicit operator error instead of trying to open local files. It may be
a hostname such as `api.trychroma.com` or a full URL. Chroma Cloud needs
`CHROMA_TENANT` and `CHROMA_DATABASE`; a self-hosted Chroma server defaults to
`default_tenant` / `default_database` and may not need an API key.

When the embedding model changes, re-run ingestion offline and upload the
vectors to the remote store.

## Data layout

```text
data/
  raw/               scraped HTML and sidecar metadata
  documents/         extracted visible text per page
  chunks/            chunks.jsonl
  embeddings/        embeddings.jsonl
  chroma/            persistent Chroma database
  traces/            one JSON per answered, non-PII question
  corpus.json        fetch status and last successful update
```

## Documentation

- `Docs/architecture.md` — component and data contracts
- `Docs/implementation.md` — phased build order
- `Docs/sources.md` — curated citation allowlist (20 official URLs)
- `Docs/sample_qa.md` — answers copied from recorded traces

## Tests

```bash
.venv\Scripts\python -m unittest discover -s tests -v
```

The tests cover all ingestion stages, indexing, PII, policy, retrieval,
generation guardrails, traces, and the UI page/endpoint.

## Limits

- This is a prototype, not a financial product. The Render and Vercel setups
  above are demo deployments with no authentication or rate limiting.
- Only the five configured pages are ingested; factsheets are linked, not
  parsed into facts.
- Answers are capped at three sentences and one citation. They are not a
  substitute for the official scheme documents.
- The source snapshot date comes from `data/corpus.json`; then answers can go
  stale as the web pages change. Re-run `grow-ingest` to refresh.
- A live LLM answer requires a valid provider key in `.env`.
- No user identity, profile, or long-term chat log is kept.
