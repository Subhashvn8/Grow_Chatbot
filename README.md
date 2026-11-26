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

## Vercel deployment

Vercel uses the root `app.py` entrypoint and installs `requirements-vercel.txt`
instead of the local ingestion dependencies. Set these environment variables in
the Vercel project:

```text
GROQ_API_KEY=your_groq_key
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_openai_key
CHROMA_HOST=your_chroma_host
CHROMA_PORT=443
CHROMA_SSL=true
CHROMA_COLLECTION=hdfc_mf_facts
CHROMA_TENANT=your_tenant
CHROMA_DATABASE=your_database
CHROMA_API_KEY=your_chroma_key
TRACE_ENABLED=false
```

Remote retrieval calls the Chroma v2 REST API directly, so `chromadb` is not
installed on Vercel. `CHROMA_HOST` may be a hostname such as `api.trychroma.com`
or a full URL. Chroma Cloud needs `CHROMA_TENANT` and `CHROMA_DATABASE`; a local
Chroma server defaults to `default_tenant` / `default_database`.

`LLM_PROVIDER`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`, `VECTOR_METRIC`, `TOP_K`,
`TRACE_DIR`, `CORPUS_FILE`, and `CHROMA_TIMEOUT_SECONDS` can also be overridden.
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

- This is a local prototype, not a deployed service or a financial product.
- Only the five configured pages are ingested; factsheets are linked, not
  parsed into facts.
- Answers are capped at three sentences and one citation. They are not a
  substitute for the official scheme documents.
- The source snapshot date comes from `data/corpus.json`; then answers can go
  stale as the web pages change. Re-run `grow-ingest` to refresh.
- A live LLM answer requires a valid provider key in `.env`.
- No user identity, profile, or long-term chat log is kept.
