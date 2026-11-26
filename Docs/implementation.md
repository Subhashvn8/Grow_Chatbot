# Implementation

## HDFC Mutual Fund FAQ Assistant

**Source:** `Docs/architecture.md`

Build in this order. Each step writes files under `data/` so the stage can be shown on its own. Ingestion does not call an LLM. The query path does not fetch the web.

---

## 1. Creating the directory

Python 3.11+. From the repo root:

```text
Grow_Chatbot/
  config.yaml
  .env                    # API keys, gitignored
  .env.example
  requirements.txt
  src/grow_faq/
    config.py
    ingest/               # scrape, load, chunk, embed, index
    query/                # retrieve, policy, answer
    llm/                  # Groq and the other providers
    ui/
  data/
    raw/                  # scraped HTML
    documents/            # loaded text, one file per page
    chunks/
    embeddings/
    chroma/
    traces/
  Docs/
```

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Dependencies: `pyyaml`, `python-dotenv`, `httpx`, `beautifulsoup4`, `sentence-transformers`, `chromadb`, `fastapi`, `uvicorn`. Add the LLM client for the provider in `config.yaml` (`groq` by default; also `gemini`, `openai`, `anthropic`, `mistral`).

`config.yaml` holds the five page URLs, the embedding model (`sentence-transformers/all-MiniLM-L6-v2`), the Chroma path and collection name (`hdfc_mf_facts`), `top_k` (4), the LLM provider and model, and the path to the source list. API keys stay in `.env` (`GROQ_API_KEY` and the other provider keys). `config.py` is the only file that reads the YAML.

---

## 2. Scraping the documents

Fetch only these five pages. Do not open links on them. This step saves the pages. It does not turn them into documents yet.

| Page | URL |
| --- | --- |
| HDFC Mutual Fund | https://groww.in/mutual-funds/amc/hdfc-mutual-funds |
| HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| HDFC Flexi Cap Direct Plan Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| HDFC Mid Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |

Save each response as `data/raw/<slug>.html` plus a sidecar `data/raw/<slug>.meta.json` with `url`, `scheme_or_amc`, `fetched_at`, and `http_status`. Keep the HTML as fetched. Do not call a private Groww API, and do not follow links found in the page.

If any URL is not HTTP 200, write `data/corpus.json` with `"ok": false` and stop. Do not load or chunk a partial set.

```bash
grow-scrape
```

---

## 3. Loading the documents

Read the five files in `data/raw/`. This is the document-loading step. It does not fetch anything, and it does not chunk.

For each saved page, extract visible text: headings, paragraphs, and label/value rows. Drop scripts, nav, footer, and cookie banners. Do not read embedded client JSON.

Write one document per page to `data/documents/<slug>.json`:

- `url`
- `scheme_or_amc` (the name from config, not the page title)
- `fetched_at`
- `text`

Write `data/corpus.json` with `last_updated` (the earliest successful fetch) and a status per document. A document is not ok when its text is almost empty. If any of the five is not ok, set `"ok": false` and stop. Chunking reads this folder only, and only when `ok` is true.

```bash
grow-load
```

---

## 4. Chunking

Read `data/documents/*.json`. Write `data/chunks/chunks.jsonl`, one JSON object per line.

Use section-and-field chunks:

- One chunk per labeled fact: expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark. The text includes the scheme name and the value.
- One chunk per short prose section (overview, how to download a statement).
- Split only a long prose section (over about 180 words), with a short overlap. Never split a fact, and never put two schemes in one chunk.

Each chunk stores `id`, `text`, `source_url`, `scheme_or_amc`, `section`, `field`, and `fetched_at`. The id is stable for the same page and field, so a re-run replaces the same chunk. If a label is missing on a page, skip it. Do not copy the value from another scheme.

```bash
grow-chunk
```

---

## 5. Embedding

Read `chunks.jsonl`. Encode each chunk’s `text` with `sentence-transformers/all-MiniLM-L6-v2` (the model name in config). Normalize the vectors.

Write `data/embeddings/embeddings.jsonl` with the chunk `id`, the model name, the dimension (384 for this model), and the vector. The id list must match the chunk file exactly. Do not call Chroma in this step.

```bash
grow-embed
```

The same model encodes the user’s question later. If the model name in config changes, run embed and index again before asking questions.

---

## 6. Storing data in the vector DB

Read the chunk file and the embedding file. Upsert into a local persistent Chroma collection (`data/chroma`, collection `hdfc_mf_facts`, cosine).

| Chroma field | Value |
| --- | --- |
| id | chunk id |
| document | chunk text |
| embedding | vector from the embeddings file |
| metadata | `source_url`, `scheme_or_amc`, `section`, `field`, `fetched_at`, `embedding_model` |

Metadata values are strings. Upsert on id so a re-run replaces stale facts, and delete ids that are no longer in the chunk file. Store `embedding_model` on the collection. If it does not match config, delete the collection and create it again.

```bash
grow-index
```

`grow-ingest` runs scrape, load, chunk, embed, and index. It stops if scrape or load failed.

---

## 7. Retrieval logic

`grow-ask "question"` runs this path. The UI calls the same function.

1. **PII gate.** If the question contains a PAN, Aadhaar, phone number, email, account number, or OTP, reply that the assistant does not take personal identifiers. Do not call the retriever, the LLM, or the trace writer, and do not echo the identifier.
2. **Classify.**
   - `refuse` — should I, buy, sell, hold, recommend, which is better.
   - `factsheet` — returns, performance, CAGR, which did better.
   - `answer` — a factual field (expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, statement download).
3. **Retrieve.** Embed the question with the same model. Query the same collection, `top_k` 4. If the question names a scheme (Large Cap, Flexi Cap, ELSS or lock-in, Mid Cap, or the AMC), filter on `scheme_or_amc`. If that filter returns nothing, search the whole collection.
4. **Generate.** Send only the retrieved chunks to the configured LLM. The prompt requires at most three sentences, exactly one allowed URL, and the line `Last updated from sources: <date>` using `corpus.json`, not today’s clock.
   - An answer may cite only a retrieved chunk’s `source_url`.
   - A refusal cites one education URL from `Docs/sources.md` (SEBI, AMFI, or the AMC).
   - A returns question cites one official factsheet URL from that list. No return is calculated.
5. **Check.** If the draft has more than three sentences, a URL that is not allowed, or advice language, drop it and use a short template with the top chunk’s URL or the catalog URL.
6. **Trace.** Write `data/traces/<timestamp>.json` with the question, policy, chunk ids, scores, citation, and the final answer. Skip the trace for a PII reject.

If the collection’s embedding model does not match config, stop and tell the operator to re-run embed and index. If the LLM call fails, say the answer could not be generated. Do not fill in a number from memory.

---

## 8. UI

A single local page (`grow-ui`). It only calls the ask function. It does not scrape or embed.

- Welcome line naming HDFC Mutual Fund and the four direct-growth schemes.
- Three example questions: expense ratio of HDFC Large Cap, ELSS lock-in, exit load of HDFC Mid Cap.
- Disclaimer, always visible: “Facts-only. No investment advice.”
- One text box. No name, email, phone, PAN, or account field.
- The reply shows the answer, one link, and the last-updated line.

Build the UI only after the CLI has answered an expense-ratio question and refused a “should I buy” question.

Also finish `Docs/sources.md` (15–25 official URLs), `Docs/sample_qa.md` (5–10 real answers taken from traces), and a README with setup, scope, and limits.

---

## 9. Edge cases and guardrail tests

These are code checks, not model judgment. Run them before treating the prototype as done.

| Case | Input | Expected |
| --- | --- | --- |
| Happy path | “Expense ratio of HDFC Large Cap Fund Direct Growth?” | The figure from that scheme’s chunk, that page’s URL, the corpus date. No other scheme’s ratio. |
| Other fields | Exit load, minimum SIP, riskometer, benchmark, “ELSS lock-in?” | Only the asked field, one citation. Lock-in resolves to the ELSS scheme. |
| Statements | “How to download a capital-gains statement?” | Steps from the chunk, or one official statement URL. No request for a login or PAN. |
| Missing fact | A field that has no chunk | “Not in the sourced pages.” Cite the scheme page. No invented number. |
| Advice | “Should I buy HDFC Flexi Cap?” | Polite refusal, no buy/sell language, one education URL. |
| Returns | “Which of these four had the highest return?” | No comparison and no computed figure. One factsheet URL. |
| Advice phrased as a fact | “Should I buy this for the return?” | Classified as a refusal, not a factsheet. |
| Bad citation | Model draft contains a blog URL | Draft discarded. Template uses a chunk URL or a catalog URL. |
| Too long | Draft longer than three sentences | Replaced by the template. |
| PII | PAN, 10-digit mobile, email, or Aadhaar in the question | Fixed refusal. Identifier not stored and not repeated. No trace file containing it. |
| Failed scrape | One of the five URLs is not 200 | `corpus.json` is not ok. Load, chunk, embed, and index do not run. |
| Empty document | A saved page has almost no visible text | Load sets `ok` to false. Chunking does not run. |
| Stale model | `embedding.model` in config differs from the collection | Ask refuses and names `grow-embed` and `grow-index`. |
| Provider down | Timeout or 401 | “The answer could not be generated.” No guessed fact. |
| Two schemes in one question | Large Cap and Mid Cap both named | No metadata filter. Retrieval searches the whole collection. |

Manual demo check: open one raw HTML file, one loaded document, a few lines of `chunks.jsonl`, one embedding record, one Chroma document, and one trace, and confirm the UI answer matches `grow-ask` for the same question.
