# Product Requirements Document

## HDFC Mutual Fund FAQ Assistant (Facts-Only RAG)

**Status:** Draft  
**Source:** `Docs/ProblemStatement.txt`  
**Product type:** Working prototype — small-corpus retrieval assistant  
**Audience for this doc:** Builder of the prototype (phased implementation)

---

## 1. Problem

Retail users and support/content teams repeatedly look up the same factual details about HDFC mutual fund schemes: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements. Those facts live on official public pages, but they are scattered and easy to answer incorrectly or with advice.

This prototype answers those facts from a fixed set of public Groww pages, cites one source on every answer, and refuses anything that is investment advice, a performance comparison, or personal data.

## 2. Goals

1. Answer factual scheme questions from a small, explicit RAG corpus.
2. Show one citation link on every answer.
3. Refuse opinion, portfolio, and returns questions with a short, polite, facts-only message and one educational link.
4. Store each RAG stage as inspectable data so the demo can show loading, chunking, embedding, retrieval, and generation separately.
5. Keep models, vector store, and provider settings in a config file so they can be changed without code edits.

## 3. Non-goals

- Investment advice, buy/sell recommendations, or portfolio construction.
- Computing, ranking, or comparing returns or performance.
- Scraping pages linked from the five seed URLs.
- Screenshots or data from the Groww app back-end.
- Third-party blogs as sources.
- Accepting or storing PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers.
- A production support desk, user accounts, or chat history tied to a person.

## 4. Users

| User | Need |
| --- | --- |
| Retail investor comparing schemes | A short factual answer with a link to the public page |
| Support / content team | Consistent answers to repetitive mutual-fund questions, with a source they can check |

## 5. Scope

### 5.1 Ingestion corpus (exactly these five pages)

Do not follow links inside these pages. Ingest only the page itself.

| # | Page | URL |
| --- | --- | --- |
| 1 | AMC overview — HDFC Mutual Fund | https://groww.in/mutual-funds/amc/hdfc-mutual-funds |
| 2 | Large Cap — HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 3 | Flexi Cap — HDFC Flexi Cap Direct Plan Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 4 | ELSS — HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| 5 | Mid Cap — HDFC Mid Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |

### 5.2 Schemes in scope

- HDFC Mutual Fund (AMC overview)
- HDFC Large Cap Fund Direct Growth
- HDFC Flexi Cap Direct Plan Growth
- HDFC ELSS Tax Saver Fund Direct Plan Growth
- HDFC Mid Cap Fund Direct Growth

### 5.3 Question types in scope

- Expense ratio
- Exit load
- Minimum SIP
- Lock-in (ELSS)
- Riskometer
- Benchmark
- How to download statements (including capital-gains statement)

### 5.4 Sources beyond the five pages

The assistant may cite one additional **official public** URL when the five pages do not contain the fact (for example an AMC factsheet, AMFI, or SEBI investor-education page). Those URLs are listed in the source list deliverable. They are not scraped into the corpus unless they are one of the five pages above.

The source-list deliverable is 15–25 URLs: the five ingested pages plus official links used for citations, factsheet redirects, and refusal education.

## 6. User experience

Tiny single-screen UI:

- Welcome line naming the assistant and the HDFC schemes it covers.
- Three example questions a user can click (factual, from the in-scope list).
- Persistent note: **“Facts-only. No investment advice.”**
- Input for one question and a response area.
- Every answer shows:
  - At most three sentences.
  - Exactly one clear citation link.
  - A line: `Last updated from sources: <date>`.
- Refusals use the same length limit, stay polite, give no opinion, and include one relevant educational link.
- No login, no profile fields, and no inputs for identity or contact data.

### 6.1 Disclaimer (UI copy)

Use this snippet in the UI (wording may be tightened, meaning may not):

> Facts-only. No investment advice. Answers are short factual lookups from public pages, with one source link. This assistant does not recommend buying, selling, or holding any scheme, and does not compare returns.

## 7. Answer policy

The model must identify the exact fact asked, then choose **answer** or **refuse**.

### 7.1 Answer

Answer when the user asks for a fact that exists in the retrieved chunks (or, for statement-download and factsheet requests, when an official public link is the correct response).

Rules:

- State only the fact asked.
- Three sentences or fewer.
- One citation URL, visible as a link.
- If the fact is missing from the corpus, say so and link the official factsheet or the scheme page. Do not invent a number.
- If asked for returns or performance, do not compute or compare. Link the official factsheet.

### 7.2 Refuse

Refuse opinionated or portfolio questions, including “Should I buy/sell?”, “Is this better than…?”, “Which fund should I pick?”, and any request for advice.

Refusal shape:

- Polite, facts-only, no recommendation.
- Three sentences or fewer.
- One relevant educational link (SEBI / AMFI / AMC investor page from the source list).

### 7.3 Reject personal data

If the user submits PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers:

- Do not store the message content that contains those values.
- Do not echo the identifier back.
- Reply that the assistant does not take personal identifiers and ask for a factual scheme question only.

## 8. RAG pipeline

This is a RAG demo. Each stage writes an explicit artifact that can be opened during the demo. Stages run in order and can be re-run independently.

```text
5 public pages
    → raw documents          (phase 1)
    → chunks + metadata      (phase 2)
    → embeddings             (phase 3)
    → Chroma collection      (phase 4)
    → retrieved chunks       (query time)
    → grounded answer + 1 URL
```

### Phase 1 — Data loading

- Fetch only the five URLs in section 5.1.
- Save raw text (and the source URL, fetch time, and HTTP status) under a data directory, one file per page.
- Do not crawl outbound or in-page links.
- Record a corpus “last updated” timestamp used later in answers.

### Phase 2 — Chunking

Chunk by the shape of the data, not by a fixed token window alone.

These pages are scheme fact sheets: labeled facts (expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark) plus short sections (fund overview, how to download statements). The chunking strategy is **section-and-field chunks**:

- One chunk per labeled fact when the page exposes a label and value.
- One chunk per narrative section (overview, statement-download steps) when the text is prose.
- Overlap only inside long prose sections. Do not merge two schemes into one chunk.
- Every chunk stores metadata: `source_url`, `scheme_or_amc`, `section`, `field` (when applicable), `fetched_at`.

Rationale: questions name a single field. Field-sized chunks keep retrieval on that fact and make the citation URL unambiguous.

Persist chunks as JSON (or JSONL) before embedding so the demo can show the chunk stage without opening the vector store.

### Phase 3 — Embedding

- Default model: `sentence-transformers/all-MiniLM-L6-v2`.
- The model name lives in config and is the only embedding model used for both ingestion and query.
- Persist embedding ids alongside chunk ids so a chunk can be traced to a vector.

### Phase 4 — Vector database

- Default store: ChromaDB (local, persistent).
- Collection name, persist path, and distance metric live in config.
- Upsert is idempotent on chunk id so re-running ingestion replaces stale chunks.

### Query path — Retrieval and answer

- Embed the user question with the same model as ingestion.
- Retrieve top-k chunks from the same Chroma collection (k in config).
- Pass only those chunks to the LLM, with instructions to:
  - answer the exact fact or refuse (section 7),
  - stay within three sentences,
  - emit exactly one citation URL drawn from chunk metadata or the approved source list,
  - add `Last updated from sources: <corpus date>`.
- Save the retrieval trace for the demo: question, chunk ids, scores, source URLs, and final answer.

## 9. Configuration

A single config file (for example `config.yaml`) holds values that must be changeable without code changes:

| Setting | Default | Notes |
| --- | --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Shared by ingestion and retrieval |
| Vector DB | ChromaDB | Persist path and collection name configurable |
| Top-k | Small integer (start at 4) | Retrieval breadth |
| LLM provider | `groq` | Also selectable: `gemini`, `openai`, `anthropic`, `mistral` |
| LLM model | Provider-specific model id | Set in config, not hard-coded in prompts |
| Corpus URLs | The five URLs in section 5.1 | List in config so scope stays explicit |
| Answer max sentences | 3 | Enforced in the prompt and checked before display |

**Secrets.** API keys are read from environment variables (for example `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`). Keys are never written into the config file, source code, sample Q&A, or this PRD. If a key was pasted into notes or chat, rotate it and treat it as exposed.

Switching provider or embedding model is a config change followed by re-embedding when the embedding model changes. Changing only the LLM does not require re-embedding.

## 10. Functional requirements

| ID | Requirement |
| --- | --- |
| FR-1 | Load and store the five public pages and no other scraped pages. |
| FR-2 | Chunk with the section-and-field strategy and persist chunks before embedding. |
| FR-3 | Embed with the configured sentence-transformer model and store vectors in the configured Chroma collection. |
| FR-4 | Retrieve with the same embedding model and the same collection used at ingestion. |
| FR-5 | Answer in-scope factual questions in ≤3 sentences with exactly one citation link. |
| FR-6 | Refuse advice and portfolio questions with a polite facts-only message and one educational link. |
| FR-7 | On returns or performance questions, do not calculate or compare; link the official factsheet. |
| FR-8 | Show `Last updated from sources: <date>` on every answer. |
| FR-9 | UI shows a welcome line, three example questions, and the facts-only disclaimer. |
| FR-10 | Do not accept or persist PII. Drop identifier-bearing input from stored traces. |
| FR-11 | Each pipeline stage leaves an artifact that can be shown in the demo. |
| FR-12 | Provider, model, embedding model, and vector DB settings come from config. |

## 11. Non-functional requirements

- **Traceability.** Any answer can be mapped to chunk ids and one URL.
- **Reproducibility.** Re-running load → chunk → embed → index from config rebuilds the corpus.
- **Transparency.** Answers stay within three sentences and always show the source date line.
- **Local-first corpus.** Chroma persists on disk; no third-party blog index.
- **No secrets in repo.** `.env` is gitignored. Config holds non-secret settings only.

## 12. Deliverables

| Deliverable | Acceptance |
| --- | --- |
| Working prototype | Runnable app, or a notebook, with the UI in section 6. If it cannot be hosted, a demo video of 3 minutes or less. |
| Source list | CSV or Markdown of the 15–25 official URLs used (5 ingested pages plus citation and education links). |
| README | Setup steps, scope (AMC + four schemes), config, env vars, and known limits. |
| Sample Q&A | 5–10 queries with the assistant’s answers and links, covering both answers and at least one refusal. |
| Disclaimer | The facts-only snippet from section 6.1, as shown in the UI. |
| Stage artifacts | Raw pages, chunks, and a way to inspect Chroma / retrieval traces. |

## 13. Acceptance scenarios

1. **Expense ratio.** “Expense ratio of HDFC Large Cap Fund Direct Growth?” returns the figure from that scheme’s page, one link to that page, and the last-updated line. No other scheme’s ratio appears.
2. **ELSS lock-in.** “ELSS lock-in?” returns the lock-in fact for HDFC ELSS Tax Saver with that scheme’s URL.
3. **Minimum SIP / exit load / riskometer / benchmark.** Each returns only the asked field, with one citation.
4. **Statements.** “How to download capital-gains statement?” returns the documented steps or points to the official page that describes them, with one link.
5. **Advice.** “Should I buy HDFC Flexi Cap?” is refused, with no recommendation and one educational link.
6. **Returns.** “Which of these four had the highest return?” does not compute a comparison; it points to the official factsheet.
7. **Missing fact.** A fact not present in the chunks is not invented.
8. **PII.** A question that includes a PAN or phone number is not stored and is not repeated in the reply.
9. **Config.** Changing the LLM provider in config routes generation to that provider. Changing the embedding model is documented as requiring a re-index.

## 14. Known limits (document in README)

- Corpus is five Groww public pages for one AMC and four direct-growth schemes. Other AMCs, regular plans, and IDCW plans are out of scope.
- Facts go stale until the load phase is run again. The UI date line is the corpus fetch time, not a live NAV.
- Groww page layout can change and break extraction; failed fetches are reported, not silently skipped.
- The assistant is not a SEBI-registered adviser. The disclaimer stays visible.
- Statement-download flows that sit behind login are described only from public instructions. The assistant never asks for credentials.

## 15. Phased build order

1. Config and data loading (five pages → raw artifacts).
2. Chunking (section-and-field → chunk artifacts).
3. Embedding (`all-MiniLM-L6-v2` by default).
4. ChromaDB index.
5. Retrieval + LLM answer/refuse policy, with stored retrieval traces.
6. Tiny UI, disclaimer, sample Q&A, source list, README.

Do not start the UI before a single factual question and a single refusal both pass against the indexed corpus.

## 16. Open decisions

| Item | Proposal |
| --- | --- |
| Default Groq model id | Set in config when the Groq account’s available models are confirmed. Do not commit a key to choose it. |
| Top-k and distance metric | Start with k=4 and cosine; tune only if sample Q&A cites the wrong page. |
| Where official factsheet URLs come from | Curate them in the source list. Do not discover them by crawling the five pages. |
| UI stack | Any small local web UI or notebook that meets section 6. Prefer a local web page if a prototype link is required. |
