# Sample Q&A recorded from traces

These seven answers were copied verbatim from `data/traces/<timestamp>.json`.
They show the answer string a user sees, the policy class recorded in the trace,
and the trace file that holds the retrieval evidence.

**Note:** no provider key was configured during the local run, so the generator
was a deterministic stub that returned the grounded draft shown here. Retrieval,
policy classification, citation allowlists, the three-sentence cap, and trace
writing were the real pipeline. Set `GROQ_API_KEY` and re-run `grow-ask` to
refresh these samples with live-provider wording.

---

## 1. Expense ratio (answer)

**Question:** Expense ratio of HDFC Large Cap Fund Direct Growth?

**Trace:** `data/traces/2026-11-26T18-32-16-859112Z.json`

```text
Expense ratio is 0.99%.
Source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
Last updated from sources: 2026-11-26
```

## 2. ELSS lock-in (answer)

**Question:** What is the ELSS lock-in period?

**Trace:** `data/traces/2026-11-26T18-32-16-906030Z.json`

```text
The statutory lock-in period is 3 years.
Source: https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
Last updated from sources: 2026-11-26
```

## 3. Exit load (answer)

**Question:** Exit load of HDFC Mid Cap Fund Direct Growth?

**Trace:** `data/traces/2026-11-26T18-32-16-954696Z.json`

```text
Exit load is 1% if redeemed within 1 year.
Source: https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth
Last updated from sources: 2026-11-26
```

## 4. Minimum SIP (answer)

**Question:** Minimum SIP for HDFC Large Cap Fund Direct Growth?

**Trace:** `data/traces/2026-11-26T18-32-16-997808Z.json`

```text
Minimum SIP is ₹100.
Source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
Last updated from sources: 2026-11-26
```

## 5. Benchmark (answer)

**Question:** Benchmark of HDFC Flexi Cap Direct Plan Growth?

**Trace:** `data/traces/2026-11-26T18-32-17-043636Z.json`

```text
Benchmark is NIFTY 500 Total Return Index.
Source: https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth
Last updated from sources: 2026-11-26
```

## 6. Advice request (refuse)

**Question:** Should I buy HDFC Flexi Cap?

**Trace:** `data/traces/2026-11-26T18-32-17-089620Z.json`

```text
I can't provide investment advice.
Source: https://investor.sebi.gov.in/ssis/investing-through-mutual-funds-01.html
Last updated from sources: 2026-11-26
```

## 7. Returns ranking (factsheet redirect)

**Question:** Which of these four had the highest return?

**Trace:** `data/traces/2026-11-26T18-32-17-125341Z.json`

```text
Returns are not calculated here.
Source: https://www.hdfcfund.com/mutual-funds/factsheets
Last updated from sources: 2026-11-26
```

---

## What a reviewer should check

- Each answer has exactly one URL from either the retrieved chunk metadata or
  the approved catalog.
- Each `Last updated from sources` line matches `data/corpus.json`, not the
  clock.
- The refuse and factsheet answers contain no comparison or computed figure.
- The trace's `chunk_ids` and `scores` show retrieval separately from the final
  sentence.
