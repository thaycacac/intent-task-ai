# ADR 0001: Hybrid NLP architecture for Intent Task AI

- **Status:** Accepted
- **Date:** 2026-09-19
- **Deciders:** CTO (Thaycacac), plan Accepted on parent epic
- **Context:** DonDon NLP quick-create needs `category`, `deadline`, `task_detail`, `priority` from free-text (VI-first, EN supported). Training corpus is AmazonScience/MASSIVE (`vi-VN` + `en-US`), which labels **intent/slots for voice assistants**, not to-do deadline/priority.

## Decision

Use a **hybrid pipeline** (not a single end-to-end MASSIVE fine-tune for all four fields):

| Stage | Approach | Rationale |
|-------|----------|-----------|
| Category | Supervised classifier trained on MASSIVE intents mapped to DonDon taxonomy (`work\|personal\|errand\|learning\|health\|other`) | MASSIVE has intent labels; mapping is a two-way door |
| Deadline | Rule / dateparser, timezone `Asia/Ho_Chi_Minh` | No deadline labels in MASSIVE; false positives worse than `null` |
| Priority | Keyword / urgency heuristics (+ optional small classifier later) | Synthetic labels only; keep blast radius small |
| `task_detail` | Deterministic normalizer (strip fillers, keep action/object) | Reversible; no LLM required for MVP |

API surface: `POST /v1/parse-task` returns the four fields plus `confidence` and `explanations`.

## Options considered

| Option | Verdict |
|--------|---------|
| A. Fine-tune only MASSIVE classifier | Rejected — cannot produce deadline/priority/`task_detail` |
| B. Hybrid NLU + extractors | **Accepted** — product fields + MLOps train→serve loop |
| C. LLM API prompt→JSON as core | Rejected for core — weak “self-train” lab story; optional later baseline |

## Consequences

- Category quality depends on intent→taxonomy mapping quality; `other` absorbs OOD.
- Deadline/priority evaluated on a VI golden set (≥50), not MASSIVE holdout.
- MVP category model may be a lightweight TF-IDF + linear classifier for CPU train/serve; HF transformer fine-tune remains an optional upgrade path (same contract).
- Observability-first: `/health`, `/metrics`, structured request logs before any cloud deploy.

## Related lenses

- **Two-way vs one-way doors** — mapping and heuristic weights are reversible; API field names are closer to one-way.
- **Build vs buy** — dateparser + sklearn/transformers over custom NER.
- **Reversibility** — artifact versions under `artifacts/`; feature flags not required for MVP single model.
