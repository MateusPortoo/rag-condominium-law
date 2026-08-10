# RAG Condominium Law

> Production-grade RAG pipeline over Brazilian condominium law (Lei 4.591/64 + Código Civil arts. 1.314–1.358). Built to be defended with metrics and a documented debug history — not just to look good.

![CI](https://github.com/MateusPortoo/rag-condominium-law/actions/workflows/pr.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

**Corpus language:** Portuguese (PT-BR) · **Interface language:** English

---

## Architecture

```
Query
  │
  ├─► Prompt Injection Detection
  │
  ├─► Semantic Cache (Qdrant, threshold 0.95)
  │         └─ hit → return cached response
  │
  ├─► Query Transformation
  │         ├─ HyDE (hypothetical document embedding)
  │         └─ Multi-Query (3 variations)
  │
  ├─► Hybrid Retrieval
  │         ├─ Dense search  (OpenAI text-embedding-3-small + Qdrant)
  │         ├─ Sparse search (BM25 via bm25s, in-memory)
  │         └─ RRF fusion    (k=60, manual implementation)
  │
  ├─► Reranking (cross-encoder, top-20 → scored)
  │
  ├─► CRAG — Corrective RAG
  │         ├─ correct   (score > 0.75) → decompose-then-recompose
  │         ├─ ambiguous (0.70–0.75)    → refine + web search (allowlist)
  │         └─ incorrect (no doc ≥ 0.70)→ web search only (allowlist)
  │
  ├─► Model Routing
  │         ├─ simple query  → Groq  llama-3.3-70b-versatile
  │         └─ complex query → OpenAI frontier model
  │
  └─► Generation (temperature=0, structured output)
            └─ SSE streaming (6 named event types)
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.11 | Standard for AI/LLM engineering roles |
| API | FastAPI + uvicorn | Async, typed, production-grade |
| Vector DB | Qdrant Cloud | Full control over hybrid search — no abstraction hiding |
| Sparse search | bm25s (in-memory) | Pluggable BM25, no managed dependency |
| Embeddings | OpenAI text-embedding-3-small | Most cited in job requirements |
| Hybrid fusion | RRF k=60 (manual) | Explainable in interview, no black-box |
| Reranking | cross-encoder (HuggingFace) | Local, swappable — see DC-01 |
| Generation | Groq + OpenAI (routed) | Cost-efficient with quality fallback |
| Evaluation | RAGAS | Dominant framework for RAG metrics |
| CI/CD | GitHub Actions | Industry standard |
| Infra | AWS EC2 + Qdrant Cloud | Defensible cost-vs-control tradeoff |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/MateusPortoo/rag-condominium-law.git
cd rag-condominium-law

# 2. Environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY

# 3. Install
pip install -e ".[dev]"

# 4. Run tests (no API key needed)
pytest tests/ -v

# 5. Ingest corpus (requires API keys)
python scripts/run_ingest.py
# Prints article size distribution → confirm chunk_size → indexes to Qdrant + BM25

# 6. Start API
uvicorn src.rag_condominios.api.main:app --reload
```

---

## Evaluation

Four RAGAS metrics tracked per golden set category:

| Metric | Measures |
|---|---|
| `context_precision` | Of retrieved chunks, how many were relevant? |
| `context_recall` | Of relevant chunks, how many were retrieved? |
| `faithfulness` | Is the answer grounded in the context? (hallucination detection) |
| `answer_relevancy` | Does the answer address the question? |

Run full evaluation:
```bash
# Via API
curl -X POST http://localhost:8000/evaluate

# Via script
python scripts/run_evaluate.py
```

Results are reported per golden set category (simple, multi-chunk, different-vocabulary, ambiguous, no-answer, adversarial).

---

## Debug Cases

### DC-01 — Wrong reranker language (planned → in progress)

**Problem:** `ms-marco-MiniLM-L-6-v2` was trained on English. Corpus is Portuguese.  
**Hypothesis:** Reranker misordered chunks in PT-BR, depressing `context_precision`.  
**Fix:** Swap to `BAAI/bge-reranker-v2-m3` (multilingual).  
**Baseline `context_precision`:** _to be measured_  
**Post-fix `context_precision`:** _to be measured_  
**Note:** Delta is directional given small golden set (~4 examples/category) — not statistically conclusive.

---

## Implementation Roadmap

See [`PHASES.md`](PHASES.md) for the full phased implementation plan.

Current phase: **Phase 1 — Skeleton that produces numbers**
