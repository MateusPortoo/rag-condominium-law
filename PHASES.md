# Roadmap de Implementação

## Princípio central
Caminho fino ponta a ponta primeiro (pergunta → resposta sem sofisticação), porque sem harness de avaliação toda decisão vira "eu acho que melhorou".

---

## Fase 1 — Esqueleto que já produz número
- [ ] **Ingest pipeline** — chunking medido, indexação Qdrant + BM25 ← *em andamento*
- [ ] **Golden set** — 25 casos com `reference_contexts` preenchidos contra chunks reais
- [ ] **Retrieval híbrido básico** — BM25 + dense + RRF, sem reranker, sem CRAG
- [ ] **Geração simples** — query → contexto bruto → resposta (Groq, 1 modelo, temperature=0)

## Fase 2 — Harness de avaliação
- [ ] **RAGAS + /evaluate** — 4 métricas contra golden set no pipeline da Fase 1 (baseline ruim esperado)
- [ ] **Prompt injection detection** — isolado, sem dependência de retrieval

## Fase 3 — Melhorar retrieval com medição
- [ ] **Reranker baseline (ms-marco)** — mede delta de `context_precision` vs Fase 2
- [ ] **CRAG** — 3 vereditos, cross-encoder como evaluator, mede impacto em `ambiguous_forcado` e `sem_resposta`
- [ ] **Query transformations (HyDE + Multi-Query)** — mede delta em categoria `vocabulario_diferente`

## Fase 4 — Robustecer
- [ ] **Model routing** — Groq/OpenAI por heurística (só após geração funcionar com 1 modelo)
- [ ] **Semantic cache** — pipeline estável antes de cachear
- [ ] **Streaming SSE** — sobre pipeline síncrono funcionando

## Fase 5 — Produção e prova de rigor
- [ ] **DC-01** — trocar ms-marco → bge-reranker-v2-m3, documentar delta com número real
- [ ] **Resiliência** — retry, circuit breaker, fallback
- [ ] **CI/CD completo** — gate CRAG, teste falso positivo do cache, lint desde commit 1
- [ ] **Docker Compose + deploy AWS**
- [ ] **Observabilidade**
