# RAG Gestão de Condomínios — Especificação Técnica

**Versão:** 1.0  
**Status:** Fechada para implementação  
**Data:** 2026-08-07

---

## 1. Visão Geral

Sistema RAG (Retrieval-Augmented Generation) sobre legislação de gestão condominial brasileira. O objetivo é responder perguntas de síndicos, moradores e administradoras com base exclusivamente no corpus oficial, com pipeline defensável em entrevista técnica — cada componente implementado à mão, cada decisão justificada com número.

**Não é:** um chatbot bonito. É um sistema que você defende com métrica e história de debug.

---

## 2. Domínio e Corpus

### 2.1 Documentos

| Documento | Fonte | Páginas estimadas |
|---|---|---|
| Lei 4.591/64 (Lei de Condomínios) | planalto.gov.br | ~40 |
| Código Civil arts. 1.314–1.358 | planalto.gov.br | ~40 |

**Total:** ~80 páginas. Corpus fixo — não há atualização automática.

### 2.2 Justificativa do domínio

- Conteúdo estático: lei federal muda raramente
- Perguntas têm resposta verificável por leitura direta do artigo
- Profissionais pagam por exatidão nesse conteúdo
- Corpus sem PII por definição (lei pública)

---

## 3. Golden Set

### 3.1 Processo de criação

1. LLM gera 50 perguntas candidatas a partir do corpus
2. Seleção manual de 25 + escrita das respostas esperadas citando artigo-fonte
3. Validação por especialista de domínio (profissional da área condominial)

**Classificação:** Silver — LLM gera candidatos, humano valida contra texto da lei. Validação cobre coerência textual com o corpus, **não** interpretação jurídica técnica. Ambiguidade jurídica não está coberta pelo golden set.

### 3.2 Categorias (25 exemplos)

| Categoria | Qtd | O que testa | Expected CRAG verdict |
|---|---|---|---|
| Simples | 6 | Retrieval direto, 1 chunk | `correct` |
| Multi-chunk | 5 | Síntese entre múltiplos artigos | `correct` ou `ambiguous` |
| Vocabulário diferente | 4 | Gap de registro — testa HyDE | `correct` se HyDE fechar; `ambiguous` se falhar |
| Ambiguous forçado | 4 | Relevância parcial, nenhum doc on-point | `ambiguous` |
| Sem resposta | 4 | Fato ausente do corpus | `incorrect` |
| Edge case adversarial | 2 | Manipulação/prompt injection via query | Sistema não deve alterar comportamento |

### 3.3 Schema de cada entrada

```json
{
  "id": "GS-001",
  "categoria": "simples",
  "pergunta": "Qual o prazo para convocar a assembleia ordinária?",
  "resposta_esperada": "A assembleia ordinária deve ser convocada pelo síndico...",
  "artigo_fonte": "Art. 1.350 CC",
  "expected_crag_verdict": "correct",
  "reference_contexts": ["chunk_id_1", "chunk_id_2"],
  "notas": ""
}
```

`reference_contexts` é obrigatório para `context_recall` no RAGAS. Para categorias `sem_resposta` e `adversarial`, o campo fica vazio — não há chunks fonte.

### 3.4 Limitação estatística

25 exemplos distribuídos em 6 categorias resultam em ~4 exemplos por categoria. Deltas medidos no DC-01 são **direcionais**, não estatisticamente robustos. Resultados indicam qual configuração é provavelmente melhor — não provam conclusivamente.

---

## 4. Pipeline de Retrieval

### 4.1 Chunking

**Estratégia:** Recursive character text splitter  
**Chunk size:** 512 tokens  
**Overlap:** 15% (~77 tokens)

**Decisão medida:** Antes de implementar, medir distribuição de tamanho dos artigos da Lei 4.591/64 e do CC. Mostrar que X% dos artigos ficam ≤ 512 tokens. Se distribuição não justificar 512, ajustar e documentar o ajuste. Número não pode ser chutado na spec final.

### 4.2 Embeddings

**Modelo:** `text-embedding-3-small` (OpenAI)  
**Dimensões:** 1536  

### 4.3 Vector Database

**Serviço:** Qdrant Cloud (free tier gerenciado)  
**Região:** `us-east-1`  
**Coleções:**

| Coleção | Conteúdo |
|---|---|
| `condominio_docs` | Chunks do corpus com embeddings |
| `query_cache` | Embeddings de queries anteriores + resposta cacheada |

### 4.4 BM25

**Biblioteca:** `bm25s`  
**Onde roda:** In-memory na API FastAPI  
**Sincronização:** `POST /ingest` reconstrói BM25 in-memory e Qdrant simultaneamente. Não há sincronização automática entre os dois índices. Corpus fixo torna reindexação evento raro.

### 4.5 Hybrid Search — RRF

**Estratégia:** Reciprocal Rank Fusion (RRF) implementado manualmente — não via Qdrant nativo.

```python
def rrf_fusion(dense_results, sparse_results, k=60):
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Candidatos por fonte:** top-50 dense + top-50 sparse → top-20 pós-RRF

**Justificativa do RRF:** Sem hiperparâmetro para tunar, robusto out-of-the-box, citado em papers de hybrid search como baseline sólido.

---

## 5. Reranking

### 5.1 Modelo (Baseline — DC-01)

**Modelo inicial:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Onde roda:** Local via `sentence-transformers`  
**Idioma:** Inglês — risco documentado para corpus em português.

### 5.2 Debug Case DC-01 (planejado)

> **Problema:** `ms-marco-MiniLM-L-6-v2` treinado em inglês, corpus em português.  
> **Hipótese:** Reranker reordena mal chunks em português, deprimindo `context_precision`.  
> **Ação:** Medir `context_precision` baseline → trocar para `BAAI/bge-reranker-v2-m3` (multilingual) → medir novamente.  
> **Critério de sucesso:** Delta positivo em `context_precision` com ressalva de amostra pequena.  
> **Nota:** Delta é direcional dado ~4 exemplos por categoria — não conclusivo estatisticamente.

### 5.3 Fluxo completo

```
Query → BM25 top-50 + Dense top-50 → RRF → top-20 → Cross-encoder scoring
```

Os scores do cross-encoder alimentam o CRAG (seção 7). Não há descarte por threshold antes do CRAG — o CRAG é quem decide o que fazer com os scores.

---

## 6. Query Transformations

**Estratégia:** Pipeline próprio, sem LangChain.

### 6.1 HyDE (Hypothetical Document Embedding)

Gera um documento hipotético que responderia a query. Usa o embedding desse documento para busca densa em vez do embedding da query original. Resolve queries curtas e vagas onde o gap entre linguagem do usuário e linguagem do corpus é grande.

### 6.2 Multi-Query (3 variações)

Gera 3 reformulações da query original com vocabulário diferente. Faz retrieval com cada uma. RRF funde as listas resultantes. Resolve o problema de vocabulário diferente do documento (categoria GS de golden set).

**Ordem de execução:** HyDE roda em paralelo com Multi-query. Os resultados são fundidos por RRF antes do reranking.

---

## 7. CRAG (Corrective RAG)

### 7.1 Evaluator

**Modelo:** Cross-encoder reutilizado (mesmo modelo do reranking).  
O score que o reranker já atribui a cada par `(query, chunk)` é o score de relevância do CRAG. Zero modelo extra.

### 7.2 Thresholds

| Veredito | Condição |
|---|---|
| `correct` | Melhor score do top-20 > 0.75 |
| `ambiguous` | Melhor score entre 0.70 e 0.75 |
| `incorrect` | Nenhum score ≥ 0.70 no top-20 |

Nenhum documento é descartado antes do CRAG. O CRAG aplica os thresholds sobre o conjunto completo pós-reranking: melhor score > 0.75 → `correct`; melhor score entre 0.70–0.75 → `ambiguous`; nenhum score ≥ 0.70 → `incorrect`.

### 7.3 Ações por veredito

**Correct:**  
Aplica decompose-then-recompose — quebra cada documento em knowledge strips menores, avalia relevância de cada strip individualmente, descarta ruído, recompõe só as strips relevantes. Não usa o documento como veio.

**Incorrect:**  
Descarta todos os documentos recuperados. Reescreve a query para formato de busca web. Executa web search restrita a `ALLOWED_DOMAINS`.

**Ambiguous:**  
Executa decompose-then-recompose (mesmo fluxo do Correct) + web search em paralelo com timeout. Funde as duas fontes antes de gerar.

**Comportamento de timeout no Ambiguous:**  
Se web search exceder timeout → degrada graciosamente para resposta baseada só no refino interno. Sinaliza `web_search_timeout: true` no evento `metadata` do SSE. **Não retorna erro para o usuário.**

### 7.4 Web Search — Allowlist

```python
ALLOWED_DOMAINS = [
    "planalto.gov.br",
    "jusbrasil.com.br",  # somente jurisprudência oficial
    "stj.jus.br",
    "tjrj.jus.br",       # exemplo de tribunal estadual
]
```

Não é busca aberta. Query é reescrita para formato de busca antes de executar.

---

## 8. Semantic Cache

**Implementação:** Coleção `query_cache` no Qdrant (não Redis, não GPTCache).  
**Threshold:** 0.95 (conservador dado domínio jurídico — falso positivo em lei tem consequência).

**Fluxo:**
```
Query entra → embed → busca query_cache → score > 0.95 → retorna cached com "cached": true
                                        → score ≤ 0.95 → pipeline completo → salva em query_cache
```

**Validation loop:** Antes de ativar em produção, rodar conjunto de pares "pergunta parecida, resposta diferente" (ex: "alugar vaga" vs "vender vaga") e medir taxa de falso positivo. Cache só ativa após validação documentada.

**Conformidade LGPD:**  
A coleção `query_cache` persiste embedding + texto das consultas de usuário em `us-east-1` (Qdrant Cloud). Diferente do corpus (fonte pública oficial, zero PII), a query do usuário pode conter dado pessoal incidental — exemplo: query que menciona unidade, vizinho ou situação específica. Isso constitui transferência internacional de dado pessoal sob LGPD (art. 33). **Aceita para fins de portfólio com ciência explícita da limitação.** Produção real exigiria base legal explícita ou hospedagem em região BR quando disponível.

---

## 9. Model Routing

### 9.1 Critério de classificação

Heurística no texto da query — zero custo, executada antes do pipeline:

```python
COMPLEX_KEYWORDS = ["compare", "diferença entre", "todos os casos", 
                    "quais são todas", "explique detalhadamente"]

def classify_query(query: str) -> Literal["simple", "complex"]:
    tokens = query.lower().split()
    if len(tokens) < 8:
        return "simple"
    if any(kw in query.lower() for kw in COMPLEX_KEYWORDS):
        return "complex"
    return "simple"
```

**Evolução planejada:** cheap-first + escalate — processa com modelo barato, escala se resposta não satisfizer critério de qualidade.

### 9.2 Modelos

| Tier | Provedor | Modelo | Quando usa |
|---|---|---|---|
| Simples | Groq | `llama-3.3-70b-versatile` | Query simples por heurística |
| Frontier | OpenAI | Modelo de raciocínio atual — verificar doc oficial no momento de implementar | Query complexa, CRAG Ambiguous, CRAG Incorrect |

**Nota:** Groq usa SDK compatível com OpenAI. Migração = trocar `base_url`. Zero refactor no pipeline.

---

## 10. Configuração de Geração

```python
temperature = 0          # RAG não quer criatividade
response_format = "json" # Structured output em todos os endpoints
```

**Prompt caching:** Feature automática da OpenAI com prefixo de system prompt estável. Não requer implementação adicional.

---

## 11. Segurança

### 11.1 Prompt Injection Detection (v1)

Validação executada na query antes de entrar no pipeline:

```python
INJECTION_PATTERNS = [
    r"ignore (as )?instru[çc][oõ]es anteriores",
    r"voc[eê] agora [eé]",
    r"system prompt",
    r"jailbreak",
]

def detect_injection(query: str) -> bool:
    if len(query) > 2000:
        return True
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False
```

Query bloqueada retorna HTTP 400 com mensagem genérica — não revela o padrão detectado.

### 11.2 Roadmap de Segurança (fora do escopo v1)

- **Data poisoning:** Validação de hash do corpus na ingestão + verificação de source URL
- **PII leakage:** Scan de resposta com `presidio` antes de retornar
- **Rate limiting:** Por IP no FastAPI middleware

---

## 12. API FastAPI

### 12.1 Endpoints

| Método | Path | Descrição |
|---|---|---|
| `POST` | `/query` | Recebe pergunta, retorna resposta + metadados |
| `POST` | `/ingest` | Recebe documento(s), executa chunking + indexação |
| `GET` | `/health` | Status de Qdrant, modelos carregados |
| `GET` | `/metrics` | Últimas N queries com métricas |
| `POST` | `/evaluate` | Roda RAGAS sobre golden set completo |

### 12.2 Schema `/query` response (sem streaming)

```json
{
  "answer": "string",
  "crag_verdict": "correct | ambiguous | incorrect",
  "sources": [
    {
      "chunk": "...",
      "score": 0.82,
      "artigo": "Art. 22 Lei 4.591/64"
    }
  ],
  "model_used": "llama-3.3-70b-versatile | gpt-4o-...",
  "query_transformed": "string (HyDE ou multi-query que gerou melhor resultado)",
  "cached": false,
  "web_search_timeout": false,
  "latency_ms": 340
}
```

### 12.3 Streaming — `/query?stream=true`

**Protocolo:** Server-Sent Events (SSE) via `StreamingResponse` do FastAPI.  
**Importante:** O endpoint retorna múltiplos **tipos de evento nomeado**, não streaming de texto puro. Cliente que assumir "só tokens" vai quebrar na leitura do `sources`.

**Eventos em ordem de emissão:**

```
event: status
data: {"message": "recuperando documentos..."}

event: status
data: {"message": "buscando fontes externas..."}   ← só no caminho Ambiguous

event: token
data: {"content": "O síndico deve"}

event: token
data: {"content": " convocar assembleia"}

event: metadata
data: {
  "crag_verdict": "correct",
  "model_used": "llama-3.3-70b-versatile",
  "cached": false,
  "web_search_timeout": false
}

event: sources
data: {"chunks": [{"text": "...", "score": 0.87, "artigo": "Art. 1.350 CC"}]}

event: error
data: {"code": "provider_timeout", "message": "..."}   ← só em falha

event: done
data: {"latency_ms": 1240}
```

---

## 13. Avaliação — RAGAS

### 13.1 Métricas

| Métrica | O que mede | Precisa de |
|---|---|---|
| `context_precision` | Dos chunks recuperados, quantos eram relevantes? | `reference_contexts` |
| `context_recall` | Dos chunks relevantes, quantos foram recuperados? | `reference_contexts` |
| `faithfulness` | Resposta é suportada pelos chunks? (detecta alucinação) | resposta + chunks |
| `answer_relevancy` | Resposta responde a pergunta? | resposta + pergunta |

### 13.2 Dashboard mínimo

Tabela por categoria do golden set × 4 métricas. Exemplo:

| Categoria | context_precision | context_recall | faithfulness | answer_relevancy |
|---|---|---|---|---|
| Simples | 0.91 | 0.88 | 0.95 | 0.93 |
| Multi-chunk | 0.74 | 0.71 | 0.89 | 0.87 |
| Vocabulário diferente | 0.68 | 0.65 | 0.91 | 0.84 |
| Ambiguous forçado | — | — | 0.82 | 0.79 |
| Sem resposta | — | — | — | — |
| Adversarial | — | — | 0.95 | — |

**Valores ilustrativos — populados com resultados reais após primeira rodada do `/evaluate`.** Células `—` onde a métrica não se aplica à categoria.

---

## 14. Infraestrutura

### 14.1 Stack

| Componente | Serviço | Custo |
|---|---|---|
| Vector DB | Qdrant Cloud free tier (`us-east-1`) | $0 permanente |
| API FastAPI | AWS EC2 t2.micro (free tier 12 meses) | $0 → ~$8/mês |
| Embeddings + Geração | OpenAI + Groq | Marginal (portfólio) |

### 14.2 Docker Compose

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - QDRANT_URL
      - OPENAI_API_KEY
      - GROQ_API_KEY
      - SEMANTIC_CACHE_THRESHOLD=0.95
      - CRAG_CORRECT_THRESHOLD=0.75
      - CRAG_INCORRECT_THRESHOLD=0.70
      - ALLOWED_DOMAINS=planalto.gov.br,jusbrasil.com.br,stj.jus.br
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Qdrant roda no cloud — não entra no compose.

---

## 15. CI/CD — GitHub Actions

### 15.1 A cada Pull Request (rápido, barato)

```
lint (ruff) + type check (mypy)
→ testes unitários (chunking, RRF, prompt injection detection)
→ build Docker image (sem publicar)
→ golden set subset (30-50 casos representativos) com RAGAS
→ gate CRAG: 1 caso por veredito — falha se veredito mudar de categoria
→ teste falso positivo semantic cache (pares "parecidos, resposta diferente")
→ pip-audit (scan de CVE nas dependências)
```

### 15.2 No merge para main (completo)

```
golden set completo (25 casos)
→ RAGAS com threshold mínimo por métrica — bloqueia se regredir
→ push Docker image para registry com tag de versão
```

**Não há deploy automático para produção.** Build + push provam o pipeline; deploy automático de sistema com dado de usuário não é necessário para demonstrar competência.

### 15.3 Thresholds de regressão (exemplo)

```yaml
min_context_precision: 0.70
min_faithfulness: 0.85
min_answer_relevancy: 0.80
```

Pipeline bloqueia merge se qualquer métrica cair abaixo.

---

## 16. Linguagem e Dependências

```
Python 3.11+

# Core
fastapi
uvicorn
pydantic

# Retrieval
bm25s
sentence-transformers
openai
groq

# Vector DB
qdrant-client

# Evaluation
ragas
datasets

# Security
re (stdlib)

# CI
ruff
mypy
pip-audit
pytest
```

---

## 17. Debug Cases Documentados

### DC-01 — Reranker em idioma errado (planejado)

**Problema:** `ms-marco-MiniLM-L-6-v2` treinado exclusivamente em inglês. Corpus em português.  
**Sintoma esperado:** `context_precision` abaixo do esperado para categorias simples e multi-chunk.  
**Diagnóstico:** Comparar ranking do ms-marco vs ranking esperado pelo golden set.  
**Correção:** Trocar para `BAAI/bge-reranker-v2-m3` (multilingual).  
**Métrica antes:** [a medir]  
**Métrica depois:** [a medir]  
**Ressalva:** Delta medido em ~4 exemplos por categoria — direcional, não estatisticamente conclusivo.

---

## 18. Limitações Documentadas

| Limitação | Impacto | Decisão |
|---|---|---|
| Golden set Silver (validação sem perito jurídico formal) | Interpretações ambíguas podem estar erradas | Aceita para portfólio |
| Qdrant Cloud em `us-east-1` | Queries do usuário saem do país (LGPD art. 33) | Aceita para portfólio; produção exige base legal |
| Amostra pequena do golden set (~4 por categoria) | Métricas são indicativas, não conclusivas | Documentado nos relatórios |
| Web search com allowlist, não vetada | Fontes pré-aprovadas podem ter informação desatualizada | Limitação documentada na resposta |
| BM25 in-memory reconstruído a cada restart | Dois índices desincronizados se ingest falhar parcialmente | Reindexação manual sem verificação automática de que os dois terminaram com sucesso — risco aceito dado corpus fixo e baixa frequência de mudança |

---

## 19. Princípios de Engenharia de Software

### 19.1 SOLID

Aplicados ao pipeline RAG — cada princípio com exemplo concreto do projeto:

**S — Single Responsibility**
Cada módulo tem uma única razão para mudar. `Retriever` só recupera documentos. `Reranker` só reordena. `CRAGEvaluator` só decide o veredito. `Generator` só chama o LLM. Nenhum desses módulos sabe que o outro existe — comunicam-se via interface.

**O — Open/Closed**
Aberto para extensão, fechado para modificação. Adicionar um novo query transformer (ex: step-back prompting) não modifica o `QueryTransformer` existente — implementa a interface e pluga no pipeline. Trocar o reranker de ms-marco para bge não muda o `Retriever`.

**L — Liskov Substitution**
`MsMarcoReranker` e `BgeReranker` são intercambiáveis via `BaseReranker`. O pipeline não sabe qual está rodando — só chama `.rerank(query, docs)` e recebe lista ordenada. Isso é o que permite o DC-01 (troca de reranker sem refatorar o pipeline).

**I — Interface Segregation**
`BaseRetriever` não força o retriever denso a implementar `.bm25_score()`. `BaseLLMProvider` não força o Groq a implementar métodos de embedding. Interfaces estreitas por responsabilidade, não uma interface gorda que toda implementação carrega pela metade.

**D — Dependency Inversion**
O `RAGPipeline` de alto nível depende de abstrações (`BaseRetriever`, `BaseReranker`, `BaseLLMProvider`), não de implementações concretas. Injeção de dependência no construtor — testável com mocks, trocável sem refatorar o pipeline.

```python
class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        crag: BaseCRAGEvaluator,
        generator: BaseGenerator,
        cache: BaseSemanticCache,
    ): ...
```

### 19.2 Clean Code

| Regra | Aplicação concreta |
|---|---|
| Nomes que revelam intenção | `reranked_docs`, não `r1`; `crag_verdict`, não `v` |
| Funções ≤ 30 linhas | Funções longas extraídas em helpers com nome descritivo |
| ≤ 2 níveis de indentação | Early return em vez de `if` aninhado |
| Sem números mágicos | `CRAG_CORRECT_THRESHOLD = 0.75`, não `0.75` hardcoded |
| Sem erros silenciosos | Exceções explícitas com contexto — nunca `except: pass` |
| Booleanos com prefixo | `is_cached`, `has_sources`, `web_search_timed_out` |
| Funções puras onde possível | Efeitos colaterais explícitos e isolados em camada própria |

### 19.3 Design Patterns

Dois padrões com justificativa concreta no problema — não encaixados para demonstrar conhecimento.

**Strategy — troca de implementação em runtime**
O DC-01 é literalmente troca de estratégia de reranker (`MsMarcoReranker` → `BgeReranker`) sem modificar o pipeline. Model routing é o mesmo mecanismo: `BaseLLMProvider` com duas implementações concretas selecionadas pela heurística. O padrão emerge da necessidade real de intercambialidade — não foi escolhido primeiro.

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...

# Model routing — seleção por dicionário, sem Factory formal
PROVIDERS: dict[str, BaseLLMProvider] = {
    "simple": GroqProvider(model="llama-3.3-70b-versatile"),
    "complex": OpenAIProvider(model=os.getenv("OPENAI_FRONTIER_MODEL")),
}
```

**Template Method — vereditos CRAG**
Os vereditos `correct` e `ambiguous` compartilham o mesmo esqueleto de decompose-then-recompose (avaliar strips → descartar ruído → recompor). A diferença é só o que acontece depois: `correct` para aí, `ambiguous` funde com busca web. Template Method evita duplicar a lógica de refino em dois lugares.

```python
class BaseCRAGEvaluator(ABC):
    def evaluate(self, query: str, docs: list[Doc]) -> str:
        verdict = self._decide_verdict(docs)
        refined = self._decompose_recompose(docs) if verdict != "incorrect" else []
        return self._handle(verdict, query, refined)

    @abstractmethod
    def _handle(self, verdict: str, query: str, refined: list[Doc]) -> str: ...
```

**Padrões descartados e por quê:**
- **Factory** para 2 provedores de LLM → dicionário resolve sem classe abstrata e subclasses
- **Chain of Responsibility** para o pipeline → fluxo sequencial fixo, não cadeia de handlers intercambiáveis
- **Observer** para métricas → o pipeline emite SSE via async generator, não múltiplos observers reagindo a subject

---

## 20. Roadmap (fora do escopo v1)

- Golden set Silver → Gold com perito jurídico validando interpretação
- Tiers adicionais do golden set (vocabulário ainda mais distante, perguntas compostas)
- Model routing: cheap-first + escalate substituindo heurística
- `pip-audit` → Dependabot automático
- Deploy automático com aprovação manual em produção
- Data poisoning: hash do corpus na ingestão
- PII leakage: scan com `presidio` antes de retornar resposta
- Região BR no Qdrant quando disponível (resolver LGPD em produção)
- LangGraph para evoluir CRAG para agente com memória de sessão
