"""CRAG refinement actions — SPEC section 7.3.

Three actions, each with a single responsibility:
  - decompose_recompose: filters sentence-level strips from ranked chunks.
  - web_search: fetches external snippets from allowed domains.
  - ambiguous_action: runs both in parallel and merges results.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import requests
from groq import APIError as GroqAPIError

from rag_condominios.core.config import CRAG_AMBIGUOUS_THRESHOLD

if TYPE_CHECKING:
    from groq import Groq

    from rag_condominios.core.protocols import BaseReranker
    from rag_condominios.retrieval.reranker import RankedResult

_log = logging.getLogger(__name__)

_WEB_SEARCH_TIMEOUT = 5
_DDGO_URL = "https://api.duckduckgo.com/"
_REWRITE_SYSTEM = (
    "Reescreva a pergunta em formato compacto para busca na web. "
    "Responda APENAS com a query de busca, sem explicação."
)


@dataclass
class _WebChunk:
    """Minimal chunk-like object built from a web search snippet."""

    text: str
    lei: str = "Web"
    artigo: str = ""


def decompose_recompose(
    query: str,
    ranked: list[RankedResult],
    reranker: BaseReranker,
) -> list[RankedResult]:
    """Filter each chunk to only the sentences relevant to the query.

    Splits chunk text into strips at sentence boundaries, scores each strip
    via the reranker, and rebuilds the chunk text from only relevant strips.
    Chunks where no strip survives are dropped entirely.
    """
    from rag_condominios.retrieval.pipeline import RetrievalResult as _RetrievalResult
    from rag_condominios.retrieval.reranker import RankedResult as _RankedResult

    refined: list[RankedResult] = []

    for chunk in ranked:
        strips = [s.strip() for s in chunk.text.split(". ") if s.strip()]
        if not strips:
            continue

        strip_results = [
            _RetrievalResult(
                chunk_id=f"{chunk.chunk_id}__strip_{i}",
                rrf_score=chunk.rrf_score,
                text=strip,
                lei=chunk.lei,
                artigo=chunk.artigo,
            )
            for i, strip in enumerate(strips)
        ]

        scored = reranker.rerank(query, strip_results)
        relevant = [r for r in scored if r.rerank_score > CRAG_AMBIGUOUS_THRESHOLD]

        if not relevant:
            continue

        recomposed = ". ".join(r.text for r in relevant)
        refined.append(
            _RankedResult(
                chunk_id=chunk.chunk_id,
                rrf_score=chunk.rrf_score,
                rerank_score=chunk.rerank_score,
                text=recomposed,
                lei=chunk.lei,
                artigo=chunk.artigo,
            )
        )

    return refined or ranked


def _rewrite_for_web(query: str, groq_client: Groq) -> str:
    """Rewrite the query as a compact web-search string."""
    from rag_condominios.retrieval.generator import GROQ_MODEL

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content if response.choices else None
    if not content:
        _log.warning("_rewrite_for_web: Groq returned empty content, using original query")
        return query
    return content


def _ddgo_snippets(search_query: str, domain: str) -> list[str] | None:
    """Fetch DuckDuckGo snippets for one domain.

    Returns None on network/parse failure (signals timed_out to caller),
    or a list (possibly empty) on a successful response.
    """
    params = {
        "q": f"site:{domain} {search_query}",
        "format": "json",
        "no_html": "1",
    }
    try:
        resp = requests.get(_DDGO_URL, params=params, timeout=_WEB_SEARCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        _log.warning("DuckDuckGo fetch failed for domain %s: %s", domain, exc)
        return None

    snippets: list[str] = []
    abstract = data.get("AbstractText", "")
    if abstract:
        snippets.append(abstract)
    for item in data.get("RelatedTopics", []):
        text = item.get("Text", "")
        if text:
            snippets.append(text)
    return snippets


def web_search(
    query: str,
    groq_client: Groq,
    allowed_domains: list[str],
) -> tuple[list[_WebChunk], bool]:
    """Search allowed domains and return text snippets.

    Returns (chunks, timed_out).
    timed_out=True only on network/API failure, not on empty results.
    """
    try:
        search_query = _rewrite_for_web(query, groq_client)
    except GroqAPIError as exc:
        _log.warning("Groq query rewrite failed, falling back to original query: %s", exc)
        search_query = query

    all_snippets: list[str] = []
    any_network_failure = False
    for domain in allowed_domains:
        result = _ddgo_snippets(search_query, domain)
        if result is None:
            any_network_failure = True
        else:
            all_snippets.extend(result)

    if not all_snippets:
        if any_network_failure:
            return [], True
        _log.info("web search returned no snippets for query=%r", search_query)
        return [], False

    chunks = [_WebChunk(text=s) for s in all_snippets]
    return chunks, False


def ambiguous_action(
    query: str,
    ranked: list[RankedResult],
    reranker: BaseReranker,
    groq_client: Groq,
    allowed_domains: list[str],
) -> tuple[list[RankedResult | _WebChunk], bool]:
    """Run decompose-recompose and web search in parallel for ambiguous verdict.

    Returns (merged_chunks, web_search_timed_out).
    If web search times out, only the recomposed results are returned.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_recompose = executor.submit(decompose_recompose, query, ranked, reranker)
        future_web = executor.submit(web_search, query, groq_client, allowed_domains)

        try:
            recomposed = future_recompose.result()
        except Exception as exc:  # noqa: BLE001 — reranker failure falls back to original ranked
            _log.error("decompose_recompose failed in ambiguous_action: %s", exc)
            recomposed = list(ranked)

        try:
            web_chunks, timed_out = future_web.result()
        except Exception as exc:  # noqa: BLE001 — web search future failure treated as timeout
            _log.error("web_search future failed in ambiguous_action: %s", exc)
            web_chunks, timed_out = [], True

    merged: list[RankedResult | _WebChunk] = list(recomposed)
    if not timed_out:
        merged.extend(web_chunks)

    return merged, timed_out
