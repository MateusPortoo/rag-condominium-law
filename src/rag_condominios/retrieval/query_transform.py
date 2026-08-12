"""Query transformation strategies — SPEC section 6.

HyDE: generate a hypothetical answer document, embed it, search with that vector.
Multi-Query: generate 3 alternative phrasings of the query and retrieve for each.

Both run in parallel via ThreadPoolExecutor (Groq calls are synchronous).
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groq import Groq
    from openai import OpenAI


_HYDE_SYSTEM = (
    "Você é um especialista em legislação condominial brasileira. "
    "Escreva um parágrafo conciso como se fosse um trecho de lei que "
    "respondesse diretamente à pergunta fornecida. Responda APENAS com o trecho, "
    "sem introdução."
)

_MULTI_QUERY_SYSTEM = (
    "Você é um assistente que reformula perguntas jurídicas. "
    "Dado uma pergunta, gere EXATAMENTE 3 reformulações com vocabulário diferente "
    "que mantenham o mesmo significado. "
    "Retorne APENAS as 3 perguntas numeradas, uma por linha, sem introdução."
)

_MULTI_QUERY_COUNT = 3


def _hyde_text(query: str, groq_client: Groq) -> str:
    """Generate a hypothetical document that would answer the query."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or query


def _multi_query_texts(query: str, groq_client: Groq) -> list[str]:
    """Generate 3 alternative phrasings of the query."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _MULTI_QUERY_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
    )
    raw = response.choices[0].message.content or ""
    lines = [
        line.lstrip("123456789. ").strip()
        for line in raw.splitlines()
        if line.strip()
    ]
    return lines[:_MULTI_QUERY_COUNT] if lines else [query]


def generate_hyde_embedding(
    query: str,
    groq_client: Groq,
    openai_client: OpenAI,
) -> list[float]:
    """Return an embedding of a hypothetical document for the query.

    Embeds the HyDE document instead of the raw query so that the vector
    lives in the same semantic space as actual law passages.
    """
    hyp_doc = _hyde_text(query, groq_client)
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=hyp_doc,
    )
    return response.data[0].embedding


def generate_multi_queries(query: str, groq_client: Groq) -> list[str]:
    """Return up to 3 reformulations of the query."""
    return _multi_query_texts(query, groq_client)


def transform_parallel(
    query: str,
    groq_client: Groq,
    openai_client: OpenAI,
) -> tuple[list[float], list[str]]:
    """Run HyDE embedding and multi-query generation in parallel.

    Returns (hyde_embedding, alternative_queries).
    Falls back to an empty embedding / original query on failure.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_hyde = executor.submit(
            generate_hyde_embedding, query, groq_client, openai_client
        )
        future_multi = executor.submit(generate_multi_queries, query, groq_client)

        hyde_embedding: list[float] = future_hyde.result()
        alt_queries: list[str] = future_multi.result()

    return hyde_embedding, alt_queries
