"""Simple generation: query + retrieved context → answer via Groq."""

from collections.abc import Iterator, Sequence
from typing import Protocol

from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_CONTEXT_CHUNKS = 5
SYSTEM_PROMPT = """Você é um assistente especializado em legislação condominial brasileira.
Responda APENAS com base nos trechos de lei fornecidos abaixo.
Se a resposta não puder ser encontrada nos trechos, diga isso claramente.
Não invente informações. Seja direto e cite o artigo quando possível."""


class _Chunk(Protocol):
    """Minimal interface required by build_context — satisfied by both
    RetrievalResult and RankedResult without explicit inheritance."""

    text: str
    lei: str
    artigo: str


def build_context(chunks: Sequence[_Chunk]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], start=1):
        header = f"[{i}] {chunk.lei} — {chunk.artigo}" if chunk.artigo else f"[{i}]"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)


def build_user_message(context: str, query: str) -> str:
    """Format the user turn: context blocks followed by the question."""
    return f"Trechos relevantes da legislação:\n\n{context}\n\nPergunta: {query}"


def generate(
    query: str,
    chunks: Sequence[_Chunk],
    groq_client: Groq,
) -> str:
    """Generate an answer given a query and retrieved context chunks."""
    context = build_context(chunks)
    user_message = build_user_message(context, query)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def generate_stream(
    query: str,
    chunks: Sequence[_Chunk],
    groq_client: Groq,
) -> Iterator[str]:
    """Stream token-by-token generation; yields non-empty content strings."""
    context = build_context(chunks)
    user_message = build_user_message(context, query)

    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        stream=True,
    )
    for delta in stream:
        content = delta.choices[0].delta.content or ""
        if content:
            yield content
