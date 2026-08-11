"""Recursive character text splitter — no LangChain dependency."""

import re

import tiktoken

from rag_condominios.core.config import TIKTOKEN_ENCODING
from rag_condominios.core.models import ArticleChunk

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 77

# Separators tried in order from coarsest to finest granularity.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def _make_chunk_id(lei: str, artigo: str, index: int) -> str:
    normalized_lei = re.sub(r"\s+", "_", lei.lower().strip())
    normalized_artigo = re.sub(r"[\s\.]+", "_", artigo.lower().strip())
    return f"{normalized_lei}_{normalized_artigo}_{index}"


def _split_on_separator(text: str, separator: str, chunk_size: int, overlap: int, encoder: tiktoken.Encoding) -> list[str]:
    """Split text by separator, then merge small pieces back up to chunk_size."""
    if separator:
        pieces = text.split(separator)
    else:
        # Character-level fallback: split into individual tokens
        tokens = encoder.encode(text)
        token_chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            token_chunks.append(encoder.decode(tokens[start:end]))
            start = end - overlap if end - overlap > start else end
        return token_chunks

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = _count_tokens(piece, encoder)
        if current_tokens + piece_tokens + (1 if current_parts else 0) > chunk_size:
            if current_parts:
                chunks.append(separator.join(current_parts))
            current_parts = [piece]
            current_tokens = piece_tokens
        else:
            current_parts.append(piece)
            current_tokens += piece_tokens + (1 if len(current_parts) > 1 else 0)

    if current_parts:
        chunks.append(separator.join(current_parts))
    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int, encoder: tiktoken.Encoding) -> list[str]:
    if _count_tokens(text, encoder) <= chunk_size:
        return [text]

    separator = separators[0]
    remaining_separators = separators[1:]

    raw_chunks = _split_on_separator(text, separator, chunk_size, overlap, encoder)

    result: list[str] = []
    for chunk in raw_chunks:
        if _count_tokens(chunk, encoder) <= chunk_size:
            result.append(chunk)
        elif remaining_separators:
            result.extend(_recursive_split(chunk, remaining_separators, chunk_size, overlap, encoder))
        else:
            result.append(chunk)
    return result


def _apply_overlap(chunks: list[str], overlap: int, encoder: tiktoken.Encoding) -> list[str]:
    """Prepend the tail of the previous chunk to maintain context continuity."""
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = encoder.encode(chunks[i - 1])
        overlap_tokens = prev_tokens[-overlap:] if len(prev_tokens) >= overlap else prev_tokens
        overlap_text = encoder.decode(overlap_tokens)
        result.append(overlap_text + " " + chunks[i])
    return result


def chunk_articles(
    articles: list[dict[str, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ArticleChunk]:
    """Split articles into token-bounded chunks with overlap."""
    encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    all_chunks: list[ArticleChunk] = []

    for article in articles:
        text = article["texto"]
        lei = article["lei"]
        artigo = article["artigo"]
        secao = article["secao"]

        raw_splits = _recursive_split(text, SEPARATORS, chunk_size, overlap, encoder)
        overlapping_splits = _apply_overlap(raw_splits, overlap, encoder)

        for index, chunk_text in enumerate(overlapping_splits):
            token_count = _count_tokens(chunk_text, encoder)
            all_chunks.append(ArticleChunk(
                id=_make_chunk_id(lei, artigo, index),
                text=chunk_text,
                token_count=token_count,
                lei=lei,
                artigo=artigo,
                secao=secao,
                chunk_index=index,
            ))

    return all_chunks
