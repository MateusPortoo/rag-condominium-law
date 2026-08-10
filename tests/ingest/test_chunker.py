"""Unit tests for ingest/chunker.py."""

import re

import tiktoken

from rag_condominios.ingest.chunker import TIKTOKEN_ENCODING, chunk_articles

ENCODER = tiktoken.get_encoding(TIKTOKEN_ENCODING)

SHORT_WORD = "condômino "  # ~3 tokens


def _make_article(text: str, lei: str = "lei_4591", artigo: str = "Art. 1") -> dict[str, str]:
    return {"lei": lei, "artigo": artigo, "texto": text, "secao": "Capítulo I"}


def _article_with_tokens(token_count: int) -> dict[str, str]:
    """Generate an article whose text is approximately token_count tokens long."""
    tokens_per_word = len(ENCODER.encode(SHORT_WORD))
    repetitions = max(1, token_count // tokens_per_word)
    text = SHORT_WORD * repetitions
    return _make_article(text)


def test_short_article_produces_single_chunk() -> None:
    article = _article_with_tokens(50)
    chunks = chunk_articles([article], chunk_size=512, overlap=77)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_long_article_produces_multiple_chunks() -> None:
    article = _article_with_tokens(1200)
    chunks = chunk_articles([article], chunk_size=512, overlap=77)
    assert len(chunks) > 1


def test_chunk_id_follows_naming_pattern() -> None:
    article = _make_article("Texto de exemplo.", lei="lei_4591", artigo="Art. 22")
    chunks = chunk_articles([article], chunk_size=512, overlap=77)
    for chunk in chunks:
        # Pattern: {lei}_{artigo}_{index}, normalized (no spaces, dots become underscores)
        assert re.match(r"^[a-z0-9_]+_art__?22_\d+$", chunk.id), (
            f"Chunk ID '{chunk.id}' does not match expected pattern"
        )


def test_metadata_preserved_on_all_chunks() -> None:
    article = _article_with_tokens(1500)
    article["secao"] = "Seção Especial"
    chunks = chunk_articles([article], chunk_size=512, overlap=77)
    for chunk in chunks:
        assert chunk.lei == article["lei"]
        assert chunk.artigo == article["artigo"]
        assert chunk.secao == article["secao"]


def test_overlap_shares_content_between_consecutive_chunks() -> None:
    article = _article_with_tokens(1200)
    chunks = chunk_articles([article], chunk_size=512, overlap=77)
    assert len(chunks) >= 2

    # The beginning of chunk N+1 should share tokens with the end of chunk N.
    chunk_0_tokens = ENCODER.encode(chunks[0].text)
    chunk_1_tokens = ENCODER.encode(chunks[1].text)

    # Take the last 'overlap' tokens of chunk 0 and first 'overlap' of chunk 1.
    tail_tokens = set(chunk_0_tokens[-77:])
    head_tokens = set(chunk_1_tokens[:77])
    shared = tail_tokens & head_tokens
    assert len(shared) > 0, (
        "Expected overlap tokens to be shared between consecutive chunks, but found none."
    )


def test_token_count_field_matches_actual_encoding() -> None:
    article = _make_article("O síndico deve convocar a assembleia ordinária anualmente.")
    chunks = chunk_articles([article], chunk_size=512, overlap=77)
    for chunk in chunks:
        actual_tokens = len(ENCODER.encode(chunk.text))
        assert chunk.token_count == actual_tokens, (
            f"token_count field {chunk.token_count} != actual {actual_tokens}"
        )


def test_multiple_articles_produce_sequential_indices_per_article() -> None:
    articles = [
        _make_article("Texto breve.", artigo="Art. 1"),
        _article_with_tokens(1200),
    ]
    articles[1]["artigo"] = "Art. 2"
    chunks = chunk_articles(articles, chunk_size=512, overlap=77)

    art1_chunks = [c for c in chunks if c.artigo == "Art. 1"]
    art2_chunks = [c for c in chunks if c.artigo == "Art. 2"]

    assert [c.chunk_index for c in art1_chunks] == list(range(len(art1_chunks)))
    assert [c.chunk_index for c in art2_chunks] == list(range(len(art2_chunks)))
