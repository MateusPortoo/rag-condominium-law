"""Model routing — classifies queries as simple or complex — SPEC section 9.

Single responsibility: decide which model tier to use based on query length
and complexity keywords. No I/O, no side effects.
"""

from __future__ import annotations

from typing import Literal

ModelTier = Literal["simple", "complex"]

_COMPLEX_KEYWORDS = [
    "compare",
    "diferença entre",
    "todos os casos",
    "quais são todas",
    "explique detalhadamente",
]

_SIMPLE_TOKEN_THRESHOLD = 8


def classify_query(query: str) -> ModelTier:
    """Return 'simple' or 'complex' based on query length and keywords.

    Rules (SPEC section 9):
      - Fewer than 8 tokens → simple
      - Contains any complex keyword → complex
      - Otherwise → simple
    """
    tokens = query.lower().split()
    if len(tokens) < _SIMPLE_TOKEN_THRESHOLD:
        return "simple"
    if any(kw in query.lower() for kw in _COMPLEX_KEYWORDS):
        return "complex"
    return "simple"
