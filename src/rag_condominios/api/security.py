"""Prompt injection detection for incoming queries (SPEC §11.1)."""

import re
import unicodedata

MAX_QUERY_LENGTH = 2000

# Codepoints used to hide injection text: zero-width chars + bidi override/isolate marks.
_HIDDEN_CODEPOINTS: frozenset[int] = frozenset({
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x2066,  # LEFT-TO-RIGHT ISOLATE
    0x2067,  # RIGHT-TO-LEFT ISOLATE
    0x2068,  # FIRST STRONG ISOLATE
    0x2069,  # POP DIRECTIONAL ISOLATE
    0xFEFF,  # BOM / ZERO WIDTH NO-BREAK SPACE
})
_HIDDEN_TABLE: dict[int, None] = {cp: None for cp in _HIDDEN_CODEPOINTS}

# Patterns straight from the spec — never reveal them in error messages.
_INJECTION_PATTERNS = [
    r"ignore (as )?instru[çc][oõ]es anteriores",
    r"voc[eê] agora [eé]",
    r"system prompt",
    r"jailbreak",
]


def _normalize(text: str) -> str:
    """NFKC-normalize and strip zero-width / bidi-override characters."""
    return unicodedata.normalize("NFKC", text).translate(_HIDDEN_TABLE)


def detect_injection(query: str) -> bool:
    """Return True if the query looks like a prompt injection attempt."""
    if len(query) > MAX_QUERY_LENGTH:
        return True
    normalized = _normalize(query)
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True
    return False
