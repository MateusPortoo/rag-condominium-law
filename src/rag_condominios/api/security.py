"""Prompt injection detection for incoming queries (SPEC §11.1)."""

import re

MAX_QUERY_LENGTH = 2000

# Patterns straight from the spec — never reveal them in error messages.
_INJECTION_PATTERNS = [
    r"ignore (as )?instru[çc][oõ]es anteriores",
    r"voc[eê] agora [eé]",
    r"system prompt",
    r"jailbreak",
]


def detect_injection(query: str) -> bool:
    """Return True if the query looks like a prompt injection attempt."""
    if len(query) > MAX_QUERY_LENGTH:
        return True
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False
