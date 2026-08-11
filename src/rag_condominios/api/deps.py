"""Shared dependency helpers for FastAPI routes."""

from fastapi import HTTPException
from groq import Groq
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_condominios.api.state import AppState


def extract_clients(state: AppState) -> tuple[OpenAI, Groq, QdrantClient]:
    """Return non-optional clients or raise HTTP 503.

    Replaces scattered `assert client is not None` calls that are silently
    removed by `python -O`. Using an explicit guard here ensures a proper
    503 response in all execution modes.
    """
    if (
        state.openai_client is None
        or state.groq_client is None
        or state.qdrant_client is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Serviço não inicializado. Verifique as variáveis de ambiente.",
        )
    return state.openai_client, state.groq_client, state.qdrant_client
