"""Start the FastAPI development server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "rag_condominios.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
