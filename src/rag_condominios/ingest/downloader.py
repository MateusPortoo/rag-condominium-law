"""Download and cache the raw HTML corpus from official government sources."""

from pathlib import Path

import httpx

RAW_DIR = Path(__file__).parents[4] / "data" / "raw"

CORPUS_SOURCES: dict[str, str] = {
    "lei_4591": "https://www.planalto.gov.br/ccivil_03/leis/l4591.htm",
    "codigo_civil": "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
}

FILE_NAMES: dict[str, str] = {
    "lei_4591": "lei_4591_64.html",
    "codigo_civil": "codigo_civil.html",
}

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT_SECONDS = 30


def _load_cached(key: str) -> str | None:
    path = RAW_DIR / FILE_NAMES[key]
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def _fetch_and_save(key: str) -> str:
    url = CORPUS_SOURCES[key]
    with httpx.Client(headers=BROWSER_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    html = response.text
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / FILE_NAMES[key]).write_text(html, encoding="utf-8")
    return html


def download_corpus() -> dict[str, str]:
    """Return HTML for each corpus document, loading from disk if already cached."""
    corpus: dict[str, str] = {}
    for key, url in CORPUS_SOURCES.items():
        cached = _load_cached(key)
        if cached is not None:
            print(f"[downloader] Loaded {FILE_NAMES[key]} from disk.")
            corpus[key] = cached
        else:
            print(f"[downloader] Fetching {url} ...")
            corpus[key] = _fetch_and_save(key)
            print(f"[downloader] Saved {FILE_NAMES[key]}.")
    return corpus
