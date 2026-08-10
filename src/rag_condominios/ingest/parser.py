"""Parse Brazilian legislation HTML into structured article dicts."""

import re

from bs4 import BeautifulSoup, NavigableString, Tag

ARTICLE_PATTERN = re.compile(
    r"^Art\.?\s*(\d+[\.\-]?\d*)[ºo°]?",
    re.IGNORECASE,
)

CC_CONDOMINIO_FIRST_ARTICLE = 1314
CC_CONDOMINIO_LAST_ARTICLE = 1358


def _normalize_article_number(raw: str) -> int | None:
    """Extract integer article number from strings like '1.314', '22', '1º'."""
    cleaned = raw.replace(".", "").replace("-", "").replace("º", "").replace("°", "").replace("o", "")
    try:
        return int(cleaned)
    except ValueError:
        return None


def _extract_section_from_predecessors(element: Tag) -> str:
    """Walk backwards through siblings and parents to find the nearest heading."""
    for sibling in element.previous_siblings:
        if not isinstance(sibling, Tag):
            continue
        if sibling.name in ("h1", "h2", "h3", "h4"):
            return sibling.get_text(separator=" ", strip=True)
    parent = element.parent
    if parent is None or parent.name == "[document]":
        return ""
    return _extract_section_from_predecessors(parent)


def _collect_article_text(article_tag: Tag) -> str:
    """Collect text from article tag and its following siblings until the next article."""
    parts = [article_tag.get_text(separator=" ", strip=True)]
    for sibling in article_tag.next_siblings:
        if isinstance(sibling, NavigableString):
            continue
        sibling_text = sibling.get_text(separator=" ", strip=True)
        if ARTICLE_PATTERN.match(sibling_text):
            break
        if sibling_text:
            parts.append(sibling_text)
    return " ".join(parts)


def _is_in_cc_range(article_number: int | None) -> bool:
    if article_number is None:
        return False
    return CC_CONDOMINIO_FIRST_ARTICLE <= article_number <= CC_CONDOMINIO_LAST_ARTICLE


def parse_articles(html: str, lei_name: str) -> list[dict[str, str]]:
    """Extract articles from legislation HTML.

    For 'codigo_civil', only articles 1.314–1.358 are returned.
    Each article is a dict with keys: lei, artigo, texto, secao.
    """
    soup = BeautifulSoup(html, "html.parser")
    is_cc = lei_name == "codigo_civil"
    articles: list[dict[str, str]] = []

    for tag in soup.find_all(["p", "div", "span", "td"]):
        raw_text = tag.get_text(separator=" ", strip=True)
        match = ARTICLE_PATTERN.match(raw_text)
        if not match:
            continue

        article_number = _normalize_article_number(match.group(1))
        if is_cc and not _is_in_cc_range(article_number):
            continue

        article_label = f"Art. {match.group(1)}"
        section = _extract_section_from_predecessors(tag)
        text = _collect_article_text(tag)

        articles.append({
            "lei": lei_name,
            "artigo": article_label,
            "texto": text,
            "secao": section,
        })

    return articles
