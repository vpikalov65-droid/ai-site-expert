"""Загрузка HTML и извлечение текста страницы."""

from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


def _is_temporary(error: BaseException) -> bool:
    if isinstance(error, requests.exceptions.SSLError):
        return False
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(error, requests.HTTPError) and error.response is not None:
        status = error.response.status_code
        return status in {408, 429} or status >= 500
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_temporary),
    reraise=True,
)
def _download(url: str) -> str:
    with requests.get(url, timeout=15) as response:
        response.raise_for_status()
        if not response.encoding or response.encoding == "ISO-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"
        return response.text


def extract_text(url: str) -> str:
    """Загружает страницу HTTP(S); возвращает до 15 000 символов."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Укажите корректный URL с http:// или https://")

    soup = BeautifulSoup(_download(url), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    if not text:
        raise ValueError("На странице не найден текст для анализа")
    return text[:15000]


if __name__ == "__main__":
    print(extract_text("https://example.com"))
