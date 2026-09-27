"""Веб-поиск для фермы: свой SearXNG на 127.0.0.1:8888.

Зачем это здесь. Бесплатные модели фермы интернета не видят: на вопросах о
свежих версиях, ценах и событиях они начинают выдумывать, а контролёр этого
не ловит — он проверяет связность, а не факты. Выдать исполнителю инструмент
нельзя: мелкие модели путаются в вызовах, а Zen-модели на инструментах ещё и
пишут файлы в домашний каталог.

Поэтому ищет сама ферма. Она делает запрос к SearXNG, сжимает выдачу в
короткую выжимку со ссылками и подкладывает её в промпт исполнителя. Для
модели это выглядит как обычный контекст задачи — никаких инструментов.

Адрес берётся из ``SEARXNG_URL``: на 81 поиск живёт в контейнере
``ai-searxng:8080``, на ноуте — службой на 8888.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

DEFAULT_URL = "http://127.0.0.1:8888/search"

# Сколько знаков выжимки отдавать модели. Бесплатные слоты фермы — модели с
# небольшим контекстом, и промпт делит его с самой задачей и инструкциями.
# Полторы тысячи знаков — это ~6 ссылок с короткими сниппетами: достаточно,
# чтобы опереться на факт, и мало, чтобы выдавить саму задачу.
DIGEST_LIMIT = 1500

# Один запрос ждёт не дольше этого. SearXNG отвечает за секунды, но при
# недоступном движке висит до собственного таймаута — ждать его бессмысленно.
TIMEOUT = 20

_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class Hit:
    title: str
    url: str
    snippet: str


def endpoint() -> str:
    return os.environ.get("SEARXNG_URL", DEFAULT_URL)


def search(query: str, *, limit: int = 6, timeout: int = TIMEOUT) -> list[Hit]:
    """Спросить у SearXNG. Пустой список — «не нашлось» или «не ответил»."""

    query = query.strip()
    if not query:
        return []
    url = endpoint() + "?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "language": "ru"}
    )
    request = urllib.request.Request(url, headers={"User-Agent": "ai-farm/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []

    hits: list[Hit] = []
    for item in data.get("results", [])[:limit]:
        title = " ".join(str(item.get("title", "")).split())
        link = str(item.get("url", "")).strip()
        snippet = " ".join(str(item.get("content", "")).split())
        if title and link:
            hits.append(Hit(title=title, url=link, snippet=snippet))
    return hits


def digest(query: str, *, limit: int = 6, max_chars: int = DIGEST_LIMIT) -> str:
    """Готовая для промпта выжимка. Пустая строка — поиск не удался.

    Результат запоминается: одна и та же подзадача проходит через черновик,
    замечания контролёра и переделку, и каждый раз ходить в сеть незачем.
    """

    key = f"{query.strip()}|{limit}|{max_chars}"
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    hits = search(query, limit=limit)
    if not hits:
        with _CACHE_LOCK:
            _CACHE[key] = ""
        return ""

    lines: list[str] = []
    used = 0
    for index, hit in enumerate(hits, 1):
        block = f"{index}. {hit.title}\n   {hit.url}"
        if hit.snippet:
            block += f"\n   {hit.snippet}"
        if used + len(block) > max_chars and lines:
            break
        lines.append(block)
        used += len(block)

    text = "\n".join(lines)
    with _CACHE_LOCK:
        _CACHE[key] = text
    return text


def context_block(query: str, *, limit: int = 6) -> str:
    """Выжимка, обёрнутая в тег для промпта. Пусто — если искать нечего."""

    text = digest(query, limit=limit)
    if not text:
        return ""
    return (
        "<свежие_данные_из_интернета>\n"
        f"Запрос: {query}\n{text}\n"
        "</свежие_данные_из_интернета>\n"
        "Это выдача поиска на сегодня. Опирайся на неё в фактах, датах и числах "
        "и ссылайся на источники.\n"
        "Три запрета, проверено на живом прогоне: не изображай вызовы "
        "инструментов и поиска — инструментов у тебя нет, пиши ответ своими "
        "словами; не подставляй похожее значение вместо номера версии, даты или "
        "цены; если нужного факта в выдаче нет — скажи прямо, что в найденных "
        "источниках его нет."
    )


def alive(timeout: int = 5) -> bool:
    """Отвечает ли поиск. Нужно, чтобы --web не ронял прогон молча."""

    try:
        with urllib.request.urlopen(endpoint() + "?q=test&format=json", timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
