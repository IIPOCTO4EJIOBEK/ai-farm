"""Проверка качества поиска на проиндексированном проекте.

Интеграционная: нужны работающая база и заполненный проект `rdp-mcp`.
Запуск: .venv/bin/python tests/test_search.py

Оценка идёт по топ-3, а не по топ-1. Это не поблажка: на вопрос «сколько
ждать появления окна» правильных ответов два — таблица диагностики в README
(«нет окна за 25 с») и код `session.py` с самим `WINDOW_TIMEOUT_SECONDS`.
Требовать единственный файл значило бы проверять не качество поиска, а
совпадение с моими ожиданиями.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.db import connect, load_env  # noqa: E402
from rag.embedder import Embedder  # noqa: E402
from rag.search import search  # noqa: E402

# (запрос, подходящие файлы). Достаточно попадания любого из них в топ-3.
CASES: list[tuple[str, set[str]]] = [
    ("сколько ждать появления окна", {"README.md", "session.py"}),
    ("почему клик попадал на несколько пикселей ниже нужного", {"README.md", "x11.py"}),
    ("как пароль передаётся, чтобы не был виден в списке процессов",
     {"README.md", "config.py", "session.py"}),
    ("проверка что буфер обмена работает", {"README.md", "x11_test.py", "x11.py"}),
    ("ctrl alt del на удалённой машине", {"README.md", "x11.py"}),
    ("запуск xfreerdp отсоединённым процессом", {"README.md", "session.py"}),
    ("тестовое окно на tkinter", {"selftest_app.py", "x11_test.py"}),
    ("какие инструменты предоставляет mcp сервер", {"README.md", "server.py"}),
    ("WINDOW_TIMEOUT_SECONDS", {"session.py"}),
    ("normalise_key", {"x11.py"}),
    ("почему xclip подвешивал выполнение", {"x11.py", "README.md"}),
    ("где хранится pid сессии", {"README.md", "session.py"}),
]

TOP_K = 3


def main() -> int:
    try:
        conn = connect(load_env())
    except Exception as exc:  # noqa: BLE001
        print(f"нет подключения к базе: {exc}")
        return 1

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunk")
        (total,) = cur.fetchone()
    if total == 0:
        print("хранилище пусто — сначала запустите: python -m rag.ingest")
        return 1

    embedder = Embedder()
    hits = 0
    for query, acceptable in CASES:
        # Сравниваем по имени файла, а не по полному пути: в хранилище лежит
        # `src/rdp_mcp/x11.py`, а в ожиданиях — `x11.py`.
        found = [Path(h.rel_path).name for h in search(conn, query, limit=TOP_K, embedder=embedder)]
        ok = any(name in acceptable for name in found)
        hits += ok
        mark = "ок     " if ok else "ПРОМАХ "
        print(f"  {mark} {query[:46]:48} -> {', '.join(found) or '—'}")

    print(f"\nв топ-{TOP_K}: {hits}/{len(CASES)} ({100 * hits // len(CASES)}%)")
    return 0 if hits >= len(CASES) * 3 // 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
