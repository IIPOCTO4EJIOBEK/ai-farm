"""Подключение к хранилищу и загрузка настроек.

Пароль берётся из файла с правами 600, а не из переменных окружения и не из
аргументов командной строки: аргументы видны в `ps` всем пользователям машины.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_ENV_FILE = Path.home() / ".config" / "ai-workspace" / "env"

# Модель даёт вектора этой размерности; должно совпадать с vector(N) в схеме.
EMBEDDING_DIM = 1024


def load_env(path: Path | None = None) -> dict[str, str]:
    """Прочитать файл подключения.

    Значения из окружения имеют приоритет — так удобно разово переопределить
    хост, не трогая файл.
    """

    path = path or Path(os.environ.get("RAG_ENV_FILE", DEFAULT_ENV_FILE))
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    for key in ("RAG_DB_HOST", "RAG_DB_PORT", "RAG_DB_NAME", "RAG_DB_USER", "RAG_DB_PASSWORD"):
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def conninfo(env: dict[str, str] | None = None) -> str:
    env = env or load_env()
    missing = [
        k
        for k in ("RAG_DB_NAME", "RAG_DB_USER", "RAG_DB_PASSWORD")
        if not env.get(k)
    ]
    if missing:
        raise RuntimeError(
            f"нет настроек подключения: {', '.join(missing)}. "
            f"Запустите sql/setup.sh — он создаст файл {DEFAULT_ENV_FILE}"
        )
    return " ".join(
        [
            f"host={env.get('RAG_DB_HOST', '127.0.0.1')}",
            f"port={env.get('RAG_DB_PORT', '5432')}",
            f"dbname={env['RAG_DB_NAME']}",
            f"user={env['RAG_DB_USER']}",
            f"password={env['RAG_DB_PASSWORD']}",
            "connect_timeout=10",
        ]
    )


def connect(env: dict[str, str] | None = None):
    """Открыть соединение. Импорт psycopg внутри — чтобы модуль читался и без него."""

    try:
        import psycopg
    except ImportError as exc:  # noqa: TRY003
        raise RuntimeError(
            "не установлен psycopg. Выполните: .venv/bin/pip install 'psycopg[binary]'"
        ) from exc
    return psycopg.connect(conninfo(env))


def to_vector_literal(values) -> str:
    """Собрать вектор в текстовый литерал pgvector.

    Отдельный пакет pgvector для Python не нужен: psycopg отдаёт строку,
    а приведение к `vector` делает сервер в самом запросе.
    """

    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"


def main() -> int:
    env = load_env()
    if not env:
        print(f"файл {DEFAULT_ENV_FILE} не найден — хранилище ещё не настроено")
        return 1
    try:
        with connect(env) as conn, conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user, version();")
            db, user, version = cur.fetchone()
            cur.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'public';"
            )
            (tables,) = cur.fetchone()
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        print(f"не удалось подключиться: {exc}")
        return 1
    print(f"база:      {db}")
    print(f"пользователь: {user}")
    print(f"таблиц:    {tables}")
    print(f"pgvector:  {row[0] if row else 'не установлен'}")
    print(f"сервер:    {version.split(',')[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
