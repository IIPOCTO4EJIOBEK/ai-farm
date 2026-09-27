"""MCP-сервер поиска по локальному хранилищу проектов.

Отдаёт модели те же данные, что лежат в PostgreSQL: фрагменты файлов с
указанием источника. Никакого доступа к произвольным файлам на диске здесь
нет — только то, что было проиндексировано.

Запускается по stdio: сервер общается с клиентом через stdin/stdout, поэтому
печатать что-либо в stdout нельзя — это сломает протокол. Отладочные
сообщения идут в stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # MCP SDK 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from rag.db import connect, load_env  # noqa: E402
from rag.search import DEFAULT_LIMIT, neighbours, search  # noqa: E402

mcp = MCPServer("knowledge")

MAX_CHARS_PER_HIT = 2000


def _format_hits(conn, hits, *, with_neighbours: bool = False) -> str:
    """Разложить находки в текст для модели.

    Соединение передаётся снаружи: соседние фрагменты добираются из той же
    базы, и открывать ради них второе подключение незачем.
    """

    if not hits:
        return "Ничего не найдено."

    parts: list[str] = []
    for hit in hits:
        header = f"[{hit.project}] {hit.rel_path} — фрагмент {hit.seq}"
        if hit.found_by != "neighbour":
            header += f"  (найдено: {hit.found_by}, оценка {hit.score:.4f})"
        body = hit.content
        if len(body) > MAX_CHARS_PER_HIT:
            body = body[:MAX_CHARS_PER_HIT] + "\n… (фрагмент обрезан при выводе)"
        parts.append(f"{header}\n{body}")

        if with_neighbours:
            for nb in neighbours(conn, hit.chunk_id, before=1, after=1):
                if nb.chunk_id != hit.chunk_id:
                    parts.append(
                        f"  ↳ соседний фрагмент {nb.seq} из {nb.rel_path}\n"
                        f"  {nb.content[:600]}"
                    )
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def search_knowledge(
    query: str,
    limit: int = DEFAULT_LIMIT,
    project: str | None = None,
    category: str | None = None,
    with_neighbours: bool = False,
) -> str:
    """Найти в хранилище фрагменты, отвечающие на вопрос.

    Поиск гибридный: векторная близость плюс полнотекстовый поиск по русскому
    словарю, результаты объединяются по позициям в списках.

    Args:
        query: вопрос или ключевые слова. Формулируйте содержательно — «как
            устроена аутентификация в rdp-mcp», а не «аутентификация rdp».
        limit: сколько фрагментов вернуть.
        project: ограничить поиск одним проектом (имя из list_projects).
        category: ограничить поиск категорией проекта.
        with_neighbours: добавить соседние фрагменты того же файла — полезно,
            если ответ обрывается на середине мысли.
    """

    if not query.strip():
        return "Пустой запрос."

    try:
        with connect(load_env()) as conn:
            hits = search(conn, query, limit=limit, project=project, category=category)
            if not hits:
                return "Ничего не найдено."
            return _format_hits(conn, hits, with_neighbours=with_neighbours)
    except Exception as exc:  # noqa: BLE001
        return f"Ошибка поиска: {type(exc).__name__}: {exc}"


@mcp.tool()
def read_document(project: str, rel_path: str) -> str:
    """Вернуть полный текст проиндексированного файла.

    Файл собирается из фрагментов по сохранённым смещениям, поэтому текст
    совпадает с исходным — включая места перекрытия, которые не дублируются.

    Args:
        project: имя проекта.
        rel_path: путь файла внутри проекта, как он показан в результатах поиска.
    """

    try:
        with connect(load_env()) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.seq, c.content, c.char_start, c.char_end
                  FROM chunk c
                  JOIN file    f ON f.id = c.file_id
                  JOIN project p ON p.id = f.project_id
                 WHERE p.name = %s AND f.rel_path = %s
                 ORDER BY c.seq
                """,
                (project, rel_path),
            )
            rows = cur.fetchall()
            if not rows:
                cur.execute(
                    "SELECT f.rel_path FROM file f JOIN project p ON p.id = f.project_id "
                    "WHERE p.name = %s ORDER BY f.rel_path LIMIT 20",
                    (project,),
                )
                known = [r[0] for r in cur.fetchall()]
                hint = "\n".join(f"  {k}" for k in known) or "  (проект пуст)"
                return f"Файл {rel_path!r} в проекте {project!r} не найден.\nЕсть такие:\n{hint}"
    except Exception as exc:  # noqa: BLE001
        return f"Ошибка чтения: {type(exc).__name__}: {exc}"

    # Сборка по смещениям: фрагменты перекрываются, поэтому при наложении
    # берём только ту часть, которая ещё не попала в результат.
    pieces: list[str] = []
    position = 0
    for _, content, start, end in rows:
        if start is None or end is None:
            pieces.append(content)
            position += len(content)
            continue
        if end <= position:
            continue
        cut = max(0, position - start)
        pieces.append(content[cut:])
        position = end

    text = "".join(pieces)
    header = f"{project}/{rel_path}  ({len(rows)} фрагм., {len(text)} символов)\n{'─' * 60}\n"
    return header + text


@mcp.tool()
def list_projects() -> str:
    """Показать проекты в хранилище: файлы, фрагменты, время последней индексации."""

    try:
        with connect(load_env()) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.name,
                       COALESCE(p.category, '—'),
                       COUNT(DISTINCT f.id)          AS files,
                       COUNT(c.id)                   AS chunks,
                       max(f.indexed_at)             AS last_run
                  FROM project p
                  LEFT JOIN file  f ON f.project_id = p.id
                  LEFT JOIN chunk c ON c.file_id    = f.id
                 GROUP BY p.id, p.name, p.category
                 ORDER BY p.name
                """
            )
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        return f"Ошибка: {type(exc).__name__}: {exc}"

    if not rows:
        return "Хранилище пусто. Запустите индексацию: python -m rag.ingest"

    lines = ["проект | категория | файлов | фрагментов | последняя индексация"]
    for name, category, files, chunks, last_run in rows:
        stamp = last_run.strftime("%Y-%m-%d %H:%M") if last_run else "не индексирован"
        lines.append(f"{name} | {category} | {files} | {chunks} | {stamp}")
    return "\n".join(lines)


@mcp.tool()
def knowledge_stats() -> str:
    """Общая сводка по хранилищу: объёмы, распределение по категориям, последние запуски."""

    try:
        with connect(load_env()) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM project")
            (projects,) = cur.fetchone()
            cur.execute("SELECT count(*) FROM file")
            (files,) = cur.fetchone()
            cur.execute("SELECT count(*), count(embedding) FROM chunk")
            chunks, embedded = cur.fetchone()
            cur.execute(
                """
                SELECT COALESCE(p.category, 'без категории'), count(c.id)
                  FROM project p
                  LEFT JOIN file  f ON f.project_id = p.id
                  LEFT JOIN chunk c ON c.file_id    = f.id
                 GROUP BY 1 ORDER BY 2 DESC
                """
            )
            by_category = cur.fetchall()
            cur.execute(
                """
                SELECT p.name, r.started_at, r.finished_at, r.files_indexed,
                       r.chunks_written, r.error
                  FROM ingest_run r
                  LEFT JOIN project p ON p.id = r.project_id
                 ORDER BY r.started_at DESC LIMIT 5
                """
            )
            runs = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        return f"Ошибка: {type(exc).__name__}: {exc}"

    lines = [
        f"проектов:   {projects}",
        f"файлов:     {files}",
        f"фрагментов: {chunks} (с вектором: {embedded})",
        "",
        "по категориям:",
    ]
    for category, count in by_category:
        lines.append(f"  {category}: {count}")

    if runs:
        lines.append("")
        lines.append("последние запуски индексации:")
        for name, started, finished, indexed, written, error in runs:
            state = "ок" if finished and not error else ("ошибка" if error else "не завершён")
            lines.append(
                f"  {started:%Y-%m-%d %H:%M} {name or '—'}: {state}, "
                f"файлов {indexed}, фрагментов {written}"
            )
            if error:
                lines.append(f"    {error[:200]}")
    return "\n".join(lines)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
