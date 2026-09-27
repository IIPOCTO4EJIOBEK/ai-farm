"""Поиск по хранилищу.

Поиск гибридный: векторная близость плюс полнотекстовый поиск, результаты
объединяются по reciprocal rank fusion.

Зачем оба. Вектора находят смысл, но промахиваются на том, что смысла не
имеет, — именах функций, путях, артикулах, номерах версий: «pgvector 0.8.6»
и «pgvector 0.7.0» для модели почти одно и то же. Слова находят точные
совпадения, но не умеют в синонимы. Вместе покрывают слабости друг друга.

RRF выбран вместо сложения оценок потому, что оценки вектора (косинус) и
текста (ts_rank_cd) в разных шкалах и несравнимы напрямую; RRF работает с
позициями в списках и потому от шкал не зависит.
"""

from __future__ import annotations

from dataclasses import dataclass

from .db import to_vector_literal
from .embedder import Embedder

# Константа сглаживания RRF. 60 — значение из исходной статьи про RRF;
# результат к ней малочувствителен.
RRF_K = 60

DEFAULT_LIMIT = 8

# Сколько кандидатов забирать из каждого списка перед слиянием. Больше, чем
# итоговый лимит: сливать имеет смысл только то, что заведомо шире выдачи.
CANDIDATES_MULTIPLIER = 4

# Сколько фрагментов одного файла пускать в выдачу.
MAX_PER_FILE = 2


@dataclass
class Hit:
    """Найденный фрагмент и сведения о том, откуда он."""

    chunk_id: int
    project: str
    rel_path: str
    seq: int
    content: str
    score: float
    found_by: str  # "vector", "text" или "vector+text"


def _vector_candidates(conn, embedding, limit, project, category) -> list[Hit]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, project, rel_path, seq, content, similarity "
            "FROM search_chunks(%s::vector, NULL, %s, %s, %s)",
            (to_vector_literal(embedding), limit, project, category),
        )
        return [
            Hit(cid, proj, rel, seq, content, float(sim), "vector")
            for cid, proj, rel, seq, content, sim in cur.fetchall()
        ]


def _text_candidates(conn, query, limit, project, category) -> list[Hit]:
    """Полнотекстовый поиск по русскому словарю.

    Запрос собирается из лексем через ИЛИ, а не через готовый
    `plainto_tsquery`. Причина: `plainto_tsquery` соединяет слова через И и
    требует, чтобы все они были в одном фрагменте. Фрагмент — 800 символов,
    и запрос из трёх слов вроде «таймаут ожидания окна» не находил вообще
    ничего, хотя каждое слово по отдельности в тексте есть. С ИЛИ совпадения
    находятся, а `ts_rank_cd` поднимает наверх те фрагменты, где слов больше.

    Лексемы берутся из `to_tsvector`, поэтому словарь, стемминг и стоп-слова
    работают так же, как при индексации, — индекс при этом используется.

    Порядок задаёт не `ts_rank_cd`, а доля найденных слов запроса. Без этого
    фрагмент, где встретилось одно распространённое слово из трёх, вставал
    выше фрагмента, где встретились все три: `ts_rank_cd` считает частоту и
    близость слов, но не то, сколько разных слов запроса покрыто. На практике
    это выглядело так, что README, где есть все слова понемногу, вытеснял
    конкретный файл с ответом.
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            WITH q AS (
                SELECT ARRAY(SELECT lexeme
                               FROM unnest(to_tsvector('russian', %(q)s))) AS lexemes
            ), qq AS (
                SELECT lexemes,
                       to_tsquery('russian',
                                  nullif(array_to_string(lexemes, ' | '), '')) AS tsq
                  FROM q
            )
            SELECT c.id, p.name, f.rel_path, c.seq, c.content,
                   (SELECT count(DISTINCT l.lexeme)
                      FROM unnest(doc.tv) l
                     WHERE l.lexeme = ANY(qq.lexemes))::float
                   / greatest(array_length(qq.lexemes, 1), 1) AS coverage
              FROM chunk c
              JOIN file    f ON f.id = c.file_id
              JOIN project p ON p.id = f.project_id
             CROSS JOIN qq
             -- Вектор документа считается один раз на строку и переиспользуется
             -- и в условии, и в подсчёте покрытия.
             CROSS JOIN LATERAL (SELECT to_tsvector('russian', c.content) AS tv) doc
             -- Если в запросе не осталось ни одной лексемы (одни стоп-слова
             -- или знаки препинания), tsq равен NULL и поиск просто не даёт
             -- результатов — векторная часть отработает одна.
             WHERE qq.tsq IS NOT NULL
               AND doc.tv @@ qq.tsq
               -- Приведение к ::text обязательно: без него PostgreSQL не может
               -- вывести тип параметра в `IS NULL` и роняет запрос с
               -- "could not determine data type of parameter".
               AND (%(proj)s::text IS NULL OR p.name     = %(proj)s::text)
               AND (%(cat)s ::text IS NULL OR p.category = %(cat)s ::text)
             ORDER BY coverage DESC,
                      ts_rank_cd(doc.tv, qq.tsq) DESC,
                      c.id
             LIMIT %(lim)s
            """,
            {"q": query, "proj": project, "cat": category, "lim": limit},
        )
        return [
            Hit(cid, proj, rel, seq, content, float(coverage), "text")
            for cid, proj, rel, seq, content, coverage in cur.fetchall()
        ]


def _fuse(
    vector_hits: list[Hit],
    text_hits: list[Hit],
    limit: int,
    *,
    max_per_file: int = MAX_PER_FILE,
) -> list[Hit]:
    """Объединить два ранжированных списка по reciprocal rank fusion.

    Фрагменты из одного файла ограничены: без ограничения выдача целиком
    заполняется одним документом, который разошёлся на несколько частей. Три
    куска README подряд не дают модели ничего сверх одного.
    """

    scores: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    origin: dict[int, set[str]] = {}

    for source, hits in (("vector", vector_hits), ("text", text_hits)):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            by_id.setdefault(hit.chunk_id, hit)
            origin.setdefault(hit.chunk_id, set()).add(source)

    merged: list[Hit] = []
    per_file: dict[tuple[str, str], int] = {}
    for chunk_id, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        if len(merged) >= limit:
            break
        hit = by_id[chunk_id]
        key = (hit.project, hit.rel_path)
        if per_file.get(key, 0) >= max_per_file:
            continue
        per_file[key] = per_file.get(key, 0) + 1
        merged.append(
            Hit(
                chunk_id=hit.chunk_id,
                project=hit.project,
                rel_path=hit.rel_path,
                seq=hit.seq,
                content=hit.content,
                score=score,
                found_by="+".join(sorted(origin[chunk_id])),
            )
        )
    return merged


def search(
    conn,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    project: str | None = None,
    category: str | None = None,
    embedder: Embedder | None = None,
) -> list[Hit]:
    """Найти фрагменты, отвечающие на запрос."""

    query = query.strip()
    if not query:
        return []

    embedder = embedder or Embedder()
    candidates = max(limit * CANDIDATES_MULTIPLIER, limit)

    vector_hits = _vector_candidates(
        conn, embedder.embed_query(query), candidates, project, category
    )
    text_hits = _text_candidates(conn, query, candidates, project, category)
    return _fuse(vector_hits, text_hits, limit)


def neighbours(conn, chunk_id: int, *, before: int = 1, after: int = 1) -> list[Hit]:
    """Соседние фрагменты того же файла.

    Нужны, когда найденный кусок начинается или обрывается на середине мысли:
    фрагмент режется по 800 символов и может не содержать конца абзаца.
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            WITH target AS (
                SELECT file_id, seq FROM chunk WHERE id = %s
            )
            SELECT c.id, p.name, f.rel_path, c.seq, c.content
              FROM chunk c
              JOIN file    f ON f.id = c.file_id
              JOIN project p ON p.id = f.project_id
              JOIN target  t ON t.file_id = c.file_id
             WHERE c.seq BETWEEN t.seq - %s AND t.seq + %s
             ORDER BY c.seq
            """,
            (chunk_id, before, after),
        )
        return [
            Hit(cid, proj, rel, seq, content, 0.0, "neighbour")
            for cid, proj, rel, seq, content in cur.fetchall()
        ]
