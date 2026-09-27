-- Схема хранилища проектов и знаний.
--
-- Идея: PostgreSQL держит и метаданные, и вектора (pgvector), поэтому
-- отдельная векторная база не нужна — один бэкап, одна транзакция.
--
-- Эмбеддинги 1024-мерные (intfloat/multilingual-e5-large). Если модель
-- сменится, размерность надо поменять здесь и переиндексировать всё.

CREATE EXTENSION IF NOT EXISTS vector;


-- Реестр проектов. Одна строка на папку в ~/projects/ai-farm/projects.
CREATE TABLE IF NOT EXISTS project (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT        NOT NULL UNIQUE,
    path        TEXT        NOT NULL UNIQUE,
    category    TEXT,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- Карта файлов. sha256 нужен, чтобы при повторной индексации трогать
-- только изменившиеся файлы, а не пересчитывать эмбеддинги заново.
CREATE TABLE IF NOT EXISTS file (
    id          BIGSERIAL PRIMARY KEY,
    project_id  BIGINT      NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    rel_path    TEXT        NOT NULL,
    sha256      TEXT        NOT NULL,
    size_bytes  BIGINT      NOT NULL,
    mtime       TIMESTAMPTZ,
    indexed_at  TIMESTAMPTZ,
    chunk_count INTEGER     NOT NULL DEFAULT 0,
    UNIQUE (project_id, rel_path)
);

CREATE INDEX IF NOT EXISTS file_sha256_idx ON file (sha256);


-- Текст чанка и его вектор. seq — порядковый номер внутри файла,
-- нужен, чтобы вернуть фрагмент в правильном порядке и показать соседей.
CREATE TABLE IF NOT EXISTS chunk (
    id          BIGSERIAL PRIMARY KEY,
    file_id     BIGINT  NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    content     TEXT    NOT NULL,
    char_start  INTEGER,
    char_end    INTEGER,
    token_count INTEGER,
    embedding   vector(1024),
    UNIQUE (file_id, seq)
);

-- Косинусное расстояние: HNSW-индекс для быстрого приближённого поиска.
CREATE INDEX IF NOT EXISTS chunk_embedding_idx
    ON chunk USING hnsw (embedding vector_cosine_ops);

-- Полнотекстовый индекс по-русски: даёт гибридный поиск (вектора + слова),
-- что заметно точнее на именах функций, путях и артикулах, где чистая
-- семантика промахивается.
CREATE INDEX IF NOT EXISTS chunk_content_fts_idx
    ON chunk USING gin (to_tsvector('russian', content));


-- Журнал индексаций: видно, когда что запускалось и чем закончилось.
CREATE TABLE IF NOT EXISTS ingest_run (
    id             BIGSERIAL PRIMARY KEY,
    project_id     BIGINT REFERENCES project(id) ON DELETE SET NULL,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    files_seen     INTEGER NOT NULL DEFAULT 0,
    files_indexed  INTEGER NOT NULL DEFAULT 0,
    files_skipped  INTEGER NOT NULL DEFAULT 0,
    chunks_written INTEGER NOT NULL DEFAULT 0,
    error          TEXT
);


-- Гибридный поиск: векторная близость, при желании ограниченная проектом
-- или категорией. Возвращает текст вместе с источником, чтобы модель могла
-- сослаться на файл и строку.
CREATE OR REPLACE FUNCTION search_chunks(
    query_embedding vector(1024),
    query_text      TEXT    DEFAULT NULL,
    match_count     INTEGER DEFAULT 8,
    filter_project  TEXT    DEFAULT NULL,
    filter_category TEXT    DEFAULT NULL
)
RETURNS TABLE (
    chunk_id   BIGINT,
    project    TEXT,
    rel_path   TEXT,
    seq        INTEGER,
    content    TEXT,
    similarity DOUBLE PRECISION
)
LANGUAGE sql STABLE AS $$
    SELECT
        c.id,
        p.name,
        f.rel_path,
        c.seq,
        c.content,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunk c
    JOIN file    f ON f.id = c.file_id
    JOIN project p ON p.id = f.project_id
    WHERE c.embedding IS NOT NULL
      AND (filter_project  IS NULL OR p.name     = filter_project)
      AND (filter_category IS NULL OR p.category = filter_category)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
$$;
