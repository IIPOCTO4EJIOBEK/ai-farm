"""Индексация проектов в хранилище.

Каждый проект — папка в `~/projects/ai-farm/projects/<имя>` с файлом
`project.toml`. Индексатор обходит папку, режет файлы на фрагменты, считает
вектора и складывает всё в PostgreSQL.

Повторный запуск дёшев: у каждого файла хранится sha256, и неизменившиеся
файлы пропускаются, не доходя до модели. Без этого повторная индексация
большого проекта занимала бы столько же, сколько первая.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_text
from .db import connect, load_env, to_vector_literal
from .embedder import Embedder

PROJECTS_ROOT = Path.home() / "projects" / "ai-farm" / "projects"

# Что не индексируем: служебные каталоги и мусор сборки.
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", "dist", "build", ".next", ".cache", "target", ".pylibs",
}

# Расширения, которые заведомо бинарные или бесполезные для поиска.
SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dll", ".exe", ".bin", ".dat",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff",
    ".pdf", ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tar",
    ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".wav", ".flac",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".iso", ".img", ".db", ".sqlite", ".sqlite3", ".lock",
}

# Файлы без расширения, которые всё же надо взять.
KEEP_NAMES = {
    "Dockerfile", "Makefile", "Jenkinsfile", "Gemfile", "Procfile",
    ".gitignore", ".dockerignore", ".env.example", "requirements.txt",
}

# Выше этого размера файл пропускаем: обычно это дампы и логи, которые
# забьют базу мусором.
DEFAULT_MAX_FILE = 5 * 1024 * 1024

# Сколько фрагментов копить перед вызовом модели. Мельчить нельзя: на каждый
# вызов модели есть накладные расходы, и по одному файлу за раз индексация
# идёт в разы медленнее.
EMBED_GROUP = 256

# Порядок важен: utf-8 первый, cp1251 — для старых русских файлов из Windows.
ENCODINGS = ("utf-8", "cp1251")


@dataclass
class ProjectMeta:
    name: str
    path: Path
    category: str | None = None
    description: str | None = None
    # Что не индексировать: имя каталога («decoded») или путь-шаблон
    # («dumps/*.csv»). Нужно там, где в проекте лежат тяжёлые или шумные
    # каталоги: 20.09.2026 в lime-taxi-app распакованный APK — 7373 файла
    # smali, и без исключения они бы уехали в хранилище целиком.
    exclude: tuple[str, ...] = ()


@dataclass
class Stats:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_written: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "files_seen": self.files_seen,
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "chunks_written": self.chunks_written,
        }


# ---------------------------------------------------------------------------
# Разбор папок проектов
# ---------------------------------------------------------------------------


def load_project(path: Path) -> ProjectMeta:
    """Прочитать project.toml. Если его нет — взять имя папки.

    Ключ `path` позволяет завести в базе проект, который лежит не здесь, а
    где-то ещё на диске, — чтобы не копировать существующие репозитории под
    `projects/` только ради индексации.
    """

    meta_file = path / "project.toml"
    data: dict = {}
    if meta_file.is_file():
        try:
            data = tomllib.loads(meta_file.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{meta_file}: не разобран ({exc})") from exc
    project = data.get("project", data)

    target = path
    if project.get("path"):
        target = Path(os.path.expanduser(project["path"]))
        if not target.is_absolute():
            target = (path / target).resolve()

    raw_exclude = project.get("exclude") or []
    if isinstance(raw_exclude, str):
        raw_exclude = [raw_exclude]

    return ProjectMeta(
        name=project.get("name") or path.name,
        path=target,
        category=project.get("category"),
        description=project.get("description"),
        exclude=tuple(str(x) for x in raw_exclude),
    )


def discover_projects(root: Path = PROJECTS_ROOT) -> list[ProjectMeta]:
    if not root.is_dir():
        return []
    # Папки на «.» и «_» — служебные, их не индексируем.
    return [
        load_project(child)
        for child in sorted(root.iterdir())
        if child.is_dir() and not child.name.startswith((".", "_"))
    ]


def should_index(path: Path, max_size: int) -> bool:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    # Файлы без расширения берём только из списка известных имён: остальное
    # обычно бинарники, у которых расширения просто нет.
    if path.suffix == "" and path.name not in KEEP_NAMES:
        return False
    try:
        return path.stat().st_size <= max_size
    except OSError:
        return False


def is_excluded(rel: Path, patterns: tuple[str, ...]) -> bool:
    """Путь под исключение? Сравниваем от корня проекта, как в .gitignore:
    «decoded» ловит каталог на любой глубине, «dumps/*.csv» — шаблон."""

    return any(rel.match(p) for p in patterns)


def walk_files(root: Path, max_size: int, exclude: tuple[str, ...] = ()):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
            and not is_excluded(rel_dir / d, exclude)
        )
        for name in sorted(filenames):
            if is_excluded(rel_dir / name, exclude):
                continue
            path = Path(dirpath) / name
            if should_index(path, max_size):
                yield path


def read_text(path: Path) -> str | None:
    """Прочитать файл как текст. None — если это не текст."""

    try:
        raw = path.read_bytes()
    except OSError:
        return None
    # Признак бинарника: нулевой байт в начале файла.
    if b"\0" in raw[:8192]:
        return None
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Работа с базой
# ---------------------------------------------------------------------------


def ensure_project(conn, meta: ProjectMeta) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO project (name, path, category, description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
                SET path        = EXCLUDED.path,
                    category    = COALESCE(EXCLUDED.category, project.category),
                    description = COALESCE(EXCLUDED.description, project.description),
                    updated_at  = now()
            RETURNING id
            """,
            (meta.name, str(meta.path), meta.category, meta.description),
        )
        return cur.fetchone()[0]


def file_state(conn, project_id: int) -> dict[str, tuple[int, str]]:
    """Карта rel_path -> (file_id, sha256) для уже проиндексированных файлов."""

    with conn.cursor() as cur:
        cur.execute("SELECT id, rel_path, sha256 FROM file WHERE project_id = %s", (project_id,))
        return {rel: (fid, sha) for fid, rel, sha in cur.fetchall()}


def write_group(conn, project_id: int, group: list[tuple[Path, str, str, list]]):
    """Записать пачку файлов: сначала вектора, потом одним махом в базу."""

    all_chunks: list[str] = []
    for _, _, _, chunks in group:
        all_chunks.extend(c.content for c in chunks)

    embedder = _embedder()
    vectors = embedder.embed_passages(all_chunks)

    offset = 0
    with conn.cursor() as cur:
        for path, rel_path, digest, chunks in group:
            stat = path.stat()
            cur.execute(
                """
                INSERT INTO file (project_id, rel_path, sha256, size_bytes, mtime,
                                  indexed_at, chunk_count)
                VALUES (%s, %s, %s, %s, to_timestamp(%s), now(), %s)
                ON CONFLICT (project_id, rel_path) DO UPDATE
                    SET sha256      = EXCLUDED.sha256,
                        size_bytes  = EXCLUDED.size_bytes,
                        mtime       = EXCLUDED.mtime,
                        indexed_at  = now(),
                        chunk_count = EXCLUDED.chunk_count
                RETURNING id
                """,
                (project_id, rel_path, digest, stat.st_size, stat.st_mtime, len(chunks)),
            )
            file_id = cur.fetchone()[0]

            # Фрагменты могли измениться в количестве — старые убираем.
            cur.execute("DELETE FROM chunk WHERE file_id = %s", (file_id,))

            for chunk in chunks:
                cur.execute(
                    """
                    INSERT INTO chunk (file_id, seq, content, char_start, char_end,
                                       token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        file_id,
                        chunk.seq,
                        chunk.content,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.token_count,
                        to_vector_literal(vectors[offset]),
                    ),
                )
                offset += 1
    conn.commit()


_EMBEDDER: Embedder | None = None


def _embedder() -> Embedder:
    """Модель грузится один раз на процесс: это самая дорогая часть запуска."""

    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


# ---------------------------------------------------------------------------
# Основной проход
# ---------------------------------------------------------------------------


def ingest_project(
    conn,
    meta: ProjectMeta,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_size: int = DEFAULT_MAX_FILE,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> Stats:
    stats = Stats()

    if dry_run:
        # Предпросмотр не должен оставлять следов: проект не создаём, запись в
        # журнале запусков не заводим.
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project WHERE name = %s", (meta.name,))
            row = cur.fetchone()
        project_id = row[0] if row else None
        known = {} if force or project_id is None else file_state(conn, project_id)
        run_id = None
    else:
        project_id = ensure_project(conn, meta)
        known = {} if force else file_state(conn, project_id)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_run (project_id) VALUES (%s) RETURNING id",
                (project_id,),
            )
            run_id = cur.fetchone()[0]
        conn.commit()

    group: list[tuple[Path, str, str, list]] = []
    group_chunks = 0

    def flush():
        nonlocal group, group_chunks
        if not group:
            return
        if not dry_run:
            write_group(conn, project_id, group)
        stats.chunks_written += sum(len(c) for _, _, _, c in group)
        group = []
        group_chunks = 0

    try:
        for path in walk_files(meta.path, max_size, meta.exclude):
            stats.files_seen += 1
            rel_path = str(path.relative_to(meta.path))
            try:
                digest = sha256_of(path)
            except OSError as exc:
                stats.files_failed += 1
                stats.errors.append(f"{rel_path}: {exc}")
                continue

            previous = known.get(rel_path)
            if previous and previous[1] == digest:
                stats.files_skipped += 1
                continue

            text = read_text(path)
            if text is None:
                stats.files_skipped += 1
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                stats.files_skipped += 1
                continue

            group.append((path, rel_path, digest, chunks))
            group_chunks += len(chunks)
            stats.files_indexed += 1
            if verbose:
                print(f"  + {rel_path} ({len(chunks)} фрагм.)")

            if group_chunks >= EMBED_GROUP:
                flush()

        flush()

        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingest_run
                       SET finished_at = now(), files_seen = %s,
                           files_indexed = %s, files_skipped = %s,
                           chunks_written = %s, error = %s
                     WHERE id = %s
                    """,
                    (
                        stats.files_seen,
                        stats.files_indexed,
                        stats.files_skipped,
                        stats.chunks_written,
                        "; ".join(stats.errors[:20]) or None,
                        run_id,
                    ),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_run SET finished_at = now(), error = %s WHERE id = %s",
                    (f"{type(exc).__name__}: {exc}", run_id),
                )
            conn.commit()
        raise

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Индексация проектов в хранилище")
    parser.add_argument("project", nargs="?", help="имя проекта (по умолчанию — все)")
    parser.add_argument("--root", type=Path, default=PROJECTS_ROOT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--max-file", type=int, default=DEFAULT_MAX_FILE)
    parser.add_argument("--force", action="store_true", help="переиндексировать всё")
    parser.add_argument("--dry-run", action="store_true", help="только показать, что будет сделано")
    args = parser.parse_args()

    projects = discover_projects(args.root)
    if args.project:
        projects = [p for p in projects if p.name == args.project]
        if not projects:
            print(f"проект {args.project!r} не найден в {args.root}")
            return 1
    if not projects:
        print(f"в {args.root} нет проектов")
        return 1

    env = load_env()
    started = time.monotonic()
    total = Stats()

    with connect(env) as conn:
        for meta in projects:
            print(f"\nпроект «{meta.name}» ({meta.path})")
            if meta.category:
                print(f"  категория: {meta.category}")
            stats = ingest_project(
                conn, meta, chunk_size=args.chunk_size, overlap=args.overlap,
                max_size=args.max_file, force=args.force, dry_run=args.dry_run,
            )
            total.files_seen += stats.files_seen
            total.files_indexed += stats.files_indexed
            total.files_skipped += stats.files_skipped
            total.chunks_written += stats.chunks_written
            total.errors.extend(stats.errors)

    elapsed = time.monotonic() - started
    print(f"\n--- итого за {elapsed:.1f} с ---")
    print(f"просмотрено файлов:  {total.files_seen}")
    print(f"проиндексировано:    {total.files_indexed}")
    print(f"пропущено:           {total.files_skipped}")
    print(f"фрагментов записано: {total.chunks_written}")
    if total.errors:
        print(f"ошибок:              {len(total.errors)}")
        for err in total.errors[:10]:
            print(f"  {err}")
    if args.dry_run:
        print("\n(это был dry-run, в базу ничего не записано)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
