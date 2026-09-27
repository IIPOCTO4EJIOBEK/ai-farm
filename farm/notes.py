"""Заметки: факты живут вне разговора, в разговоре остаётся только id.

Зачем. Любой текст, попавший в переписку, остаётся в ней навсегда и
перечитывается каждым следующим запросом. Длинный фрагмент — конфиг, вывод
команды, кусок документации — дешевле положить в файл, а в разговор вернуть
короткий идентификатор. Понадобится — достанем по id или найдём поиском.

Хранилище намеренно простое: файлы плюс журнал-указатель. База знаний с
pgvector умеет искать по смыслу, но требует запущенного Postgres и
векторизации; заметки должны работать всегда, в том числе когда база лежит.
Когда база поднята — заметку можно отправить и туда, но это не обязательно.

Идентификатор делается из времени: он короткий, сортируется по порядку
появления и не требует счётчика, который пришлось бы согласовывать между
процессами.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

NOTES_DIR = Path.home() / ".local" / "share" / "ai-farm" / "notes"
INDEX_FILE = NOTES_DIR / "index.jsonl"

SNIPPET_CHARS = 240
MAX_TITLE_CHARS = 70


@dataclass
class Note:
    id: str
    title: str
    tags: list[str]
    ts: float
    path: Path
    size: int

    def when(self) -> str:
        return time.strftime("%d.%m %H:%M", time.localtime(self.ts))

    def line(self) -> str:
        tags = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"{self.id}  {self.when()}  {self.size:>7} б{tags}  {self.title}"


def _slug(text: str) -> str:
    """Короткое имя файла из первой строки заметки."""

    words = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text)[:6]
    return "-".join(w.lower() for w in words)[:MAX_TITLE_CHARS] or "заметка"


def _title(text: str) -> str:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "заметка")
    first = re.sub(r"^#+\s*", "", first)
    return first[:MAX_TITLE_CHARS]


def save(text: str, *, tags: list[str] | None = None, note_id: str | None = None) -> Note:
    """Сохранить заметку и вернуть её описание.

    Идентификатор по умолчанию — время с точностью до секунды. Если за ту же
    секунду записали две заметки, добавляем счётчик: иначе вторая затрёт
    первую, и потеря будет незаметной.
    """

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    base = note_id or time.strftime("%Y%m%d-%H%M%S")
    ident = base
    counter = 1
    while (NOTES_DIR / f"{ident}.md").exists():
        counter += 1
        ident = f"{base}-{counter}"

    path = NOTES_DIR / f"{ident}.md"
    tags = [t.strip() for t in (tags or []) if t.strip()]
    header = f"# {_title(text)}\n\n"
    if tags:
        header += f"теги: {', '.join(tags)}\n\n"
    path.write_text(header + text + "\n", encoding="utf-8")

    note = Note(id=ident, title=_title(text), tags=tags, ts=time.time(),
                path=path, size=path.stat().st_size)
    with INDEX_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "id": note.id, "title": note.title, "tags": note.tags,
            "ts": note.ts, "size": note.size, "file": path.name,
        }, ensure_ascii=False) + "\n")
    return note


def load_index() -> list[Note]:
    """Прочитать указатель заметок.

    Указатель может разойтись с каталогом: заметку могли удалить руками.
    Поэтому при чтении проверяем, что файл на месте, и такие записи
    пропускаем — иначе поиск будет предлагать то, чего нет.
    """

    if not INDEX_FILE.is_file():
        return []
    notes: list[Note] = []
    for line in INDEX_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = NOTES_DIR / str(data.get("file") or f"{data.get('id')}.md")
        if not path.is_file():
            continue
        notes.append(Note(
            id=str(data.get("id", "?")),
            title=str(data.get("title", "")),
            tags=list(data.get("tags") or []),
            ts=float(data.get("ts") or 0),
            path=path,
            size=path.stat().st_size,
        ))
    return notes


def find(query: str = "", *, limit: int = 20) -> list[Note]:
    """Найти заметки по всем словам запроса.

    Ищем по заголовку, тегам и телу. Требуем все слова, а не любое: при
    поиске по любому слову запрос «такси аэропорт» выдаёт всё, где
    встречается «такси», и нужное теряется в списке.
    """

    notes = sorted(load_index(), key=lambda n: -n.ts)
    words = [w.lower() for w in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", query)]
    if not words:
        return notes[:limit]

    hits: list[Note] = []
    for note in notes:
        try:
            body = note.path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        haystack = f"{note.title.lower()} {' '.join(note.tags).lower()} {body}"
        if all(word in haystack for word in words):
            hits.append(note)
        if len(hits) >= limit:
            break
    return hits


def get(ident: str) -> str | None:
    """Прочитать заметку по идентификатору.

    Принимаем и полный id, и однозначное начало: печатать целиком
    «20260919-014233-3» всякий раз, когда хватает «20260919-0142», незачем.
    """

    exact = NOTES_DIR / f"{ident}.md"
    if exact.is_file():
        return exact.read_text(encoding="utf-8")
    matches = [n for n in load_index() if n.id.startswith(ident)]
    if len(matches) == 1:
        return matches[0].path.read_text(encoding="utf-8")
    return None


def snippet(note: Note) -> str:
    """Первая осмысленная строка тела — для показа в списке."""

    try:
        text = note.path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("теги:"):
            return line[:SNIPPET_CHARS]
    return ""
