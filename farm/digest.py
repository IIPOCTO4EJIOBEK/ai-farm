"""Сжатие длинного разговора в короткий бриф.

Зачем это нужно. Разговор с Claude Code растёт и никогда не уменьшается, а
каждый запрос отправляет его провайдеру целиком. Замер 19.09.2026: сессия
на 10.6 МБ, 1432 запроса, в среднем 91 184 токена контекста на запрос,
130 млн перечитанных токенов. Контекста в 42 раза больше, чем нового текста,
и цена каждого шага растёт вместе с длиной разговора.

Идея разделения: «горячее» — текущая задача — остаётся в разговоре,
«холодное» — всё остальное — уезжает в файл и в базу знаний. Тогда новая
сессия начинается с брифа на пару килобайт вместо десяти мегабайт.

Два решения, которые здесь важны:

* **Сжимает бесплатная модель.** Выжимку делает Zen, а не DeepSeek: платная
  модель получает уже сжатое, иначе экономия съедается самим сжатием.
* **Из разговора берётся не всё.** Вывод инструментов — это основной объём
  и почти нулевая ценность для брифа: там простыни файлов и логи. Остаются
  запросы человека, ответы помощника и пути к файлам из вызовов инструментов.
  Такой отбор сохраняет скелет работы и выбрасывает балласт.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .backends import BackendError
from .models import build_pool, pick

CONTEXT_DIR = Path.home() / ".local" / "share" / "ai-farm" / "context"
TRANSCRIPTS = Path.home() / ".claude" / "projects"

# Размер куска для одной выжимки. Бесплатные модели Zen держат заметно
# меньший контекст, чем платные, поэтому режем с запасом.
CHUNK_CHARS = 12_000

# Потолок числа кусков. Разговор на 10 МБ целиком не сжать ни по времени,
# ни по смыслу: свежее важнее старого, поэтому берём хвост.
MAX_CHUNKS = 8

DIGEST_SYSTEM = """Ты — архивариус рабочей сессии. Тебе дают фрагмент стенограммы \
работы инженера с ИИ-ассистентом. Сделай сжатую выжимку.

Пиши только то, что пригодится при продолжении работы:
- что просили сделать и что в итоге решили;
- какие файлы и каталоги затронуты (полные пути);
- какие команды и приёмы сработали, а какие нет;
- что осталось незаконченным и на чём именно остановились.

Не пересказывай переписку, не хвали и не оценивай. Никаких вступлений вроде
«в этом фрагменте». Только факты списком. Если фрагмент пустой по смыслу —
ответь одним словом: ПУСТО."""

MERGE_SYSTEM = """Ты — архивариус. Тебе дают несколько выжимок из одной рабочей \
сессии, по порядку. Сведи их в один связный бриф.

Структура строго такая:
## Задача
## Что сделано
## Файлы и пути
## Решения и причины
## Осталось сделать

Повторы убери. Противоречия между фрагментами разрешай в пользу более позднего.
Пиши по-русски, сжато, без вступлений."""


@dataclass
class Turn:
    """Один ход разговора, очищенный от балласта."""

    role: str
    text: str
    stamp: str = ""


@dataclass
class DigestResult:
    brief: str
    cold_path: Path | None
    turns: int
    chars_in: int
    chars_out: int
    chunks: int
    slots: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.chars_in / max(1, self.chars_out)

    def summary(self) -> str:
        lines = [
            f"ходов разобрано:  {self.turns}",
            f"знаков на входе:  {self.chars_in:,}".replace(",", " "),
            f"знаков в брифе:   {self.chars_out:,}".replace(",", " "),
            f"сжатие:           в {self.ratio:.0f} раз",
            f"кусков на выжимку: {self.chunks}",
            f"модели:           {', '.join(sorted(set(self.slots))) or '—'}",
        ]
        if self.cold_path:
            lines.append(f"полный текст:     {self.cold_path}")
        lines.extend(self.notes)
        return "\n".join(lines)


def latest_transcript(root: Path = TRANSCRIPTS) -> Path | None:
    """Самый свежий журнал основной сессии.

    Раскладка: ``<корень>/<проект>/<сессия>.jsonl``, а журналы подагентов
    лежат глубже — ``<корень>/<проект>/<сессия>/subagents/agent-*.jsonl``.

    Разница принципиальная. Подагент пишет свой журнал прямо во время работы,
    поэтому «самый свежий файл» почти всегда оказывается его журналом — а
    сжимать нужно разговор с человеком, а не побочную переписку агента.
    На этих граблях я уже постоял: первый прогон сжал именно подагента.

    Поэтому берём только журналы верхнего уровня и пропускаем всё, что лежит
    в подкаталоге ``subagents``.
    """

    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for path in root.rglob("*.jsonl"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        # Основная сессия лежит ровно на уровень ниже корня.
        if len(relative.parts) != 2:
            continue
        if "subagents" in relative.parts:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_turns(path: Path) -> list[Turn]:
    """Разобрать журнал сессии в ходы, отбросив балласт.

    Отбрасываем всё, что не несёт смысла для брифа: вывод инструментов
    (основной объём), размышления, служебные записи. От вызовов инструментов
    оставляем только имя и путь — по ним видно, что делали, без простыней.
    """

    turns: list[Turn] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = record.get("message") or {}
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        stamp = str(record.get("timestamp", ""))[11:16]

        if isinstance(message.get("content"), str):
            text = message["content"].strip()
            if text:
                turns.append(Turn(role, text, stamp))
            continue

        parts: list[str] = []
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append(block.get("text", "").strip())
            elif kind == "tool_use":
                # Из вызова инструмента нужен только след: что и над чем.
                name = block.get("name", "?")
                args = block.get("input") or {}
                target = args.get("file_path") or args.get("path") or args.get("command") or ""
                target = str(target).splitlines()[0][:120] if target else ""
                parts.append(f"[{name}: {target}]" if target else f"[{name}]")
            # tool_result и thinking пропускаем осознанно: это и есть объём.
        text = "\n".join(p for p in parts if p).strip()
        if text:
            turns.append(Turn(role, text, stamp))
    return turns


def render(turns: list[Turn], *, tail_chars: int) -> str:
    """Собрать стенограмму, оставив только хвост разговора.

    Свежее важнее старого: чем дальше вглубь, тем меньше в ходах того, что
    ещё имеет значение. Поэтому по умолчанию берём хвост, а не начало.
    """

    lines = [f"[{t.stamp}] {t.role}: {t.text}" for t in turns]
    text = "\n\n".join(lines)
    if len(text) <= tail_chars:
        return text
    cut = text[-tail_chars:]
    # Обрезаем по границе хода, чтобы не начинать с середины фразы.
    head, _, rest = cut.partition("\n\n")
    return rest or head


def chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Порезать стенограмму на куски по границам строк."""

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.splitlines():
        if length + len(line) > size and current:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _free_slots(count: int):
    """Взять бесплатные слоты с умением писать текст."""

    pool = build_pool(include_paid=False)
    chosen = []
    excluded: set[str] = set()
    for index in range(count):
        try:
            chosen.append(pick(pool, skill="text", exclude=excluded, free=True, rotation=index))
        except LookupError:
            break
        excluded.add(chosen[-1].name)
    return chosen


def summarize(text: str, *, max_chunks: int = MAX_CHUNKS, progress=None) -> tuple[str, list[str], int]:
    """Сжать текст бесплатными моделями. Возвращает бриф, имена слотов и число кусков."""

    parts = chunk(text)
    # Хвост важнее начала — если кусков слишком много, выбрасываем самые старые.
    if len(parts) > max_chunks:
        parts = parts[-max_chunks:]

    slots = _free_slots(len(parts) + 1)
    if not slots:
        raise BackendError("нет доступных бесплатных слотов для сжатия")
    used: list[str] = []
    pieces: list[str] = []

    for index, part in enumerate(parts):
        slot = slots[index % len(slots)]
        if progress:
            progress(index + 1, len(parts), slot.name)
        try:
            reply = slot.backend.complete(part, system=DIGEST_SYSTEM, max_tokens=1200)
        except BackendError:
            # Слот не ответил — пробуем следующий, а не теряем кусок.
            for spare in slots:
                if spare.name in used:
                    continue
                try:
                    reply = spare.backend.complete(part, system=DIGEST_SYSTEM, max_tokens=1200)
                    slot = spare
                    break
                except BackendError:
                    continue
            else:
                continue
        used.append(slot.name)
        body = reply.text.strip()
        if body and body.upper() != "ПУСТО":
            pieces.append(body)

    if not pieces:
        return "", used, len(parts)
    if len(pieces) == 1:
        return pieces[0], used, len(parts)

    # Сводим куски в один бриф тем же бесплатным тиром.
    merger = slots[-1]
    joined = "\n\n---\n\n".join(pieces)
    try:
        reply = merger.backend.complete(joined, system=MERGE_SYSTEM, max_tokens=1800)
        used.append(merger.name)
        return reply.text.strip(), used, len(parts)
    except BackendError:
        # Свести не удалось — отдаём куски как есть: это хуже, но не пусто.
        return joined, used, len(parts)


def save_cold(text: str, *, tag: str = "session") -> Path:
    """Положить полный текст в «холодное» хранилище и вернуть путь."""

    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = CONTEXT_DIR / f"{stamp}-{tag}.md"
    path.write_text(text, encoding="utf-8")
    return path


def run(*, session: Path | None = None, tail_chars: int = 400_000,
        max_chunks: int = MAX_CHUNKS, keep_cold: bool = True,
        progress=None) -> DigestResult:
    """Сжать сессию в бриф."""

    path = session or latest_transcript()
    if path is None:
        raise FileNotFoundError("журналов сессий не найдено")
    turns = read_turns(path)
    full = render(turns, tail_chars=tail_chars)
    brief, slots, chunks = summarize(full, max_chunks=max_chunks, progress=progress)
    cold = save_cold(full, tag=path.stem[:8]) if keep_cold else None
    return DigestResult(
        brief=brief,
        cold_path=cold,
        turns=len(turns),
        chars_in=len(full),
        chars_out=len(brief),
        chunks=chunks,
        slots=slots,
    )
