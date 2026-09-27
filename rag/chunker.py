"""Нарезка текста на перекрывающиеся фрагменты.

Смысл: у модели эмбеддингов есть предел длины входа, и всё, что за него
выходит, отбрасывается. Поэтому текст режется заранее — с сохранением
смещений в исходном файле, чтобы потом можно было показать, откуда фрагмент
взят, и вернуть соседей.

Перекрытие делается сдвигом начала следующего фрагмента назад, а не
дописыванием хвоста предыдущего: так `content` остаётся ровно срезом
`text[char_start:char_end]`, и смещения не врут.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

# Размер фрагмента в символах. Модель e5 принимает 512 токенов; для русского
# текста это примерно 1200-1500 символов, для кода — меньше. 800 оставляет
# запас и даёт осмысленные куски.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150

# Границы, по которым разрешено резать: конец абзаца, конец предложения,
# конец строки. Порядок в чередовании `|` не важен — важны сами позиции.
_BREAK_RE = re.compile(r"\n[ \t]*\n|(?<=[.!?])[ \t]+|\n")

# Если разрыв приходится ровно на границу лимита, разрешаем уехать чуть
# дальше — так фрагмент не рвётся на середине слова.
_LOOKAHEAD = 80

# Фрагменты короче этого в конце файла приклеиваются к предыдущему, иначе
# в базе оседает мусор вида «}» с собственным вектором.
_MIN_TAIL = 120


@dataclass(frozen=True)
class Chunk:
    """Фрагмент текста и его место в исходном файле."""

    seq: int
    content: str
    char_start: int
    char_end: int
    token_count: int


def _breakpoints(text: str) -> list[int]:
    """Позиции, по которым можно разорвать текст (конец предыдущего куска)."""

    return sorted({m.end() for m in _BREAK_RE.finditer(text)})


def _estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов.

    Точный токенизатор тянуть не хочется, а для отчётности хватает правила
    «один токен ≈ 4 символа»: на русском оно занижает, на коде завышает, но
    порядок величины даёт верный.
    """

    return max(1, round(len(text) / 4))


def _cut_point(text: str, start: int, limit: int, points: list[int]) -> int:
    """Где закончить фрагмент, начавшийся в `start`, не выходя за `limit`."""

    # Сначала ищем готовую границу: последнюю перед лимитом.
    idx = bisect_right(points, limit) - 1
    if idx >= 0 and points[idx] > start:
        return points[idx]

    # Готовой границы нет — это длинный блок без точек и переводов строк
    # (минифицированный файл, очень длинная строка). Режем по пробелу.
    window = text.rfind(" ", limit - _LOOKAHEAD, limit)
    if window > start:
        return window + 1
    return limit


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Разбить текст на перекрывающиеся фрагменты.

    Возвращает список `Chunk` с честными смещениями: `content` всегда равен
    `text[char_start:char_end]`.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть в диапазоне [0, chunk_size)")

    # Единый вид переводов строк: иначе смещения, посчитанные здесь, не
    # совпадут с тем, что лежит в файле.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []

    points = _breakpoints(text)
    total = len(text)
    spans: list[tuple[int, int]] = []

    start = 0
    while start < total:
        limit = start + chunk_size
        end = total if limit >= total else _cut_point(text, start, limit, points)

        raw = text[start:end]
        # Обрезаем краевые пробелы, сохраняя смещения.
        core_start = start + (len(raw) - len(raw.lstrip()))
        core_end = start + len(raw.rstrip())

        if core_end > core_start:
            spans.append((core_start, core_end))

        if end >= total:
            break
        # Сдвигаем начало назад на величину перекрытия, но гарантированно
        # продвигаемся вперёд, чтобы цикл не зациклился.
        start = max(core_end - overlap, core_start + 1)

    # Короткий хвост приклеиваем к предыдущему фрагменту.
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < _MIN_TAIL:
        last_start, last_end = spans.pop()
        prev_start, _ = spans[-1]
        spans[-1] = (prev_start, last_end)

    chunks: list[Chunk] = []
    for seq, (cs, ce) in enumerate(spans):
        content = text[cs:ce]
        chunks.append(
            Chunk(
                seq=seq,
                content=content,
                char_start=cs,
                char_end=ce,
                token_count=_estimate_tokens(content),
            )
        )
    return chunks
