"""Проверки нарезки. Запуск: python3 tests/test_chunker.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import chunk_text  # noqa: E402

CASES: list[tuple[str, callable]] = []


def case(fn):
    CASES.append((fn.__name__, fn))
    return fn


@case
def offsets_honest():
    """content обязан быть ровно срезом исходного текста."""

    text = "\n\n".join(
        f"Абзац номер {i}. " + "Здесь довольно длинное предложение про инфраструктуру. " * 8
        for i in range(40)
    )
    for ch in chunk_text(text):
        assert ch.content == text[ch.char_start : ch.char_end], f"seq={ch.seq}"


@case
def chunks_do_not_exceed_limit():
    text = "Слово " * 5000
    limit = 800
    chunks = chunk_text(text, chunk_size=limit)
    for ch in chunks[:-1]:
        assert len(ch.content) <= limit, f"seq={ch.seq}: {len(ch.content)}"
    assert chunks, "должны получиться фрагменты"


@case
def no_truncation_on_huge_single_line():
    """Тот самый случай из референса: один блок на миллионы символов.

    Там он обрезался до 512 символов и терялся целиком. Здесь обязан
    распасться на множество фрагментов.
    """

    text = "x" * 2_000_000
    chunks = chunk_text(text)
    assert len(chunks) > 2000, f"получилось всего {len(chunks)}"
    covered = sum(len(c.content) for c in chunks)
    assert covered >= len(text), f"покрытие {covered} < {len(text)}"


@case
def overlap_present():
    """Соседние фрагменты должны перекрываться, иначе теряется контекст стыка."""

    text = "\n".join(f"строка {i} с некоторым содержимым" for i in range(300))
    chunks = chunk_text(text, chunk_size=300, overlap=100)
    assert len(chunks) > 3
    overlapping = sum(
        1
        for a, b in zip(chunks, chunks[1:])
        if b.char_start < a.char_end
    )
    assert overlapping >= len(chunks) - 2, f"перекрытий {overlapping}"


@case
def whole_text_covered():
    """Ничего не потеряно: каждый значащий символ попал хотя бы в один фрагмент."""

    text = "Первый абзац.\n\nВторой абзац с текстом.\n\n" + "Ещё текст. " * 400
    chunks = chunk_text(text)
    seen = bytearray(len(text))
    for ch in chunks:
        for i in range(ch.char_start, ch.char_end):
            seen[i] = 1
    for i, byte in enumerate(seen):
        if not text[i].isspace():
            assert byte, f"символ {i} ({text[i]!r}) не попал ни в один фрагмент"


@case
def empty_and_blank():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  \t ") == []


@case
def short_text_single_chunk():
    chunks = chunk_text("Короткий текст.")
    assert len(chunks) == 1
    assert chunks[0].content == "Короткий текст."
    assert chunks[0].char_start == 0


@case
def tiny_tail_merged():
    """Одиночный «}» в конце не должен становиться отдельным фрагментом."""

    text = "Основной текст. " * 200 + "\n\n}"
    chunks = chunk_text(text)
    assert len(chunks[-1].content) >= 120, repr(chunks[-1].content)


@case
def crlf_normalised():
    """Смещения считаются по нормализованному тексту — проверяем, что не рвётся."""

    text = "Строка один.\r\n\r\nСтрока два.\r\n" * 100
    for ch in chunk_text(text):
        assert "\r" not in ch.content


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  ПРОВАЛ {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ОШИБКА {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ок     {name}")
    print(f"\nвсего {len(CASES)}, провалов {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
