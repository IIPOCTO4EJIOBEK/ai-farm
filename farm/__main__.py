"""Командная строка фермы: ``python -m farm "задача"``.

Подкоманды:

* ``run <задача>`` — прогнать задачу через ферму (по умолчанию);
* ``pool`` — показать слоты: кто бесплатный, кто платный и какими умениями
  наделён;
* ``route <текст>`` — показать, какой домен выберется под текст: полезно,
  когда ответ специалиста выглядит не по теме;
* ``roles`` — вывести домены и их критерии приёмки;
* ``usage`` — баланс DeepSeek, расход токенов и экономия от бесплатных
  моделей.

Полезные ключи: ``--free`` (запретить платный тир), ``--quiet`` (только
ответ), ``--max-subtasks``, ``--max-iter``, ``--workers``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import digest as digest_module
from . import ledger as ledger_module
from . import notes as notes_module
from . import websearch
from .backends import BackendError
from .ledger import Ledger, calibrate, load, summarize
from .models import build_pool
from .orchestrator import Farm
from .roles import DOMAINS, GENERAL, domain_by_key, route


def _cmd_pool(args: argparse.Namespace) -> int:
    pool = build_pool(include_paid=not args.free)
    free = [slot for slot in pool if slot.free]
    print(f"слотов всего: {len(pool)}, из них бесплатных: {len(free)}")
    print("порядок: очерёдность — чем меньше, тем раньше берут; качество — чем больше, тем сильнее модель.\n")
    print(f"{'слот':<38} {'цена':<5} {'очер':<5} {'кач':<4} умения")
    for slot in sorted(pool, key=lambda item: (item.tier, item.name)):
        print(
            f"{slot.name:<38} {'free' if slot.free else 'PAID':<5} "
            f"{slot.tier:<5} {slot.quality:<4} {','.join(sorted(slot.skills))}"
        )
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    domain = route(" ".join(args.text))
    print(f"домен: {domain.key} — {domain.title}")
    print("\nкритерии приёмки:")
    for index, criterion in enumerate(domain.criteria, 1):
        print(f"  {index}. {criterion}")
    return 0


def _cmd_roles(_: argparse.Namespace) -> int:
    for domain in (GENERAL, *DOMAINS):
        print(f"\n=== {domain.key} — {domain.title}")
        print(f"исполнитель: {domain.persona}")
        print("критерии:")
        for criterion in domain.criteria:
            print(f"  - {criterion}")
    return 0


def _money(value: float) -> str:
    return ledger_module.money(value)


def _cmd_usage(args: argparse.Namespace) -> int:
    """Показать баланс, расход и экономию."""

    records = load()
    totals = summarize(records)
    prices, verified = Ledger().prices, Ledger().prices_verified

    print("БАЛАНС DEEPSEEK")
    info = ledger_module.balance()
    if info is None:
        print("  недоступен: не ответил api.deepseek.com или нет ключа в ~/.fcc/.env")
    else:
        print(f"  сейчас:   {_money(float(info['total']))} {info['currency']}")
        print(f"  пополнено: {_money(float(info['topped_up']))}, грант: {_money(float(info['granted']))}")
        print(f"  доступен: {'да' if info['available'] else 'НЕТ'}")
        ledger_module.save_balance_snapshot(info)

    # Раздел идёт сразу после баланса намеренно: это самая крупная статья
    # расхода, и без него консоль показывает экономию на копейках, умалчивая
    # о рублях. Числа точные — их пишет сам клиент по каждому запросу.
    usage = ledger_module.session_usage()
    print("\nСЕССИИ CLAUDE CODE  (тот же DeepSeek; учёт фермы этого не видит)")
    for line in usage.lines():
        if line:
            print(line)
    if usage.requests:
        models = ", ".join(f"{name} — {count}" for name, count in
                           sorted(usage.models.items(), key=lambda item: -item[1]))
        print(f"  модели:              {models}")
        print(f"  всего оплачиваемых токенов: {usage.billed_tokens:,}".replace(",", " "))
        print("  цена запроса растёт вместе с длиной разговора: контекст перечитывается целиком")

    if not totals.calls:
        print("\nРАСХОД\n  записей нет — журнал пуст, ферма ещё не вызывала моделей")
        print(f"\nжурнал: {ledger_module.LEDGER_FILE}")
        return 0

    first = time.strftime("%d.%m.%Y %H:%M", time.localtime(min(r.ts for r in records)))
    last = time.strftime("%d.%m.%Y %H:%M", time.localtime(max(r.ts for r in records)))

    print(f"\nРАСХОД  (учёт ведётся с {first} по {last})")
    print(f"  вызовов:            {totals.calls}")
    print(f"  из них бесплатных:  {totals.free_calls} ({totals.free_share:.0%})")
    print(f"  токенов всего:      {totals.tokens:,}")
    print(f"    на бесплатных:    {totals.free_tokens:,}")
    print(f"    на платных:       {totals.paid_tokens:,}")
    if totals.estimated_calls:
        print(f"  оценка по длине текста: {totals.estimated_calls} вызовов (Zen не отдаёт счётчики)")
    print(f"  потрачено:          {_money(totals.spent)}")
    mark = "" if verified else "   ← ставки не сверены, это оценка"
    print(f"  сэкономлено:        {_money(totals.saved)}{mark}")
    if totals.spent + totals.saved > 0:
        share = totals.saved / (totals.spent + totals.saved)
        print(
            f"  без бесплатных моделей счёт был бы больше на {share:.0%} "
            f"({_money(totals.saved + totals.spent)} вместо {_money(totals.spent)})"
        )

    print("\nПО СЛОТАМ")
    for slot, count in sorted(totals.by_slot.items(), key=lambda item: -item[1])[:12]:
        tokens = sum(r.tokens for r in records if r.slot == slot)
        print(f"  {slot:<34} {count:>4} вызовов  {tokens:>9,} токенов")

    spent_money = ledger_module.spent_between(ledger_module.load_balance_snapshots(), min(r.ts for r in records))
    derived = calibrate(records, spent_money) if spent_money is not None else None
    prices_file = ledger_module.ensure_prices_file()
    print(f"\nСТАВКИ  (файл {prices_file})")
    for name, rate in sorted(prices.items()):
        print(f"  {name:<30} ${rate.input:.2f}/М вх   ${rate.output:.2f}/М вых")
    if derived is not None:
        print(
            f"\n  по замерам баланса выходит {derived:.2f} $/М токенов.\n"
            "  Это верхняя граница: в разницу баланса попадает и всё остальное,\n"
            "  что ходило в DeepSeek между замерами, — например сессии Claude Code."
        )
    elif spent_money is None:
        print(
            "\n  Калибровка появится, когда накопится история баланса:\n"
            "  снимок делается при каждом запуске `ai-farm usage`."
        )
    return 0


PREVIEW_DIR = Path.home() / ".local" / "share" / "ai-preview"


def _unwrap_code_fence(text: str) -> str:
    """Снять обёртку ``` из ответа модели.

    Модель почти всегда отдаёт код в ограде с указанием языка. Если записать
    такой ответ в файл как есть, страница или скрипт окажутся битыми: первая
    строка станет ```html вместо <!DOCTYPE html>. Снимаем ровно одну внешнюю
    ограду, внутренние не трогаем — они часть содержимого.
    """

    lines = text.strip().splitlines()
    if len(lines) < 2 or not lines[0].lstrip().startswith("```"):
        return text
    if lines[-1].strip() != "```":
        return text
    return "\n".join(lines[1:-1])


def _cmd_run(args: argparse.Namespace) -> int:
    task = " ".join(args.task).strip()
    if not task:
        print("нужна задача: ai-farm \"что сделать\"", file=sys.stderr)
        return 2
    if args.web and not websearch.alive():
        # Искать нечем — предупреждаем сразу, а не после прогона: иначе
        # окажется, что модель отвечала без свежих данных, а мы этого не знали.
        print(
            f"веб-поиск не отвечает ({websearch.endpoint()}) — задача пойдёт без него",
            file=sys.stderr,
        )
    farm = Farm(
        max_subtasks=args.max_subtasks,
        max_iter=args.max_iter,
        workers=args.workers,
        max_paid_calls=args.max_paid_calls,
        include_paid=not args.free,
        default_domain=domain_by_key(args.domain) if args.domain else GENERAL,
        web=args.web,
        web_limit=args.web_limit,
    )
    result = farm.run(task)
    if not args.quiet:
        print(result.summary(), file=sys.stderr)
        print("\n" + "=" * 60 + "\n", file=sys.stderr)

    # Куда девать ответ. Смысл ключей --out и --preview в том, чтобы объёмный
    # результат не возвращался в контекст вызывающего: наружу уходит только
    # отчёт о файле. Иначе экономия фермы съедается перечитыванием её ответа.
    if args.out or args.preview:
        body = _unwrap_code_fence(result.answer)
        if not body.strip():
            # Пустой файл — не результат. Раньше он записывался молча, и
            # прогон выглядел удачным: проверено 19.09.2026, когда все
            # воркеры упали, а страница вышла нулевого размера.
            print("ферма вернула пустой результат — файл не записан", file=sys.stderr)
            return 1
        if args.preview:
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            safe = "".join(c for c in args.preview if c.isalnum() or c in "-_") or "index"
            target = PREVIEW_DIR / f"{safe}.html"
        else:
            target = Path(args.out).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        print(f"записано: {target}")
        print(f"размер:   {target.stat().st_size} байт")
        if args.preview:
            print(f"открыть:  http://127.0.0.1:8791/{target.stem}")
        print(f"строк:    {body.count(chr(10)) + 1}")
        return 0

    print(result.answer)
    return 0 if result.answer else 1


def _cmd_digest(args: argparse.Namespace) -> int:
    """Сжать сессию в бриф бесплатными моделями."""

    def progress(index: int, total: int, slot: str) -> None:
        print(f"  кусок {index}/{total}: {slot}", file=sys.stderr, flush=True)

    session = Path(args.session).expanduser() if args.session else None
    print("сжимаю разговор…", file=sys.stderr, flush=True)
    try:
        result = digest_module.run(
            session=session,
            tail_chars=args.tail,
            max_chunks=args.max_chunks,
            keep_cold=not args.no_cold,
            progress=progress,
        )
    except (FileNotFoundError, BackendError) as exc:
        print(f"не удалось сжать: {exc}", file=sys.stderr)
        return 1

    print(result.summary(), file=sys.stderr)
    print("\n" + "=" * 60 + "\n", file=sys.stderr)
    if args.out:
        target = Path(args.out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.brief, encoding="utf-8")
        print(f"бриф записан: {target}")
        return 0
    print(result.brief)
    return 0 if result.brief else 1


def _cmd_note(args: argparse.Namespace) -> int:
    """Положить текст в заметки, вернув короткий идентификатор."""

    text = " ".join(args.text).strip()
    if not text and not sys.stdin.isatty():
        # Позволяет писать в заметку из конвейера, не гоняя текст через
        # аргументы командной строки.
        text = sys.stdin.read().strip()
    if not text:
        print("нужен текст: ai-farm note \"что запомнить\"", file=sys.stderr)
        return 2
    note = notes_module.save(text, tags=args.tag)
    print(note.line())
    print(f"файл: {note.path}")
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    """Найти заметки или показать одну по идентификатору."""

    if args.show:
        text = notes_module.get(args.show)
        if text is None:
            print(f"заметка не найдена: {args.show}", file=sys.stderr)
            return 1
        print(text)
        return 0

    query = " ".join(args.query).strip()
    found = notes_module.find(query, limit=args.limit)
    if not found:
        print("ничего не найдено" if query else "заметок пока нет")
        return 1
    print(f"каталог: {notes_module.NOTES_DIR}\n")
    for note in found:
        print(note.line())
        body = notes_module.snippet(note)
        if body:
            print(f"      {body}")
    return 0


SUBCOMMANDS = ("run", "pool", "route", "roles", "usage", "digest", "note", "recall")

# Ключи, за которыми идёт значение. Нужны, чтобы отличить значение ключа от
# начала текста задачи при разборе аргументов ниже.
VALUE_FLAGS = ("--max-subtasks", "--max-iter", "--max-paid-calls", "--workers", "--domain",
               "--out", "--preview")


def _extract_output_flags(argv: list[str]) -> list[str]:
    """Вынести ``--out`` и ``--preview`` из любого места команды в начало.

    Зачем это нужно. У подкоманды ``run`` текст задачи объявлен как
    ``REMAINDER`` — «всё до конца строки». Это удобно: в задаче можно
    писать дефисы и кавычки. Но у медали есть обратная сторона: ключ,
    поставленный ПОСЛЕ текста, попадает не в разбор, а в саму задачу.
    ``ai-farm "сделай список" --out файл`` молча печатал список в консоль,
    а файла не создавал.

    Ошибка коварна тем, что выглядит как успех. В прогоне 19.09.2026 файл
    всё-таки появился — но его записала модель через MCP-инструмент
    server-filesystem, о чём никто не просил. Своя работа и чужая
    выглядят одинаково, пока не посмотришь на содержимое.

    Поэтому разбираем ключи до argparse: забираем их вместе со значениями и
    возвращаем в начало, где главный разбор их увидит.
    """

    kept: list[str] = []
    out = preview = ""
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("--out", "--preview") and index + 1 < len(argv):
            if token == "--out":
                out = argv[index + 1]
            else:
                preview = argv[index + 1]
            index += 2
            continue
        kept.append(token)
        index += 1

    prefix: list[str] = []
    if out:
        prefix += ["--out", out]
    if preview:
        prefix += ["--preview", preview]
    return prefix + kept


def _insert_default_command(argv: list[str]) -> list[str]:
    """Подставить ``run``, если подкоманда не названа.

    Без этого ``ai-farm --free "задача"`` разбирался бы как имя подкоманды, и
    argparse ругался бы «invalid choice». Ищем первый аргумент, не похожий на
    ключ, пропуская значения ключей, и если это не известная подкоманда —
    считаем его началом задачи.
    """

    index = 0
    while index < len(argv):
        token = argv[index]
        if token in VALUE_FLAGS:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        if token in SUBCOMMANDS:
            return argv
        return argv[:index] + ["run"] + argv[index:]
    return argv + ["run"]


def _named_command(argv: list[str]) -> str:
    """Какую подкоманду назвали: первое слово, не похожее на ключ.

    Значения ключей пропускаем, иначе текст задачи после ``--out`` сошёл бы за
    имя подкоманды. Если подкоманды нет — считаем, что это ``run``.
    """

    index = 0
    while index < len(argv):
        token = argv[index]
        if token in VALUE_FLAGS:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token if token in SUBCOMMANDS else "run"
    return "run"


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Ключи --out/--preview выносим в начало только для `run` (и когда
    # подкоманда не названа). У `digest` свой --out, и вынос его ломает:
    # ключ подбирает главный разбор, подкоманда получает пустую строку и
    # печатает бриф в вывод вместо файла — ровно то, от чего ключ и спасает.
    # Проверено 20.09.2026: `ai-farm digest --out FILE` файла не создавал.
    if _named_command(raw) != "digest":
        raw = _extract_output_flags(raw)
    argv = _insert_default_command(raw)
    parser = argparse.ArgumentParser(prog="farm", description=__doc__.splitlines()[0])
    parser.add_argument("--free", action="store_true", help="только бесплатные слоты")
    parser.add_argument("--quiet", action="store_true", help="печатать только ответ")
    parser.add_argument("--max-subtasks", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=2, help="потолок переделок на подзадачу")
    parser.add_argument("--max-paid-calls", type=int, default=3, help="потолок платных вызовов")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--domain", default="", help="домен вручную; иначе по ключевым словам")
    parser.add_argument(
        "--web",
        action="store_true",
        help="подложить исполнителям свежую выдачу локального SearXNG",
    )
    parser.add_argument("--web-limit", type=int, default=6, help="сколько ссылок брать из поиска")
    # Ключи ниже отдают результат в файл вместо stdout. Это не удобство, а
    # экономия: ответ объёмом в десятки килобайт, напечатанный в вывод, осядет
    # в контексте вызывающего и будет перечитываться при каждом следующем
    # запросе. Через --out наружу уходит только строка отчёта.
    parser.add_argument("--out", default="", help="записать ответ в файл, не печатая его")
    parser.add_argument("--preview", default="", help="положить ответ в живой предпросмотр под этим именем")
    subparsers = parser.add_subparsers(dest="command")
    runner = subparsers.add_parser("run", help="прогнать задачу через ферму")
    runner.add_argument("task", nargs=argparse.REMAINDER)
    subparsers.add_parser("pool", help="показать слоты")
    router = subparsers.add_parser("route", help="показать выбранный домен")
    router.add_argument("text", nargs="+")
    subparsers.add_parser("roles", help="домены и критерии приёмки")
    subparsers.add_parser("usage", help="баланс DeepSeek, расход и экономия")

    digester = subparsers.add_parser("digest", help="сжать сессию в бриф бесплатными моделями")
    digester.add_argument("--session", default="", help="путь к журналу; иначе самый свежий")
    digester.add_argument("--tail", type=int, default=400_000, help="сколько знаков хвоста брать")
    digester.add_argument("--max-chunks", type=int, default=8, help="потолок кусков на выжимку")
    digester.add_argument("--out", default="", help="записать бриф в файл")
    digester.add_argument("--no-cold", action="store_true", help="не сохранять полный текст")

    noter = subparsers.add_parser("note", help="запомнить текст, вернув короткий id")
    # Не REMAINDER: он забирает всё до конца строки, и ключи --tag оказались бы
    # внутри текста заметки. nargs="+" останавливается на ключе.
    noter.add_argument("text", nargs="+")
    noter.add_argument("--tag", action="append", default=[], help="тег; можно повторять")

    recaller = subparsers.add_parser("recall", help="найти заметки или показать одну")
    recaller.add_argument("query", nargs="*")
    recaller.add_argument("--show", default="", help="показать заметку по id")
    recaller.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "pool": _cmd_pool,
        "route": _cmd_route,
        "roles": _cmd_roles,
        "usage": _cmd_usage,
        "digest": _cmd_digest,
        "note": _cmd_note,
        "recall": _cmd_recall,
    }
    return handlers[args.command or "run"](args)


if __name__ == "__main__":
    raise SystemExit(main())
