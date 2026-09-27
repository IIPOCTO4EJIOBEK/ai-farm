"""Учёт расхода: кто сколько токенов съел и сколько это стоило бы на платном тире.

Зачем отдельный модуль. Ферма существует ради экономии, а экономию нельзя
увидеть, если не считать. Считать надо две разные вещи:

* **фактические деньги** — сколько реально ушло с баланса DeepSeek;
* **сэкономленное** — сколько стоили бы на платном тире те вызовы, которые
  ушли на бесплатные слоты. Это оценка, а не факт, и она честна ровно
  настолько, насколько верны ставки.

Отсюда два принципа, заложенные в код:

1. **Факт важнее оценки.** Баланс берётся у DeepSeek напрямую, а не
   вычисляется по ставкам. Ставки нужны только для счётчика экономии.
2. **Ставки — настраиваемые и датированные.** Они лежат в
   ``~/.config/ai-farm/prices.json``, значения по умолчанию помечены как
   непроверенные (``verified: false``), и есть ``calibrate()``: она выводит
   фактическую ставку из изменения баланса. Придумывать цены за провайдера
   нельзя — счёта за них не будет, а вывод получится ложный.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LEDGER_FILE = Path.home() / ".local" / "share" / "ai-farm" / "usage.jsonl"
PRICES_FILE = Path.home() / ".config" / "ai-farm" / "prices.json"
FCC_ENV = Path.home() / ".fcc" / ".env"
BALANCE_URL = "https://api.deepseek.com/user/balance"


# ----------------------------------------------------------------------
# Ставки
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Rate:
    """Ставка за миллион токенов в долларах.

    ``input`` — это cache miss, как в прайсе провайдера; попадание в кеш
    стоит в пятьдесят раз дешевле, но у нас его отдельно не считают, поэтому
    оценка экономии идёт по худшему для нас случаю (cache miss) — завышать
    экономию нечем.

    ``input_offpeak``/``output_offpeak`` — непиковые часы: у DeepSeek они
    ровно вдвое дешевле пиковых (проверено по прайсу 19.09.2026). Если их не
    задать, они выводятся из пиковых делением пополам.
    """

    input: float
    output: float
    input_offpeak: float | None = None
    output_offpeak: float | None = None

    def at(self, ts: float | None = None) -> tuple[float, float]:
        """Ставки (вход, выход) для момента ``ts``: в пик дороже вдвое."""

        if ts is None or is_peak(ts):
            return self.input, self.output
        return (
            self.input_offpeak if self.input_offpeak is not None else self.input / 2,
            self.output_offpeak if self.output_offpeak is not None else self.output / 2,
        )

    def cost(self, tokens_in: int, tokens_out: int, ts: float | None = None) -> float:
        rate_in, rate_out = self.at(ts)
        return (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000


def is_peak(ts: float) -> bool:
    """Пиковые часы DeepSeek: 01:00–04:00 и 06:00–10:00 UTC по будням.

    Время берётся из отметки самого вызова, а не из «сейчас»: иначе пересчёт
    старого журнала зависел бы от того, когда его читают.
    """

    moment = datetime.fromtimestamp(ts, tz=timezone.utc)
    if moment.weekday() >= 5:  # суббота и воскресенье целиком непиковые
        return False
    return 1 <= moment.hour < 4 or 6 <= moment.hour < 10


# Ставки по умолчанию — с прайса DeepSeek от 19.09.2026
# (https://api-docs.deepseek.com/quick_start/pricing/). Платный тир у нас —
# deepseek-flash; имена deepseek-v4-flash и deepseek-chat обслуживаются той же
# моделью DeepSeek-V4.1-Flash и тарифицируются по цене Flash (сноска прайса).
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "deepseek/deepseek-v4-flash": {"input": 0.30, "output": 1.20,
                                   "input_offpeak": 0.15, "output_offpeak": 0.60},
    "deepseek/deepseek-v4-pro": {"input": 1.32, "output": 3.96,
                                 "input_offpeak": 0.66, "output_offpeak": 1.98},
    "deepseek/deepseek-chat": {"input": 0.30, "output": 1.20,
                               "input_offpeak": 0.15, "output_offpeak": 0.60},
}
DEFAULT_RATE = Rate(input=0.30, output=1.20, input_offpeak=0.15, output_offpeak=0.60)


def _opt_float(value: object) -> float | None:
    """Необязательная ставка из JSON: None, если её там нет или она не число."""

    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_prices() -> tuple[dict[str, Rate], bool]:
    """Прочитать ставки. Возвращает ставки и признак «проверены человеком».

    ``verified`` — не украшение: пока он ложный, вывод об экономии помечается
    как оценка, и никто не примет выдуманное число за факт.
    """

    prices = {name: Rate(**value) for name, value in DEFAULT_PRICES.items()}
    if not PRICES_FILE.is_file():
        return prices, False
    try:
        raw = json.loads(PRICES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return prices, False
    verified = bool(raw.pop("verified", False)) if isinstance(raw, dict) else False
    for name, value in (raw or {}).items():
        if isinstance(value, dict) and "input" in value and "output" in value:
            prices[name] = Rate(
                input=float(value["input"]),
                output=float(value["output"]),
                input_offpeak=_opt_float(value.get("input_offpeak")),
                output_offpeak=_opt_float(value.get("output_offpeak")),
            )
    return prices, verified


def rate_for(prices: dict[str, Rate], model: str) -> Rate:
    """Ставка для модели. Неизвестная модель считается по ставке по умолчанию."""

    if model in prices:
        return prices[model]
    short = model.split("/")[-1]
    for name, rate in prices.items():
        if name.split("/")[-1] == short:
            return rate
    return DEFAULT_RATE


# ----------------------------------------------------------------------
# Записи
# ----------------------------------------------------------------------


@dataclass
class Record:
    """Один вызов модели."""

    ts: float
    slot: str
    model: str
    free: bool
    tokens_in: int
    tokens_out: int
    estimated_tokens: bool = False
    cost_usd: float = 0.0
    would_cost_usd: float = 0.0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class Ledger:
    """Журнал вызовов. Пишется построчно, чтобы падение не потеряло историю."""

    def __init__(self, path: Path = LEDGER_FILE, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self.prices, self.prices_verified = load_prices()
        self.session: list[Record] = []

    def record(
        self,
        *,
        slot: str,
        model: str,
        free: bool,
        tokens_in: int,
        tokens_out: int,
        estimated_tokens: bool = False,
    ) -> Record:
        rate = rate_for(self.prices, model)
        now = time.time()
        would = rate.cost(tokens_in, tokens_out, now)
        entry = Record(
            ts=now,
            slot=slot,
            model=model,
            free=free,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_tokens=estimated_tokens,
            cost_usd=0.0 if free else would,
            would_cost_usd=would,
        )
        self.session.append(entry)
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            except OSError:
                # Учёт не должен ронять работу фермы: не записалось — не беда.
                pass
        return entry

    # ------------------------------------------------------------------

    def session_totals(self) -> "Totals":
        return summarize(self.session)

    def all_totals(self) -> "Totals":
        return summarize(load(self.path))

    def session_line(self) -> str:
        """Строка об экономии за текущий прогон — для вывода после ответа."""

        totals = self.session_totals()
        if not totals.calls:
            return ""
        mark = "" if self.prices_verified else " (оценка: ставки не сверены)"
        return (
            f"вызовов: {totals.calls} (бесплатных {totals.free_calls}) | "
            f"токенов: {totals.tokens:,} | "
            f"потрачено: {money(totals.spent)} | "
            f"сэкономлено: {money(totals.saved)}{mark}"
        )


@dataclass
class Totals:
    calls: int = 0
    free_calls: int = 0
    tokens: int = 0
    free_tokens: int = 0
    paid_tokens: int = 0
    spent: float = 0.0
    saved: float = 0.0
    estimated_calls: int = 0
    by_slot: dict[str, int] = field(default_factory=dict)

    @property
    def free_share(self) -> float:
        return self.free_calls / self.calls if self.calls else 0.0


def money(value: float) -> str:
    """Сумма с точностью по величине.

    Один вызов модели стоит доли цента, и при четырёх знаках экономия за
    прогон выглядела бы как $0.0000 — то есть как отсутствие экономии.
    Поэтому мелкие суммы показываются с шестью знаками.
    """

    size = abs(value)
    if size >= 1:
        return f"${value:,.2f}"
    if size >= 0.01:
        return f"${value:,.4f}"
    return f"${value:,.6f}"


def load(path: Path = LEDGER_FILE) -> list[Record]:
    if not path.is_file():
        return []
    records: list[Record] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(
            Record(
                ts=float(payload.get("ts", 0)),
                slot=str(payload.get("slot", "?")),
                model=str(payload.get("model", "?")),
                free=bool(payload.get("free", False)),
                tokens_in=int(payload.get("tokens_in", 0)),
                tokens_out=int(payload.get("tokens_out", 0)),
                estimated_tokens=bool(payload.get("estimated_tokens", False)),
                cost_usd=float(payload.get("cost_usd", 0)),
                would_cost_usd=float(payload.get("would_cost_usd", 0)),
            )
        )
    return records


def summarize(records: list[Record]) -> Totals:
    totals = Totals(calls=len(records))
    for entry in records:
        totals.tokens += entry.tokens
        totals.spent += entry.cost_usd
        if entry.free:
            totals.free_calls += 1
            totals.free_tokens += entry.tokens
            totals.saved += entry.would_cost_usd
        else:
            totals.paid_tokens += entry.tokens
        if entry.estimated_tokens:
            totals.estimated_calls += 1
        totals.by_slot[entry.slot] = totals.by_slot.get(entry.slot, 0) + 1
    return totals


# ----------------------------------------------------------------------
# Баланс
# ----------------------------------------------------------------------


def deepseek_key() -> str:
    """Ключ DeepSeek из конфига FCC. В вывод не попадает."""

    if not FCC_ENV.is_file():
        return ""
    for line in FCC_ENV.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "DEEPSEEK_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""


def balance(timeout: float = 45.0) -> dict[str, object] | None:
    """Спросить у DeepSeek фактический баланс.

    Запрос идёт напрямую, без прокси: DeepSeek через корпоративный прокси не
    работает, а ``trust_env`` в urllib не отключается — поэтому переменные
    окружения на время запроса убираются вручную.
    """

    key = deepseek_key()
    if not key:
        return None
    request = urllib.request.Request(
        BALANCE_URL, headers={"Authorization": f"Bearer {key}"}
    )
    saved = {name: os.environ.pop(name) for name in list(os.environ) if name.lower().endswith("_proxy")}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    finally:
        os.environ.update(saved)

    info = (payload.get("balance_infos") or [{}])[0]
    try:
        total = float(info.get("total_balance", 0))
    except (TypeError, ValueError):
        total = 0.0
    return {
        "available": bool(payload.get("is_available")),
        "currency": info.get("currency", "USD"),
        "total": total,
        "granted": float(info.get("granted_balance", 0) or 0),
        "topped_up": float(info.get("topped_up_balance", 0) or 0),
    }


def calibrate(records: list[Record], spent_money: float) -> float | None:
    """Вывести фактическую ставку из потраченных денег и учтённых токенов.

    Возвращает долларов за миллион токенов или None, если данных мало. Это
    единственный способ получить настоящую ставку, не выдумывая её: баланс
    знает провайдер, токены знаем мы. Оговорка: в разницу баланса попадает и
    всё остальное, что ходило в DeepSeek за тот же период, — включая сессии
    Claude Code. Поэтому калибровка верна, только если учёт ведётся с
    известного момента и посторонних трат между замерами не было.
    """

    paid = [entry for entry in records if not entry.free]
    tokens = sum(entry.tokens for entry in paid)
    if not paid or tokens < 1000 or spent_money <= 0:
        return None
    return spent_money / tokens * 1_000_000


# ----------------------------------------------------------------------
# Снимки баланса
# ----------------------------------------------------------------------

BALANCE_FILE = Path.home() / ".local" / "share" / "ai-farm" / "balance.jsonl"


def save_balance_snapshot(info: dict[str, object]) -> None:
    """Записать текущий баланс в историю.

    История нужна, чтобы измерить расход деньгами, а не ставками: баланс
    до и после — это факт, который сообщил провайдер.
    """

    try:
        BALANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with BALANCE_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": time.time(), **info}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def load_balance_snapshots() -> list[dict[str, object]]:
    if not BALANCE_FILE.is_file():
        return []
    snapshots: list[dict[str, object]] = []
    for line in BALANCE_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            snapshots.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return snapshots


def spent_between(snapshots: list[dict[str, object]], since: float) -> float | None:
    """Сколько денег ушло с баланса между первым снимком после ``since`` и последним."""

    window = [s for s in snapshots if float(s.get("ts", 0)) >= since]
    if len(window) < 2:
        return None
    first = float(window[0].get("total", 0))
    last = float(window[-1].get("total", 0))
    if first <= 0:
        return None
    return max(0.0, first - last)


def ensure_prices_file() -> Path:
    """Создать файл ставок с плейсхолдером, если его нет.

    Файл создаётся с ``verified: false`` намеренно: пока человек не сверил
    числа с прайсом провайдера, экономия считается оценкой.
    """

    if PRICES_FILE.is_file():
        return PRICES_FILE
    PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"verified": False, "comment": "ставки за миллион токенов в USD; сверь с прайсом провайдера и поставь verified: true"}
    payload.update(DEFAULT_PRICES)
    PRICES_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return PRICES_FILE


# ----------------------------------------------------------------------
# Расход самой сессии Claude Code
# ----------------------------------------------------------------------

TRANSCRIPTS = Path.home() / ".claude" / "projects"


@dataclass
class SessionUsage:
    """Расход сессий Claude Code — того канала, что идёт в DeepSeek напрямую.

    Учёт фермы видит только вызовы самой фермы. Но сессия — тот же платный
    DeepSeek, и тратит она на порядки больше: замер 19.09.2026 дал $0.000089
    у фермы против $0.52 у сессии за полчаса. Пока этот расход не виден в
    консоли, вопрос «почему тратится много» остаётся без ответа.

    Числа здесь точные, а не оценочные: клиент сам пишет расход по каждому
    запросу в журнал сессии. Ключевая величина — перечитанный контекст:
    он и есть основная статья расхода, потому что каждый запрос заново
    отправляет весь разговор.
    """

    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    cache_write: int = 0
    models: dict[str, int] = field(default_factory=dict)
    files: int = 0

    @property
    def billed_tokens(self) -> int:
        """Все токены, за которые платим: вход, выход и перечитанный контекст."""

        return self.tokens_in + self.tokens_out + self.cache_read

    def context_per_request(self) -> int:
        """Сколько контекста в среднем тащит один запрос."""

        return int(self.cache_read / self.requests) if self.requests else 0

    def lines(self) -> list[str]:
        if not self.requests:
            return ["  записей о расходе не найдено"]
        ratio = self.cache_read / max(1, self.tokens_in + self.tokens_out)
        return [
            f"  запросов:            {self.requests:,}".replace(",", " "),
            f"  вход:                {self.tokens_in:,}".replace(",", " "),
            f"  выход:               {self.tokens_out:,}".replace(",", " "),
            f"  перечитано контекста:{self.cache_read:>13,}".replace(",", " "),
            f"  средний контекст на запрос: {self.context_per_request():,}".replace(",", " ")
            + " токенов",
            f"  контекста больше, чем нового текста, в {ratio:.0f} раз" if ratio >= 1 else "",
        ]


def session_usage(root: Path = TRANSCRIPTS) -> SessionUsage:
    """Собрать расход по всем журналам сессий Claude Code.

    Журналы лежат по сессиям и растут в течение работы. Читаем все: интересует
    суммарная картина, а не одна сессия. Битые строки пропускаем — журнал
    пишется на ходу, и последняя строка может быть недописанной.
    """

    usage = SessionUsage()
    if not root.is_dir():
        return usage
    for path in root.rglob("*.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        seen = False
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = (record.get("message") or {}).get("usage")
            if not data:
                continue
            seen = True
            usage.requests += 1
            usage.tokens_in += int(data.get("input_tokens") or 0)
            usage.tokens_out += int(data.get("output_tokens") or 0)
            usage.cache_read += int(data.get("cache_read_input_tokens") or 0)
            usage.cache_write += int(data.get("cache_creation_input_tokens") or 0)
            model = (record.get("message") or {}).get("model") or "?"
            usage.models[model] = usage.models.get(model, 0) + 1
        if seen:
            usage.files += 1
    return usage
