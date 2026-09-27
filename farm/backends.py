"""Бэкенды моделей для фермы: локальный Ollama, бесплатный Zen, DeepSeek.

Каждый бэкенд умеет одно — получить ответ на текстовый запрос. Выбор маршрута
живёт в ``models.py``, а не здесь: бэкенд не решает, когда его позовут.

Общее правило для всех: ``trust_env=False``. Иначе httpx подхватит
HTTP_PROXY из окружения и локальные вызовы (Ollama, шлюз FCC) уйдут в
корпоративный прокси — запрос вернётся с 502 и будет выглядеть как поломка
модели, а не сети.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

log = logging.getLogger("farm.backends")

OLLAMA_URL = "http://127.0.0.1:11434"
FCC_URL = "http://127.0.0.1:8082"
OPENCODE = Path.home() / ".opencode" / "bin" / "opencode"
PROFILE_BASE = Path.home() / ".local" / "share" / "oc-free"
PROXY_SCRIPT = Path("/etc/profile.d/99-proxy.sh")

# Источники кредов для аккаунтов Zen. Ключи не хранятся в коде — берём файл.
ZEN_AUTH = {
    "A": Path.home() / ".local" / "share" / "opencode" / "auth.json",
    "B": Path.home() / ".local" / "share" / "oc2" / "opencode" / "auth.json",
}


class BackendError(RuntimeError):
    """Модель не ответила. Ферма должна пережить это, а не упасть."""


def _kill_tree(proc: subprocess.Popen, *, drain: bool = True) -> None:
    """Убить процесс вместе со всеми потомками.

    Проверено 19.09.2026: ``subprocess.run(timeout=...)`` убивает только
    внешний ``script``, а вложенный ``opencode run`` остаётся жить и жрать
    память. За час накопилось 18 таких сирот на 7 ГБ, и машина встала.
    Поэтому клиент запускается в отдельной группе процессов
    (``start_new_session=True``), а таймаут убивает группу целиком.

    ``drain=False`` — для вызова из сторожа: ``communicate`` дочитывает
    каналы и закрывает их, а главный поток в это же время читает те же
    дескрипторы и падает с ``OSError: [Errno 9] Bad file descriptor``
    (проверено 19.09.2026, уронило прогон «Таксометра»). Из чужого потока
    процесс только убивают — дочитать даст тот, кто его запускал.
    """

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        proc.kill()
    if not drain:
        return
    try:
        proc.communicate(timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _run_capped(
    cmd: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    watchdog=None,
) -> tuple[int, str, str]:
    """Запустить команду с потолком времени и без сирот.

    Возвращает ``(код, stdout, stderr)``. Отдельная обёртка нужна потому,
    что ``subprocess.run`` не даёт ручки на процесс после таймаута, а без
    неё потомков не убить.

    ``watchdog`` — необязательная функция ``(proc, stop_event)``: она
    работает, пока команда считает, и может убить её раньше потолка.
    """

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Группу запоминаем сразу: к моменту уборки процесс уже собран
    # ``communicate``, и ``getpgid`` по нему не отвечает.
    pgid = proc.pid
    stop = threading.Event()
    thread = None
    if watchdog is not None:
        thread = threading.Thread(target=watchdog, args=(proc, stop), daemon=True)
        thread.start()
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        raise
    except OSError as exc:
        # Дескрипторы канала закрылись под руками: процесс уже мёртв, но
        # дочитать нечего. Это отказ вызова, а не поломка фермы — пусть
        # подзадача уйдёт на переделку, как при любом другом отказе модели
        # (проверено 19.09.2026: без этого падал весь прогон целиком).
        _kill_tree(proc, drain=False)
        raise BackendError(f"канал вызова оборвался: {exc}") from exc
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=2.0)
        # Внешний ``script`` умеет выйти раньше своего потомка, и тогда
        # ``opencode run`` живёт дальше сам по себе (родитель — init, 470 МБ
        # на процесс). Группа у них общая, поэтому после каждого вызова
        # добиваем её целиком: вывод уже прочитан, терять нечего.
        # Проверено 19.09.2026 — без этого за час набежали гигабайты сирот.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
    return proc.returncode, out or "", err or ""


@dataclass
class Reply:
    """Ответ модели вместе с расходом токенов.

    Токены нужны учёту: без них не видно ни фактической траты, ни экономии.
    У DeepSeek и Ollama счётчики настоящие, у Zen — нет: CLI OpenCode их не
    отдаёт, поэтому там работает оценка по символам, и она помечается
    ``estimated_tokens``. Смешивать одно с другим без пометки нельзя, иначе
    выдуманное число попадёт в отчёт об экономии как факт.
    """

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_tokens: bool = False


def estimate_tokens(text: str) -> int:
    """Оценка числа токенов по длине текста.

    Множитель 1/4 — грубая, но устойчивая прикидка для смеси русского и
    латиницы; для кода он ближе к 1/3. Точность здесь не критична: значение
    всё равно помечено как оценка.
    """

    return max(1, len(text) // 4)


class Backend(Protocol):
    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> Reply: ...

    def available(self) -> bool:
        """Готов ли бэкенд отвечать прямо сейчас.

        Слот, поймавший лимит, честно говорит об этом заранее: иначе ферма
        тратит на него ровно столько же времени, сколько на живого.
        """
        ...


def _proxy_env() -> dict[str, str]:
    """Окружение с прокси.

    Фри-тир Zen отдавался только через прокси: напрямую провайдер отвечал
    Forbidden / «not available in your country», и это регион-блок, а не
    запрет фри-тира. Проверено 19.09.2026: напрямую фри-модели теперь тоже
    отвечают, а прокси жив (на запрос без учётных данных он отвечает 407,
    с ними — 200), поэтому оставляем как есть: канал рабочий, а лишняя
    перестройка маршрута ничего не даёт. В неинтерактивной оболочке
    переменных нет, поэтому читаем системный файл.
    """

    if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        return {}
    if not PROXY_SCRIPT.is_file():
        return {}
    env: dict[str, str] = {}
    for line in PROXY_SCRIPT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        name, _, raw = line[len("export "):].partition("=")
        name = name.strip()
        if name.lower() in {"http_proxy", "https_proxy", "no_proxy"}:
            env[name] = raw.strip().strip('"').strip("'")
    return env


@dataclass
class OllamaBackend:
    """Локальная модель. Бесплатно, офлайн, без лимитов."""

    model: str
    url: str = OLLAMA_URL
    timeout: float = 600.0

    def available(self) -> bool:
        """Локальные модели не отказывают по квоте — они всегда в строю."""
        return True

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> Reply:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        try:
            response = httpx.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=self.timeout,
                trust_env=False,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama/{self.model}: {exc}") from exc
        payload = response.json()
        # content приходит null у thinking-моделей, когда весь бюджет ушёл на
        # размышление: `or ""` здесь вместо значения по умолчанию в .get(),
        # иначе на None падает .strip().
        text = ((payload.get("message") or {}).get("content") or "").strip()
        if not text:
            # Пустой ответ — не ответ. Такое бывает, когда модель упирается в
            # num_predict на служебном тексте; ферма должна уйти на другой слот,
            # а не записать пустоту как результат.
            raise BackendError(
                f"ollama/{self.model}: пустой ответ "
                f"(done_reason={payload.get('done_reason')}, лимит {max_tokens} токенов)"
            )
        return Reply(
            text=text,
            # Ollama считает токены сама и отдаёт их в ответе — оценка не нужна.
            tokens_in=int(payload.get("prompt_eval_count") or 0),
            tokens_out=int(payload.get("eval_count") or 0),
        )


# Лимиты у Zen — на аккаунт, а не на модель. Проверено 19.09.2026: аккаунт A
# начал отвечать «Rate limit exceeded» в 11:24 и держал отказ больше двух
# часов, отказывая сразу по всем семи моделям, — а аккаунт B в это же время
# отвечал. Без этой памяти ферма по очереди стучится в семь мёртвых слотов, и
# каждый отказ стоит полного ожидания. Поэтому аккаунт, поймавший лимит,
# уходит в отстой, а задача идёт к живому.
COOLDOWN_SECONDS = 1800.0

_COOLDOWN: dict[str, float] = {}
_COOLDOWN_LOCK = threading.Lock()

# Признаки в журнале профиля: дело в квоте, а не в сети или в самой задаче.
_LIMIT_MARKERS = ("rate limit", "insufficient balance", "no payment method", "quota")


def cooling(account: str) -> bool:
    """Стоит ли аккаунт в отстое. Истёкший отстой снимается сам."""
    with _COOLDOWN_LOCK:
        until = _COOLDOWN.get(account, 0.0)
        if not until:
            return False
        if until <= time.time():
            del _COOLDOWN[account]
            return False
        return True


def cooldown_left(account: str) -> float:
    """Сколько секунд аккаунт ещё в отстое."""
    with _COOLDOWN_LOCK:
        return max(0.0, _COOLDOWN.get(account, 0.0) - time.time())


def _put_on_cooldown(account: str) -> None:
    with _COOLDOWN_LOCK:
        _COOLDOWN[account] = time.time() + COOLDOWN_SECONDS


def _limit_in_log(account: str, model: str, lines: int = 80) -> str:
    """Найти в журнале профиля причину отказа.

    CLI не отдаёт ошибку наружу: печатает баннер и остаётся висеть, а
    настоящая причина лежит в его журнале. Читаем хвост и ищем признаки
    исчерпанной квоты.
    """

    log_path = PROFILE_BASE / account / model / "opencode" / "log" / "opencode.log"
    if not log_path.is_file():
        return ""
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return ""
    for line in reversed(tail):
        lowered = line.lower()
        for marker in _LIMIT_MARKERS:
            if marker in lowered:
                return marker
    return ""


@dataclass
class ZenBackend:
    """Бесплатная модель OpenCode Zen через клиент OpenCode.

    Фри-тир отдаётся только внутри самого OpenCode, поэтому здесь запускается
    его CLI, а не HTTP-запрос к API. У каждой пары «аккаунт x модель» свой
    каталог состояния: иначе параллельные воркеры делят одну историю сессий.
    """

    model: str
    account: str = "A"
    # 420 с: большой HTML пишется минутами, и низкий потолок рубит честную
    # работу (проверено 19.09.2026 — при 150 с воркеры сменялись каждые
    # полторы минуты и файл не рождался вовсе). Мёртвый аккаунт ловит
    # сторож по журналу, а не этот потолок, поэтому ждать его не приходится.
    timeout: float = 420.0

    def _profile_dir(self) -> Path:
        source = ZEN_AUTH.get(self.account)
        if source is None:
            raise BackendError(f"неизвестный аккаунт Zen: {self.account}")
        if not source.is_file():
            raise BackendError(f"нет файла кредов для аккаунта {self.account}: {source}")
        target = PROFILE_BASE / self.account / self.model
        (target / "opencode").mkdir(parents=True, exist_ok=True)
        creds = target / "opencode" / "auth.json"
        if not creds.exists():
            creds.write_bytes(source.read_bytes())
            creds.chmod(0o600)
        return target

    def available(self) -> bool:
        return not cooling(self.account)

    def _timeout_reason(self) -> str:
        """Объяснить таймаут: исчерпан лимит аккаунта или просто долгий ответ.

        Разница принципиальная. При лимите в отстой уходит весь аккаунт, и
        остальные его модели даже не пробуются — это экономит минуты на
        каждой подзадаче. При обычном таймауте виноват один слот, и
        наказывать за это аккаунт не за что.
        """

        marker = _limit_in_log(self.account, self.model)
        if marker:
            _put_on_cooldown(self.account)
            return (
                f"zen/{self.model}: аккаунт {self.account} исчерпал лимит "
                f"({marker}) — в отстое {COOLDOWN_SECONDS / 60:.0f} мин"
            )
        return f"zen/{self.model}: таймаут {self.timeout:.0f} с"

    def _watch_limit(self, proc: subprocess.Popen, stop: threading.Event) -> None:
        """Сторож: ловить исчерпанный лимит аккаунта, пока модель думает.

        При лимите CLI OpenCode не выходит и не отвечает — печатает баннер и
        молчит. Признак виден в журнале профиля почти сразу, поэтому ждать
        потолка незачем: сторож снимает аккаунт с очереди и убивает вызов,
        освобождая слот остальным подзадачам.
        """

        while not stop.wait(5.0):
            if proc.poll() is not None:
                return
            marker = _limit_in_log(self.account, self.model)
            if marker:
                _put_on_cooldown(self.account)
                log.info("zen/%s: аккаунт %s снят с очереди (%s)",
                         self.model, self.account, marker)
                _kill_tree(proc, drain=False)
                return

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> Reply:
        if not OPENCODE.is_file():
            raise BackendError(f"клиент OpenCode не найден: {OPENCODE}")
        if not self.available():
            raise BackendError(
                f"zen/{self.account}: аккаунт в отстое ещё "
                f"{cooldown_left(self.account):.0f} с — лимит исчерпан"
            )
        text = f"{NO_TOOLS}\n\n{system}\n\n{prompt}" if system else f"{NO_TOOLS}\n\n{prompt}"
        inner = f"{OPENCODE} run -m opencode/{self.model} {shlex.quote(text)}"
        # Без pty этот CLI молчит, а `script -qec` даёт ему терминал.
        # stdin закрываем: иначе script съедает вход вызывающего процесса.
        cmd = ["script", "-qec", inner, "/dev/null"]
        env = {**os.environ, **_proxy_env(), "XDG_DATA_HOME": str(self._profile_dir())}
        try:
            code, stdout, stderr = _run_capped(
                cmd, timeout=self.timeout, env=env, watchdog=self._watch_limit
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(self._timeout_reason()) from exc
        if code != 0:
            # Вызов мог убить сторож, и тогда код — это просто «убит»;
            # настоящая причина в отстое аккаунта, о ней и сообщаем.
            if cooling(self.account):
                raise BackendError(
                    f"zen/{self.model}: аккаунт {self.account} исчерпал лимит — "
                    f"в отстое ещё {cooldown_left(self.account) / 60:.0f} мин"
                )
            detail = (stderr or stdout).strip()[-400:]
            raise BackendError(f"zen/{self.model}: код {code}: {detail}")
        answer = _clean_cli_output(stdout)
        if not answer:
            # CLI завершился с нулём, но ничего не сказал: считать это ответом
            # нельзя, иначе пустота уедет в результат.
            raise BackendError(f"zen/{self.model}: клиент завершился молча, ответа нет")
        # CLI OpenCode не сообщает расход токенов, поэтому здесь только оценка
        # по длине текста — и она честно помечена как оценка.
        return Reply(
            text=answer,
            tokens_in=estimate_tokens(system) + estimate_tokens(prompt),
            tokens_out=estimate_tokens(answer),
            estimated_tokens=True,
        )


def _strip_ansi(text: str) -> str:
    """Убрать escape-последовательности, которые CLI рисует для терминала."""

    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\x1b":
            index += 1
            if index < len(text) and text[index] == "[":
                index += 1
                while index < len(text) and not text[index].isalpha():
                    index += 1
            index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


# Два запрета, без которых ферма получает не ответ, а отчёт о работе, которой
# не просила. Проверено на живом прогоне 19.09.2026: бесплатная модель через
# клиент OpenCode уходила писать файлы — сначала встроенным Write, потом
# командой оболочки, потом инструментом MCP server-filesystem. В ответ
# попадал след вызова, контролёр его законно браковал, и обе попытки
# подзадачи уходили в брак. Со стороны это выглядит как «ферма ничего не
# делает», хотя диспетчер отработал и подзадачи раздал.
#
# Конфигом это не лечится: правка разделов agent и permission в конфиге
# OpenCode ломает распознавание фри-тира («free tier can only be used from
# within OpenCode»), а ключи tools и mcp он принимает, но не применяет.
# Поэтому уговариваем словами и чистим вывод.
NO_TOOLS = (
    "У тебя нет инструментов: ни файлов, ни оболочки, ни поиска. Не вызывай "
    "их и не описывай вызовы. Ничего не сохраняй и не запускай. Ответь "
    "обычным текстом — это единственный принимаемый формат."
)

# Служебная шапка TUI: «> build · ling-3.0-flash-fin-free». Снять её надо
# обязательно: она уезжает в ответ, а оттуда — в сводку диспетчера, где
# тратит токены и выглядит как часть результата.
_CLI_BANNER = re.compile(r"^\s*>\s*[\w-]+\s*·.*$")

# Следы вызовов инструментов. Проверка идёт по началу строки, а не поиском
# подстроки: иначе пострадал бы обычный текст, где встречается слово «Write»
# или знак доллара.
_TOOL_NOISE = (
    re.compile(r"^\s*⚙\s"),                      # вызов инструмента MCP
    re.compile(r"^\s*[←→]\s+\w"),                 # встроенный инструмент
    re.compile(r"^\s*\$\s+\S"),                   # команда оболочки
    re.compile(r"^\s*Wrote file successfully\.?\s*$"),
    re.compile(r"^\s*\d+\s+\w+\s+(written|updated)\s*$"),
)


def _clean_cli_output(text: str) -> str:
    """Убрать из вывода CLI служебные строки интерфейса и следы инструментов.

    Пустые строки в начале снимаются вместе с шапкой, и в одном цикле с ней:
    терминал печатает шапку не всегда первой строкой, и проверка «первая
    строка — шапка» на выводе с ведущим переводом строки не срабатывает.

    Следы вызовов вычищаются по всему тексту, а не только в начале: модель
    успевает написать абзац, потом сорваться в инструмент, потом попрощаться.
    """

    lines = _strip_ansi(text).splitlines()
    while lines and (not lines[0].strip() or _CLI_BANNER.match(lines[0])):
        lines.pop(0)
    lines = [line for line in lines if not any(p.match(line) for p in _TOOL_NOISE)]
    return "\n".join(lines).strip()


NVIDIA_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_KEY_FILE = Path.home() / ".config" / "ai-farm" / "nvidia.key"


def nvidia_key() -> str:
    """Ключ NVIDIA. Сначала окружение, потом файл — как у остальных кредов."""

    from_env = os.environ.get("NVIDIA_API_KEY", "").strip()
    if from_env:
        return from_env
    if NVIDIA_KEY_FILE.is_file():
        return NVIDIA_KEY_FILE.read_text(encoding="utf-8").strip()
    return ""


def _proxy_url() -> str | None:
    """Адрес прокси строкой — httpx хочет его параметром, а не в окружении."""

    env = _proxy_env() or os.environ
    for name in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        value = env.get(name)
        if value:
            return value
    return None


@dataclass
class NvidiaBackend:
    """Бесплатные модели NVIDIA NIM через OpenAI-совместимый API.

    Фри-тир здесь не про кредиты, а про скорость: около 40 запросов в минуту
    на ключ, без счётчика токенов. Для фермы это ценнее кредитов — узкое
    место у нас не объём, а число обращений.

    Из этой сети NVIDIA доступна только через прокси: напрямую
    ``integrate.api.nvidia.com`` отвечает 451 (гео-блок) — проверено 27.09.2026
    на обоих эндпоинтах, и ``/v1/models``, и ``/v1/chat/completions``. Прокси
    здесь не подстраховка, а обязательное условие: без него каждый вызов
    вернёт 451, а вызывающий код примет это за ошибку модели.
    """

    model: str
    url: str = NVIDIA_URL
    timeout: float = 150.0

    def available(self) -> bool:
        """Лимит здесь поминутный и приходит ответом 429, а не отказом ключа.

        Держать модель в отстое на полчаса из-за минутной вспышки смысла нет:
        следующий вызов, скорее всего, пройдёт.
        """
        return bool(nvidia_key())

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> Reply:
        key = nvidia_key()
        if not key:
            raise BackendError(
                f"нет ключа NVIDIA: положите его в {NVIDIA_KEY_FILE} "
                "или в переменную NVIDIA_API_KEY"
            )
        proxy = _proxy_url()
        if not proxy:
            raise BackendError(
                "nvidia: не найден прокси, а без него NVIDIA отвечает 451 (гео-блок)"
            )
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        try:
            response = httpx.post(
                f"{self.url}/chat/completions",
                json={"model": self.model, "messages": messages, "max_tokens": max_tokens},
                headers={"Authorization": f"Bearer {key}"},
                timeout=self.timeout,
                proxy=proxy,
                trust_env=False,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            hint = {
                401: "ключ не принят",
                403: "ключ не принят или модель закрыта на фри-тире",
                410: "модель снята с эксплуатации — замените её в NVIDIA_MODELS",
                429: "лимит запросов исчерпан",
                451: "гео-блок: нужен прокси",
            }.get(code, "")
            raise BackendError(
                f"nvidia/{self.model}: HTTP {code}{' — ' + hint if hint else ''}"
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"nvidia/{self.model}: {exc}") from exc
        data = response.json()
        choices = data.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            # У reasoning-моделей (kimi-k3, glm-5.3) content приходит null, если
            # бюджет токенов целиком ушёл на размышление. Брать вместо ответа
            # reasoning_content нельзя — в результат попадёт черновик мыслей.
            # Пустой ответ честнее: слот признаётся неудачным, и ферма берёт
            # следующую модель.
            text = (message.get("content") or "").strip()
        if not text:
            # Пустой ответ — не ответ: иначе ферма запишет прогон как удачный.
            raise BackendError(
                f"nvidia/{self.model}: пустой ответ "
                f"(finish_reason={(choices[0].get('finish_reason') if choices else None)})"
            )
        usage = data.get("usage") or {}
        return Reply(
            text=text,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
        )


@dataclass
class FccBackend:
    """DeepSeek через шлюз FCC. Платный, поэтому только для сложного."""

    model: str = "claude-sonnet-4-5"
    url: str = FCC_URL
    timeout: float = 600.0

    def available(self) -> bool:
        """Платный тир лимитов фермы не знает — он про деньги, не про квоту."""
        return True

    def _token(self) -> str:
        env_file = Path.home() / ".fcc" / ".env"
        if not env_file.is_file():
            raise BackendError(f"нет конфига FCC: {env_file}")
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ANTHROPIC_AUTH_TOKEN":
                return value.strip().strip('"').strip("'")
        raise BackendError("в ~/.fcc/.env нет ANTHROPIC_AUTH_TOKEN")

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> Reply:
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        try:
            response = httpx.post(
                f"{self.url}/v1/messages",
                json=payload,
                headers={"Authorization": f"Bearer {self._token()}"},
                timeout=self.timeout,
                trust_env=False,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"fcc/{self.model}: {exc}") from exc
        data = response.json()
        blocks = data.get("content") or []
        usage = data.get("usage") or {}
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        if not text:
            # Пустой ответ — это не ответ. Типичная причина: модель рассуждает
            # (блок thinking) и тратит на размышления весь max_tokens, так и не
            # дойдя до текста. Молча вернуть пустоту нельзя: ферма приняла бы
            # её за результат и записала бы прогон как успешный.
            kinds = ",".join(sorted({b.get("type", "?") for b in blocks})) or "нет блоков"
            raise BackendError(
                f"fcc/{self.model}: пустой ответ (блоки: {kinds}, "
                f"stop_reason={data.get('stop_reason')}) — вероятно, весь бюджет ушёл в размышления"
            )
        return Reply(
            text=text,
            # Это настоящий расход, за который списывают деньги, — основа учёта.
            tokens_in=int(usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
        )
