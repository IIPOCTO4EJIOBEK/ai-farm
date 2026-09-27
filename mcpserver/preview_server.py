"""MCP-сервер живого предпросмотра страниц.

Задача — как в онлайн-генераторах сайтов: модель публикует HTML, страница
сразу открывается в браузере, а каждая правка появляется на экране без
ручного обновления. Отсюда разделение на две части:

* этот файл — инструменты MCP: записать страницу, открыть её, перечислить,
  заменить фрагмент;
* ``preview_http`` — HTTP-сервер, который раздаёт страницы и держит канал
  событий. Он живёт ОТДЕЛЬНЫМ процессом.

Разделение не косметическое. Клиент MCP перезапускает свой сервер при
обновлении конфига и при переподключении; если страницы раздавались бы из
этого же процесса, предпросмотр отваливался бы вместе с ним — в браузере
осталась бы мёртвая вкладка, а следующая публикация поднимала бы сервер на
другом порту, в другом процессе, с другим списком подписчиков. Отдельный
процесс переживает и перезапуск, и закрытие сессии.

Связи между частями нет: инструменты пишут файлы, а сервер сам замечает
правку по времени изменения файлов и рассылает перезагрузку. Файлы и есть
канал, поэтому не нужны ни порты управления, ни сокеты, ни общее состояние.

Запускается по stdio. Печатать в stdout нельзя — это ломает протокол MCP;
все служебные сообщения идут в stderr.

Почему это не «просто открыть файл в браузере»: file:// не умеет ни
перезагрузку по событию, ни список страниц, ни адрес, которым можно
поделиться. Здесь у каждой страницы постоянный адрес, и правка меняет то,
что уже открыто.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx

try:  # MCP SDK 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

mcp = MCPServer("preview")

PAGES_DIR = Path.home() / ".local" / "share" / "ai-preview"
DEFAULT_PORT = 8791
MAX_PAGE_BYTES = 4_000_000
HERE = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
LOG_FILE = Path.home() / ".local" / "share" / "ai-preview" / "preview.log"


def _safe_name(name: str) -> str:
    """Привести имя страницы к безопасному виду.

    Имя приходит от модели, а попадает в путь на диске и в адрес. Поэтому
    разрешены только буквы, цифры, дефис и подчёркивание: иначе «../» в имени
    вывел бы запись за пределы каталога страниц.
    """

    cleaned = "".join(char for char in name if char.isalnum() or char in "-_")
    return cleaned or "index"


def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _alive(port: int) -> bool:
    """Признак живого сервера предпросмотра на порту.

    Спрашиваем список страниц, а не просто открываем соединение: так
    проверяется, что на порту именно наш сервер, а не чужая программа,
    случайно занявшая тот же номер. trust_env=False — иначе локальный запрос
    уйдёт в корпоративный прокси.
    """

    try:
        response = httpx.get(f"{_base_url(port)}/__pages", timeout=1.0, trust_env=False)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _spawn(port: int) -> None:
    """Запустить сервер предпросмотра отдельным процессом.

    start_new_session отвязывает его от нашей группы процессов: MCP-клиент
    завершает свой сервер вместе с детьми, и без этого предпросмотр умирал бы
    при каждом перезапуске — ровно та поломка, из-за которой этот файл
    разделён на две части.
    """

    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "ab", buffering=0)  # noqa: SIM115 — живёт вместе с процессом
    subprocess.Popen(
        [str(PYTHON), "-m", "mcpserver.preview_http", "--port", str(port)],
        cwd=str(HERE.parent),
        env={**os.environ, "PYTHONPATH": str(HERE.parent)},
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _ensure_server() -> tuple[int, bool]:
    """Поднять сервер предпросмотра, если его нет. Возвращает порт и признак запуска.

    Сначала проверяем, не отвечает ли сервер на прежнем порту: при живом
    сервере поднимать второй нельзя — открытые вкладки подписаны на первый,
    и новая публикация ушла бы в никуда.
    """

    for port in range(DEFAULT_PORT, DEFAULT_PORT + 20):
        if _alive(port):
            return port, False

    for port in range(DEFAULT_PORT, DEFAULT_PORT + 20):
        _spawn(port)
        # Процессу нужно время на импорт и открытие сокета; ждём коротко и
        # проверяем, а не спим фиксированную паузу.
        for _ in range(40):
            time.sleep(0.1)
            if _alive(port):
                return port, True
    raise RuntimeError("не удалось запустить сервер предпросмотра")


def _pages(port: int) -> list[dict[str, object]]:
    try:
        response = httpx.get(f"{_base_url(port)}/__pages", timeout=2.0, trust_env=False)
        return json.loads(response.text)
    except (httpx.HTTPError, json.JSONDecodeError):
        return []


def _write_page(name: str, html: str) -> tuple[str, Path]:
    if len(html.encode("utf-8")) > MAX_PAGE_BYTES:
        raise ValueError(
            f"страница больше {MAX_PAGE_BYTES // 1_000_000} МБ — похоже на ошибку генерации"
        )
    safe = _safe_name(name)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    page = PAGES_DIR / f"{safe}.html"
    page.write_text(html, encoding="utf-8")
    return safe, page


def _publish(name: str, html: str, *, open_browser: bool) -> str:
    port, started = _ensure_server()
    safe, page = _write_page(name, html)
    url = f"{_base_url(port)}/{safe}"
    lines = [
        f"опубликовано: {safe}",
        f"адрес:        {url}",
        f"файл:         {page}",
        f"размер:       {page.stat().st_size} байт",
        f"сервер:       {'запущен сейчас' if started else 'уже работал'} (порт {port})",
    ]
    if open_browser:
        lines.append("браузер:      " + _open(url))
    lines.append("живое обновление: включено — правка этой страницы появится в браузере сама")
    return "\n".join(lines)


def _open(url: str) -> str:
    """Открыть адрес в браузере.

    webbrowser в неинтерактивной сессии может не найти графическую среду,
    поэтому при неудаче пробуем xdg-open с явным DISPLAY — иначе публикация
    выглядела бы успешной, а окно не появилось бы.
    """

    try:
        if webbrowser.open(url):
            return "открыт"
    except Exception:  # noqa: BLE001 — причин много, важен итог
        pass
    env = {**os.environ, "DISPLAY": os.environ.get("DISPLAY") or ":0"}
    for opener in ("xdg-open", "firefox"):
        try:
            subprocess.Popen(
                [opener, url],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            return f"открыт через {opener}"
        except FileNotFoundError:
            continue
    return f"открыть не удалось — откройте вручную: {url}"


# ----------------------------------------------------------------------
# Инструменты
# ----------------------------------------------------------------------


@mcp.tool()
def preview_publish(name: str, html: str, open_browser: bool = True) -> str:
    """Опубликовать страницу и показать её в браузере.

    name — короткое имя без пути (латиница, цифры, дефис), например
    «dilizhans». html — полный текст страницы. Адрес страницы постоянный:
    повторная публикация с тем же именем обновляет открытую вкладку сама,
    перезагружать вручную не нужно.
    """

    try:
        return _publish(name, html, open_browser=open_browser)
    except (ValueError, OSError, RuntimeError) as exc:
        return f"не опубликовано: {exc}"


@mcp.tool()
def preview_update(name: str, html: str) -> str:
    """Заменить содержимое уже опубликованной страницы.

    Отличие от preview_publish только в намерении: браузер не открывается
    заново, а открытая вкладка перезагружается сама.
    """

    try:
        return _publish(name, html, open_browser=False)
    except (ValueError, OSError, RuntimeError) as exc:
        return f"не обновлено: {exc}"


@mcp.tool()
def preview_replace(name: str, find: str, replace: str, count: int = 0) -> str:
    """Заменить фрагмент в опубликованной странице, не пересылая её целиком.

    Удобно для мелких правок — поменять цвет, текст, заголовок, — когда
    гонять весь HTML через контекст дорого. count=0 заменяет все вхождения,
    иначе — не больше указанного числа. Открытая страница обновится сама.
    """

    safe = _safe_name(name)
    page = PAGES_DIR / f"{safe}.html"
    if not page.is_file():
        return f"нет страницы {safe}: сначала preview_publish"
    html = page.read_text(encoding="utf-8")
    if find not in html:
        return f"фрагмент не найден в {safe} — замена не сделана"
    occurrences = html.count(find)
    done = html.replace(find, replace) if count <= 0 else html.replace(find, replace, count)
    try:
        port, _ = _ensure_server()
        _write_page(safe, done)
    except (ValueError, OSError, RuntimeError) as exc:
        return f"не обновлено: {exc}"
    replaced = occurrences if count <= 0 else min(occurrences, count)
    return (
        f"заменено вхождений: {replaced}\n"
        f"адрес: {_base_url(port)}/{safe}\nстраница обновлена"
    )


@mcp.tool()
def preview_list() -> str:
    """Показать опубликованные страницы: имена, размеры, время изменения."""

    port, _ = _ensure_server()
    pages = _pages(port)
    if not pages:
        return f"страниц пока нет\nкаталог: {PAGES_DIR}\nсервер:  {_base_url(port)}"
    lines = [f"каталог: {PAGES_DIR}", f"сервер:  {_base_url(port)}", "",
             f"{'имя':<20} {'размер':>9}  изменена"]
    for page in pages:
        lines.append(f"{page['name']:<20} {page['bytes']:>9}  {page['modified']}")
    return "\n".join(lines)


@mcp.tool()
def preview_status() -> str:
    """Состояние предпросмотра: сервер, порт, текущая страница, число страниц."""

    for port in range(DEFAULT_PORT, DEFAULT_PORT + 20):
        if _alive(port):
            pages = _pages(port)
            session = "отдельный процесс (переживает перезапуск сессии)"
            return (
                f"сервер:   {_base_url(port)}\n"
                f"каталог:  {PAGES_DIR}\n"
                f"процесс:  {session}\n"
                f"страниц:  {len(pages)}\n"
                f"текущая:  {pages[-1]['name'] if pages else '—'}"
            )
    return (
        "сервер предпросмотра не запущен — поднимется при первой публикации\n"
        f"каталог: {PAGES_DIR}"
    )


@mcp.tool()
def preview_open(name: str) -> str:
    """Открыть опубликованную страницу в браузере."""

    safe = _safe_name(name)
    if not (PAGES_DIR / f"{safe}.html").is_file():
        return f"нет страницы {safe}"
    port, _ = _ensure_server()
    url = f"{_base_url(port)}/{safe}"
    return f"открыто: {url} ({_open(url)})"


@mcp.tool()
def preview_stop() -> str:
    """Остановить сервер предпросмотра. Опубликованные файлы остаются на диске."""

    for port in range(DEFAULT_PORT, DEFAULT_PORT + 20):
        if not _alive(port):
            continue
        try:
            # Штатной ручки остановки у сервера нет: он живёт отдельно и не
            # должен выключаться по чужой команде из другого процесса. Поэтому
            # находим владельца порта и завершаем его.
            found = subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return f"остановить не удалось: {exc}"
        if found.returncode == 0:
            return f"сервер на порту {port} остановлен; страницы остались в {PAGES_DIR}"
        return f"владелец порта {port} не найден"
    return "сервер и так не запущен"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
