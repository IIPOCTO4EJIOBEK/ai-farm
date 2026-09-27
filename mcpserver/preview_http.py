"""HTTP-сервер предпросмотра: раздаёт страницы и рассылает живые обновления.

Живёт отдельным процессом, а не внутри MCP-сервера. Причина простая: MCP-клиент
перезапускает свой сервер при каждом обновлении конфига и при переподключении, и
если страница раздаётся из того же процесса, предпросмотр отваливается вместе с
ним. Отдельный процесс переживает и перезапуск, и закрытие сессии.

Об изменениях сервер узнаёт, следя за каталогом страниц: сравнивает время
правки файлов. Так не нужен общий канал с MCP-сервером — файлы и есть канал.
Опрос раз в полсекунды незаметен на глаз, а сложности с блокировками не
приносит.

    python -m mcpserver.preview_http [--port 8791] [--pages КАТАЛОГ]
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGES_DIR = Path.home() / ".local" / "share" / "ai-preview"
DEFAULT_PORT = 8791
POLL_SECONDS = 0.5

# Вставка перед </body>: подписка на события сервера. Пока страница открыта,
# правка файла доезжает до экрана сама.
LIVE_RELOAD = """
<script>
(function () {
  // Живое обновление. Если сервер недоступен, страница остаётся как есть —
  // для простого просмотра это не ошибка.
  try {
    var source = new EventSource('/__events');
    source.onmessage = function (event) {
      if (event.data === 'reload') { location.reload(); }
    };
  } catch (error) { /* просмотр без обновления */ }
})();
</script>
"""


def safe_name(name: str) -> str:
    """Привести имя страницы к безопасному виду.

    Имя приходит от модели, а попадает и в путь на диске, и в адрес. Поэтому
    разрешены только буквы, цифры, дефис и подчёркивание: иначе «../» в имени
    вывел бы запись за пределы каталога страниц.
    """

    cleaned = "".join(char for char in name if char.isalnum() or char in "-_")
    return cleaned or "index"


def inject_live_reload(html: str) -> str:
    """Вставить клиент живого обновления перед закрытием body."""

    marker = html.lower().rfind("</body>")
    if marker == -1:
        return html + LIVE_RELOAD
    return html[:marker] + LIVE_RELOAD + html[marker:]


class Watcher:
    """Следит за каталогом страниц и будит подписчиков при изменениях."""

    def __init__(self, pages: Path) -> None:
        self.pages = pages
        self.lock = threading.Lock()
        self.subscribers: list[queue.Queue[str]] = []
        self.version = 0
        self.current = ""
        self._stamps: dict[str, float] = {}

    def subscribe(self) -> queue.Queue[str]:
        channel: queue.Queue[str] = queue.Queue(maxsize=8)
        with self.lock:
            self.subscribers.append(channel)
        return channel

    def unsubscribe(self, channel: queue.Queue[str]) -> None:
        with self.lock:
            if channel in self.subscribers:
                self.subscribers.remove(channel)

    def notify(self) -> None:
        with self.lock:
            self.version += 1
            for subscriber in list(self.subscribers):
                try:
                    subscriber.put_nowait("reload")
                except queue.Full:
                    pass

    def scan(self) -> None:
        stamps: dict[str, float] = {}
        latest_name, latest_time = "", 0.0
        for path in self.pages.glob("*.html"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            stamps[path.name] = mtime
            if mtime > latest_time:
                latest_name, latest_time = path.stem, mtime
        changed = stamps != self._stamps
        self._stamps = stamps
        if latest_name:
            self.current = latest_name
        if changed:
            self.notify()

    def loop(self) -> None:
        while True:
            try:
                self.scan()
            except OSError:
                pass
            time.sleep(POLL_SECONDS)


class Handler(BaseHTTPRequestHandler):
    """Раздача страниц, списка и канала событий."""

    protocol_version = "HTTP/1.1"
    watcher: Watcher

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Молчать: сервер работает фоном, логи только мешают."""

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path == "/__events":
            self._stream_events()
            return
        if path == "/__pages":
            self._send(200, list_json(self.watcher).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path in ("/", "/__root"):
            name = self.watcher.current
            if not name:
                self._send(404, "страниц пока нет".encode("utf-8"),
                           "text/plain; charset=utf-8")
                return
        else:
            name = safe_name(path.strip("/").removesuffix(".html"))

        page = self.watcher.pages / f"{name}.html"
        if not page.is_file():
            self._send(404, f"нет страницы {name}".encode("utf-8"),
                       "text/plain; charset=utf-8")
            return
        html = page.read_text(encoding="utf-8", errors="replace")
        self._send(200, inject_live_reload(html).encode("utf-8"), "text/html; charset=utf-8")

    def _stream_events(self) -> None:
        """Канал событий: соединение живёт, пока открыта вкладка."""

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        channel = self.watcher.subscribe()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    event = channel.get(timeout=15)
                except queue.Empty:
                    # Комментарий вместо события: держит соединение живым,
                    # чтобы браузер не закрыл его по таймауту.
                    self.wfile.write(b": keep-alive\n\n")
                else:
                    self.wfile.write(f"data: {event}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.watcher.unsubscribe(channel)


def list_json(watcher: Watcher) -> str:
    pages = []
    for path in sorted(watcher.pages.glob("*.html")):
        try:
            stat = path.stat()
        except OSError:
            continue
        pages.append(
            {
                "name": path.stem,
                "url": f"http://127.0.0.1:{watcher.port}/{path.stem}",
                "bytes": stat.st_size,
                "modified": time.strftime("%H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
    return json.dumps(pages, ensure_ascii=False)


def serve(port: int, pages: Path, *, retries: int = 20) -> int:
    pages.mkdir(parents=True, exist_ok=True)
    watcher = Watcher(pages)
    watcher.port = port  # type: ignore[attr-defined]
    watcher.scan()
    threading.Thread(target=watcher.loop, daemon=True).start()

    handler = type("BoundHandler", (Handler,), {"watcher": watcher})
    for candidate in range(port, port + retries):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
        except OSError:
            continue
        watcher.port = candidate  # type: ignore[attr-defined]
        print(f"предпросмотр: http://127.0.0.1:{candidate}", file=sys.stderr, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    print("не удалось занять порт", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP-сервер предпросмотра страниц")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--pages", type=Path, default=PAGES_DIR)
    args = parser.parse_args()
    return serve(args.port, args.pages)


if __name__ == "__main__":
    raise SystemExit(main())
