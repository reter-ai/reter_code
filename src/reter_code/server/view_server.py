"""
RETER View Server.

HTTP + WebSocket server for pushing live markdown/mermaid content to browsers.
Runs alongside ZMQ in a daemon thread using aiohttp.

::: This is-in-layer Presentation-Layer.
::: This is-in-component View-Server.
::: This depends-on aiohttp.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from aiohttp import web, WSMsgType

logger = logging.getLogger(__name__)

_VIEW_DIR = Path(__file__).parent / "view"


class ViewServer:
    """HTTP + WebSocket server for live browser visualization.

    ::: This is-in-layer Presentation-Layer.
    ::: This is a service.
    ::: This is stateful.

    Serves an HTML page with mermaid.js + marked.js and pushes content
    to connected browsers via WebSocket.
    """

    _MAX_HISTORY = 200

    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 history_dir: Optional[str] = None,
                 diagrams_dir: Optional[str] = None):
        self._host = host
        self._port = port  # 0 = OS picks free port
        self._actual_port: int = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner: Optional[web.AppRunner] = None
        self._ready = threading.Event()
        self._clients: Set[web.WebSocketResponse] = set()
        self._last_message: Optional[Dict[str, Any]] = None

        # File-based history + diagrams directory
        self._history_dir: Optional[Path] = None
        self._history_index: List[Dict[str, Any]] = []  # metadata only
        self._next_id: int = 1
        self._diagrams_dir: Optional[Path] = None

        if diagrams_dir:
            self._diagrams_dir = Path(diagrams_dir)
            self._diagrams_dir.mkdir(parents=True, exist_ok=True)

        if history_dir:
            self._history_dir = Path(history_dir)
            self._history_dir.mkdir(parents=True, exist_ok=True)
            self._load_history_index()

    @property
    def port(self) -> int:
        return self._actual_port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._actual_port}"

    # -- routes ---------------------------------------------------------------

    async def _handle_index(self, request: web.Request) -> web.Response:
        import time
        html = (_VIEW_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("/app.js", f"/app.js?v={int(time.time())}")
        return web.Response(
            text=html,
            content_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )

    async def _handle_style(self, request: web.Request) -> web.Response:
        css = (_VIEW_DIR / "style.css").read_text(encoding="utf-8")
        return web.Response(
            text=css,
            content_type="text/css",
            headers={"Cache-Control": "no-cache"},
        )

    async def _handle_app_js(self, request: web.Request) -> web.Response:
        js = (_VIEW_DIR / "app.js").read_text(encoding="utf-8")
        return web.Response(
            text=js,
            content_type="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    async def _handle_favicon(self, request: web.Request) -> web.Response:
        svg = (_VIEW_DIR / "favico.svg").read_text(encoding="utf-8")
        return web.Response(text=svg, content_type="image/svg+xml")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        logger.debug("ViewServer: WS client connected (%d total)", len(self._clients))
        try:
            # Send cached content immediately (fall back to disk on server restart)
            if self._last_message is None and self._history_index:
                last = self._history_index[-1]
                data = self._read_history_file(last["id"])
                if data:
                    self._last_message = {
                        "id": data["id"], "type": data["content_type"],
                        "content": data["content"],
                    }
            if self._last_message is not None:
                await ws.send_str(json.dumps(self._last_message))
            async for msg in ws:
                if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self._clients.discard(ws)
            logger.debug("ViewServer: WS client disconnected (%d remain)", len(self._clients))
        return ws

    # -- history helpers ------------------------------------------------------

    @staticmethod
    def _extract_title(content_type: str, content: str) -> str:
        """Extract a short title from the content for the history list."""
        if content_type == "markdown":
            m = re.search(r"^#{1,6}\s+(.+)", content, re.MULTILINE)
            if m:
                return m.group(1).strip()[:120]
        if content_type == "mermaid":
            first = content.strip().splitlines()[0] if content.strip() else ""
            return first[:120] or "Mermaid diagram"
        # html or fallback
        return content.strip()[:80] or "Untitled"

    def _history_file_path(self, item_id: int) -> Path:
        """Return the path for a history JSON file by ID."""
        return self._history_dir / f"{item_id:05d}.json"

    def _load_history_index(self) -> None:
        """Scan history directory and build in-memory index on startup."""
        self._history_index = []
        self._next_id = 1
        if not self._history_dir or not self._history_dir.is_dir():
            return
        files = sorted(self._history_dir.glob("*.json"))
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self._history_index.append({
                    "id": data["id"],
                    "timestamp": data["timestamp"],
                    "content_type": data["content_type"],
                    "title": data["title"],
                })
                if data["id"] >= self._next_id:
                    self._next_id = data["id"] + 1
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning("ViewServer: skipping corrupt history file %s: %s", f, e)

    def _read_history_file(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Read a single history JSON file by ID."""
        if not self._history_dir:
            return None
        path = self._history_file_path(item_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("ViewServer: failed to read history file %s: %s", path, e)
            return None

    def _save_history(self, message: Dict[str, Any]) -> Optional[int]:
        """Write a history JSON file and return the new ID."""
        if not self._history_dir:
            return None
        content_type = message.get("type", "html")
        content = message.get("content", "")
        title = message.get("title") or self._extract_title(content_type, content)
        item_id = self._next_id
        self._next_id += 1
        record = {
            "id": item_id,
            "timestamp": _time.time(),
            "content_type": content_type,
            "title": title,
            "content": content,
        }
        path = self._history_file_path(item_id)
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        self._history_index.append({
            "id": item_id,
            "timestamp": record["timestamp"],
            "content_type": content_type,
            "title": title,
        })
        # Prune oldest entries beyond limit
        while len(self._history_index) > self._MAX_HISTORY:
            oldest = self._history_index.pop(0)
            old_path = self._history_file_path(oldest["id"])
            try:
                old_path.unlink(missing_ok=True)
            except OSError:
                pass
        return item_id

    async def _handle_history_list(self, request: web.Request) -> web.Response:
        """GET /api/history — list recent history items (no content)."""
        # Return index in reverse chronological order (newest first)
        items = list(reversed(self._history_index))
        return web.json_response(items)

    async def _handle_history_item(self, request: web.Request) -> web.Response:
        """GET /api/history/{id} — return full content for one item."""
        item_id = int(request.match_info["id"])
        data = self._read_history_file(item_id)
        if not data:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(data)

    async def _handle_history_save(self, request: web.Request) -> web.Response:
        """POST /api/history/{id}/save — save content to diagrams folder."""
        if not self._diagrams_dir:
            return web.json_response({"error": "no diagrams directory"}, status=500)
        item_id = int(request.match_info["id"])
        data = self._read_history_file(item_id)
        if not data:
            return web.json_response({"error": "not found"}, status=404)

        content_type, title, content = data["content_type"], data["title"], data["content"]
        # Sanitize title for filename
        safe = re.sub(r'[^\w\s-]', '', title).strip()[:60]
        safe = re.sub(r'\s+', '_', safe) or f"view_{item_id}"
        ext = ".md" if content_type == "markdown" else ".mmd"
        filename = f"{safe}{ext}"
        filepath = self._diagrams_dir / filename

        # Avoid overwriting — append counter
        counter = 1
        while filepath.exists():
            filepath = self._diagrams_dir / f"{safe}_{counter}{ext}"
            counter += 1

        filepath.write_text(content, encoding="utf-8")
        return web.json_response({"path": str(filepath)})

    async def _handle_open_file(self, request: web.Request) -> web.Response:
        """POST /api/open-file — open a file with the system default app."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad request"}, status=400)
        filepath = body.get("path", "")
        if not filepath or not Path(filepath).is_file():
            return web.json_response({"error": "file not found"}, status=404)
        # Security: only allow opening files inside the diagrams dir
        if self._diagrams_dir:
            try:
                Path(filepath).resolve().relative_to(self._diagrams_dir.resolve())
            except ValueError:
                return web.json_response({"error": "forbidden"}, status=403)
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", filepath])
        else:
            subprocess.Popen(["xdg-open", filepath])
        return web.json_response({"ok": True})

    # -- broadcast ------------------------------------------------------------

    async def _broadcast(self, message: Dict[str, Any]) -> None:
        row_id = self._save_history(message)
        payload_dict = dict(message)
        if row_id is not None:
            payload_dict["id"] = row_id
        self._last_message = payload_dict
        payload = json.dumps(payload_dict)
        closed = []
        for ws in self._clients:
            try:
                await ws.send_str(payload)
            except Exception:
                closed.append(ws)
        for ws in closed:
            self._clients.discard(ws)

    def push(self, message: Dict[str, Any]) -> None:
        """Thread-safe push from sync code to all WS clients."""
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    # -- lifecycle ------------------------------------------------------------

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except RuntimeError:
            pass  # "Event loop stopped before Future completed" on shutdown

    async def _serve(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/style.css", self._handle_style)
        app.router.add_get("/app.js", self._handle_app_js)
        app.router.add_get("/favico.svg", self._handle_favicon)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/api/history", self._handle_history_list)
        app.router.add_get("/api/history/{id}", self._handle_history_item)
        app.router.add_post("/api/history/{id}/save", self._handle_history_save)
        app.router.add_post("/api/open-file", self._handle_open_file)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        # Resolve actual port
        sockets = site._server.sockets  # type: ignore[union-attr]
        if sockets:
            self._actual_port = sockets[0].getsockname()[1]

        self._ready.set()

        # Keep running until loop is stopped
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await self._runner.cleanup()

    def start(self) -> None:
        """Start the view server in a daemon thread. Blocks until port is bound."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="view-server")
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("ViewServer failed to start within 10s")
        logger.info("ViewServer listening on %s", self.url)

    def stop(self) -> None:
        """Stop the view server."""
        if self._loop and not self._loop.is_closed():
            # Cancel all tasks and stop the loop
            for task in asyncio.all_tasks(self._loop):
                self._loop.call_soon_threadsafe(task.cancel)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("ViewServer stopped")
