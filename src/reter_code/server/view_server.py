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
import sqlite3
import subprocess
import sys
import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, Optional, Set

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

    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 db_path: Optional[str] = None):
        self._host = host
        self._port = port  # 0 = OS picks free port
        self._actual_port: int = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner: Optional[web.AppRunner] = None
        self._ready = threading.Event()
        self._clients: Set[web.WebSocketResponse] = set()
        self._last_message: Optional[Dict[str, Any]] = None

        # SQLite persistence + diagrams directory
        self._db: Optional[sqlite3.Connection] = None
        self._diagrams_dir: Optional[Path] = None
        if db_path:
            self._diagrams_dir = Path(db_path).parent / "diagrams"
            self._diagrams_dir.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS view_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    content_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_vh_timestamp
                ON view_history(timestamp DESC)
            """)
            self._db.commit()

    @property
    def port(self) -> int:
        return self._actual_port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._actual_port}"

    # -- routes ---------------------------------------------------------------

    async def _handle_index(self, request: web.Request) -> web.Response:
        html = (_VIEW_DIR / "index.html").read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html")

    async def _handle_style(self, request: web.Request) -> web.Response:
        css = (_VIEW_DIR / "style.css").read_text(encoding="utf-8")
        return web.Response(text=css, content_type="text/css")

    async def _handle_app_js(self, request: web.Request) -> web.Response:
        js = (_VIEW_DIR / "app.js").read_text(encoding="utf-8")
        return web.Response(text=js, content_type="application/javascript")

    async def _handle_favicon(self, request: web.Request) -> web.Response:
        svg = (_VIEW_DIR / "favico.svg").read_text(encoding="utf-8")
        return web.Response(text=svg, content_type="image/svg+xml")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        logger.debug("ViewServer: WS client connected (%d total)", len(self._clients))
        try:
            # Send cached content immediately (fall back to DB on server restart)
            if self._last_message is None and self._db:
                row = self._db.execute(
                    "SELECT id, content_type, content "
                    "FROM view_history ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if row:
                    self._last_message = {
                        "id": row[0], "type": row[1], "content": row[2],
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

    def _save_history(self, message: Dict[str, Any]) -> Optional[int]:
        """Insert into view_history and return the new row id."""
        if not self._db:
            return None
        content_type = message.get("type", "html")
        content = message.get("content", "")
        title = self._extract_title(content_type, content)
        cur = self._db.execute(
            "INSERT INTO view_history (timestamp, content_type, title, content) "
            "VALUES (?, ?, ?, ?)",
            (_time.time(), content_type, title, content),
        )
        self._db.commit()
        return cur.lastrowid

    async def _handle_history_list(self, request: web.Request) -> web.Response:
        """GET /api/history — list recent history items (no content)."""
        if not self._db:
            return web.json_response([])
        rows = self._db.execute(
            "SELECT id, timestamp, content_type, title "
            "FROM view_history ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
        items = [
            {"id": r[0], "timestamp": r[1], "content_type": r[2], "title": r[3]}
            for r in rows
        ]
        return web.json_response(items)

    async def _handle_history_item(self, request: web.Request) -> web.Response:
        """GET /api/history/{id} — return full content for one item."""
        if not self._db:
            return web.json_response({"error": "no database"}, status=500)
        item_id = int(request.match_info["id"])
        row = self._db.execute(
            "SELECT id, timestamp, content_type, title, content "
            "FROM view_history WHERE id = ?",
            (item_id,),
        ).fetchone()
        if not row:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({
            "id": row[0], "timestamp": row[1], "content_type": row[2],
            "title": row[3], "content": row[4],
        })

    async def _handle_history_save(self, request: web.Request) -> web.Response:
        """POST /api/history/{id}/save — save content to diagrams folder."""
        if not self._db or not self._diagrams_dir:
            return web.json_response({"error": "no database"}, status=500)
        item_id = int(request.match_info["id"])
        row = self._db.execute(
            "SELECT content_type, title, content FROM view_history WHERE id = ?",
            (item_id,),
        ).fetchone()
        if not row:
            return web.json_response({"error": "not found"}, status=404)

        content_type, title, content = row
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
        if self._db:
            self._db.close()
            self._db = None
        logger.info("ViewServer stopped")
