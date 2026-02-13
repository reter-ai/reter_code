"""
RETER Console UI.

Rich terminal interface for RETER server with live updates.

::: This is-in-layer UI-Layer.
::: This is-in-component Console-UI.
::: This depends-on rich.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from .reter_server import ReterServer


@dataclass
class QueryLogEntry:
    """Single query log entry.

    ::: This is-in-layer Core-Layer.
    ::: This is a value-object.
    ::: This is stateless.
    """
    timestamp: float
    method: str
    duration_ms: float
    result_count: int
    error: Optional[str] = None

    @property
    def time_str(self) -> str:
        """Format timestamp as HH:MM:SS."""
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")


@dataclass
class ServerStatus:
    """Server status information.

    ::: This is-in-layer Core-Layer.
    ::: This is a value-object.
    ::: This is stateful.
    """
    started_at: float = field(default_factory=time.time)
    total_sources: int = 0
    total_wmes: int = 0
    total_vectors: int = 0
    connected_clients: int = 0
    queries_count: int = 0
    errors_count: int = 0
    avg_query_time_ms: float = 0.0

    # Progress tracking
    current_operation: Optional[str] = "Initializing..."  # Start with initializing
    progress_current: int = 0
    progress_total: int = 0
    current_file: Optional[str] = None  # Current file being processed
    initialized: bool = False  # True when server is ready

    # Spinner for indeterminate progress
    spinner_active: bool = False
    spinner_text: str = ""
    spinner_frame: int = 0


class ConsoleUI:
    """Rich console output for RETER server.

    ::: This is-in-layer Presentation-Layer.
    ::: This is a adapter.
    ::: This is stateful.
    ::: This depends-on `rich.console.Console`.

    Provides a live-updating terminal interface showing:
    - Server status and statistics
    - Recent query log
    - Progress bars for long operations
    - Client connections
    """

    MAX_LOG_ENTRIES = 50

    def __init__(self, server: "ReterServer"):
        """Initialize console UI.

        Args:
            server: Reference to RETER server for stats
        """
        if not RICH_AVAILABLE:
            raise ImportError("Rich library not available. Install with: pip install rich")

        self.server = server
        self.console = Console()
        self._refresh_thread = None

        self.status = ServerStatus()
        self.query_log: deque[QueryLogEntry] = deque(maxlen=self.MAX_LOG_ENTRIES)

        self._running = False
        self._keyboard_callback = None  # Set by server for keyboard polling

        # Log viewer state
        self._view_mode: str = "dashboard"  # "dashboard", "debug_log", "nlq_log", "source_tree"
        self._log_scroll_offset: int = 0  # 0 = follow tail, positive = lines scrolled up

        # Source tree cache (built once on enter, not every render)
        self._source_tree_cache: Optional[List[tuple]] = None
        self._source_tree_count: int = 0

        # Cached values (computed once, not every render)
        self._cached_core_version: Optional[str] = None
        self._cached_build_ts: Optional[str] = None
        self._cached_languages: Optional[str] = None
        self._cache_populated: bool = False

    def __rich__(self) -> Layout:
        """Return layout for Rich rendering."""
        return self._build_layout()

    def start(self) -> None:
        """Start display refresh thread."""
        import threading
        self._running = True
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _refresh_loop(self) -> None:
        """Background thread that refreshes the display."""
        import io
        import os
        import sys
        import time

        last_size = (0, 0)
        render_console = None  # Reuse Console across frames

        # Hide cursor at start
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        try:
            while self._running:
                try:
                    # Get current terminal size
                    try:
                        term_size = os.get_terminal_size()
                        width = term_size.columns
                        height = term_size.lines
                    except OSError:
                        width = 120
                        height = 30

                    current_size = (width, height)
                    size_changed = current_size != last_size
                    last_size = current_size

                    # Only do full clear on resize
                    if size_changed and current_size != (0, 0):
                        sys.stdout.write("\033[2J")  # ANSI clear (no subprocess)
                        render_console = None  # Force new Console with new size

                    # Reuse Console unless size changed
                    if render_console is None or size_changed:
                        render_console = Console(
                            width=width, height=height - 1,
                            force_terminal=True, file=io.StringIO()
                        )

                    # Render to string buffer (fast, no I/O)
                    render_console.file = io.StringIO()
                    render_console.print(self._build_layout(), height=height - 1, end="")
                    output = render_console.file.getvalue()

                    # Single write to stdout (minimizes flicker)
                    sys.stdout.write("\033[H" + output + "\033[J")
                    sys.stdout.flush()

                    # Poll keyboard if callback is set
                    if self._keyboard_callback:
                        try:
                            self._keyboard_callback()
                        except Exception:
                            pass
                except Exception:
                    pass  # Ignore errors during refresh
                # Faster refresh in log view for live-tail feel
                time.sleep(0.15 if self._view_mode != "dashboard" else 0.5)
        finally:
            # Show cursor when done
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    def stop(self) -> None:
        """Stop live display."""
        self._running = False
        # Clear screen on exit
        try:
            self.console.clear()
        except Exception:
            pass

    def _build_layout(self) -> Layout:
        """Build console layout with panels."""
        if self._view_mode in ("debug_log", "nlq_log"):
            return self._build_log_viewer_layout()
        if self._view_mode == "source_tree":
            return self._build_source_tree_layout()

        # Dashboard layout
        layout = Layout()

        # Fixed sizes for header, progress, footer - main is flexible
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="progress", size=9),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="queries", ratio=2)
        )

        # Update panels
        layout["header"].update(self._build_header())
        layout["status"].update(self._build_status_panel())
        layout["queries"].update(self._build_query_panel())
        layout["progress"].update(self._build_progress_panel())
        layout["footer"].update(self._build_footer())

        return layout

    def _build_header(self) -> Panel:
        """Build header panel."""
        title = Text()
        from reter_code import __version__
        title.append("RETER Server", style="bold blue")
        title.append(f" v{__version__}", style="dim")

        # Show port (only if bound)
        port = self.server._actual_query_port
        if port > 0:
            title.append(f"  [tcp://127.0.0.1:{port}]", style="green")
        else:
            title.append("  [binding...]", style="yellow")

        # Show view server URL
        if self.server._view_server:
            title.append(f"  [{self.server._view_server.url}]", style="magenta")

        # Show project root
        if self.server.config.project_root:
            root_str = str(self.server.config.project_root)
            # Truncate if too long
            max_len = 50
            if len(root_str) > max_len:
                root_str = "..." + root_str[-(max_len-3):]
            title.append(f"  [{root_str}]", style="cyan")

        return Panel(title, style="blue")

    def _build_status_panel(self) -> Panel:
        """Build status panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim")
        table.add_column("Value", style="bold")

        # Uptime
        uptime = time.time() - self.status.started_at
        uptime_str = self._format_duration(uptime)
        table.add_row("Uptime", uptime_str)

        # Stats from server
        stats = self.server.stats
        table.add_row("", "")
        table.add_row("Sources", str(self.status.total_sources))
        table.add_row("Facts", f"{self.status.total_wmes:,}")
        table.add_row("Vectors", f"{self.status.total_vectors:,}")
        table.add_row("", "")
        table.add_row("Queries", str(stats.get("requests_handled", 0)))
        table.add_row("Errors", str(stats.get("errors", 0)))
        table.add_row("Avg Time", f"{stats.get('avg_request_time_ms', 0):.1f}ms")

        # Populate cache once (expensive lookups)
        if not self._cache_populated:
            try:
                from importlib.metadata import version as pkg_version
                self._cached_core_version = pkg_version("reter_core")
                from reter import owl_rete_cpp
                self._cached_build_ts = getattr(owl_rete_cpp, "__build_timestamp__", None)
            except Exception:
                pass
            try:
                from ..reter_loaders import LANGUAGE_CONFIGS
                langs = sorted(LANGUAGE_CONFIGS.keys())
                self._cached_languages = f"{len(langs)}: {', '.join(langs)}"
            except Exception:
                pass
            self._cache_populated = True

        # C++ core version
        if self._cached_core_version:
            table.add_row("", "")
            table.add_row("Core", self._cached_core_version)
            if self._cached_build_ts:
                table.add_row("Built", self._cached_build_ts)

        # Supported languages
        if self._cached_languages:
            table.add_row("", "")
            table.add_row("Languages", self._cached_languages)

        # Config
        try:
            from ..services.config_loader import get_config_loader
            loader = get_config_loader()
            config = loader.config
            if config:
                table.add_row("", "")
                config_path = loader.config_path
                if config_path:
                    table.add_row("Config", str(config_path.name))
                for key, value in config.items():
                    val_str = str(value)
                    if len(val_str) > 40:
                        val_str = val_str[:37] + "..."
                    table.add_row(f"  {key}", val_str)
        except Exception:
            pass

        return Panel(table, title="Status", border_style="blue")

    def _build_progress_panel(self) -> Panel:
        """Build progress panel for current operation."""
        content = Text()

        # Spinner animation frames
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        if self.status.spinner_active:
            # Show spinner animation
            frame = spinner_frames[self.status.spinner_frame % len(spinner_frames)]
            self.status.spinner_frame += 1
            content.append(f"{frame} ", style="bold cyan")
            content.append(self.status.spinner_text, style="bold yellow")
            content.append("...", style="dim")
        elif self.status.current_operation:
            # Operation name
            content.append(self.status.current_operation, style="bold yellow")

            # Progress bar (only show if we have progress data)
            if self.status.progress_total > 0:
                percent = self.status.progress_current / self.status.progress_total * 100

                bar_width = 50
                filled = int(bar_width * percent / 100)
                content.append("\n  [", style="dim")  # Single newline
                content.append("█" * filled, style="green")
                content.append("░" * (bar_width - filled), style="dim")
                content.append("]", style="dim")
                content.append(f"  {percent:5.1f}%", style="bold white")
                content.append(f"   {self.status.progress_current:,} / {self.status.progress_total:,}", style="dim")

            # Current file - always show on new line
            content.append("\n  ")  # Single newline
            if self.status.current_file:
                content.append("→ ", style="cyan")
                # Show more of the path since we have space
                display_path = self._format_file_path(self.status.current_file, max_width=80)
                content.append(display_path, style="cyan")
            else:
                content.append("→ ", style="dim")
                content.append("waiting...", style="dim")
        elif self.status.initialized:
            content.append("Ready", style="bold green")
            content.append(" - Server is idle, waiting for queries\n\n", style="dim")
            # Show browser URL if view server is running
            if self.server._view_server:
                content.append("  Browser: ", style="dim")
                content.append(self.server._view_server.url, style="bold magenta underline")
                content.append("\n", style="dim")
            else:
                content.append("\n", style="dim")
            content.append("  Add MCP:  ", style="dim")
            project_root = str(self.server.config.project_root) if self.server.config.project_root else ""
            content.append(self._get_mcp_command(project_root), style="bold white")
            content.append("\n  ", style="dim")
            content.append("[C]", style="bold")
            content.append("opy to clipboard", style="dim")
        else:
            content.append("Starting up...", style="bold yellow")

        return Panel(content, title="Progress", border_style="yellow")

    def _build_query_panel(self) -> Panel:
        """Build query log panel."""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Time", style="dim", width=8)
        table.add_column("Method", width=15)
        table.add_column("Results", justify="right", width=8)
        table.add_column("Time", justify="right", width=8)

        # Show recent queries (most recent first)
        for entry in reversed(list(self.query_log)[-20:]):
            time_str = entry.time_str
            method = entry.method

            if entry.error:
                result_str = Text("ERROR", style="red")
            else:
                result_str = str(entry.result_count)

            duration_str = f"{entry.duration_ms:.0f}ms"

            # Color code by duration
            if entry.duration_ms > 1000:
                duration_style = "red"
            elif entry.duration_ms > 100:
                duration_style = "yellow"
            else:
                duration_style = "green"

            table.add_row(
                time_str,
                method,
                result_str,
                Text(duration_str, style=duration_style)
            )

        return Panel(table, title="Query Log", border_style="blue")

    def _build_footer(self) -> Panel:
        """Build footer panel."""
        text = Text()
        text.append("[K]", style="bold")
        text.append("ompact  ", style="dim")
        text.append("[D]", style="bold")
        text.append("ebug log  ", style="dim")
        text.append("[N]", style="bold")
        text.append("lq log  ", style="dim")
        text.append("[S]", style="bold")
        text.append("ources  ", style="dim")
        text.append("[Ctrl+C]", style="bold")
        text.append(" Exit", style="dim")

        return Panel(text, style="dim")

    # =========================================================================
    # Log viewer
    # =========================================================================

    def _build_log_viewer_layout(self) -> Layout:
        """Build layout for log file viewer mode."""
        import os

        try:
            term_size = os.get_terminal_size()
            term_height = term_size.lines
        except OSError:
            term_height = 30

        # Determine log file info
        if self._view_mode == "debug_log":
            filename = "debug_trace.log"
            title = "Debug Trace Log"
        else:
            filename = "nlq_debug.log"
            title = "NLQ Debug Log"

        log_path = self._get_log_path(filename)

        # Header
        header_text = Text()
        header_text.append(title, style="bold blue")
        header_text.append(f"  [{log_path}]", style="dim cyan")

        # Footer
        footer_text = Text()
        footer_text.append("[ESC]", style="bold")
        footer_text.append(" Back  ", style="dim")
        footer_text.append("[↑/↓]", style="bold")
        footer_text.append(" Scroll  ", style="dim")
        footer_text.append("[Home]", style="bold")
        footer_text.append(" Top  ", style="dim")
        footer_text.append("[End]", style="bold")
        footer_text.append(" Follow", style="dim")
        if self._log_scroll_offset == 0:
            footer_text.append("  ", style="dim")
            footer_text.append("● FOLLOWING", style="bold green")

        # Get terminal width for line truncation (subtract panel borders)
        try:
            term_width = os.get_terminal_size().columns - 4
        except OSError:
            term_width = 116

        # Available height for log text:
        # term_height - 1 (console) - 3 (header) - 3 (footer) - 2 (content panel borders)
        visible_lines = max(1, term_height - 9)

        # Read only what we need from the tail of the file
        max_lines_needed = visible_lines + self._log_scroll_offset
        lines = self._read_log_tail(filename, max_lines_needed)
        total_lines = len(lines)

        # Calculate visible window
        if self._log_scroll_offset == 0:
            # Follow mode: show last N lines
            start = max(0, total_lines - visible_lines)
            end = total_lines
        else:
            # Scrolled: show lines ending at total - offset
            end = max(0, total_lines - self._log_scroll_offset)
            start = max(0, end - visible_lines)

        # Build content
        content = Text(no_wrap=True, overflow="ellipsis")
        if total_lines == 0:
            content.append("(empty log file)", style="dim italic")
        else:
            visible = lines[start:end]
            for i, line in enumerate(visible):
                if i > 0:
                    content.append("\n")
                # Truncate long lines to terminal width
                display = line[:term_width] + "…" if len(line) > term_width else line
                # Syntax-highlight log levels
                if " ERROR " in line or " CRITICAL " in line:
                    content.append(display, style="red")
                elif " WARNING " in line:
                    content.append(display, style="yellow")
                elif " DEBUG " in line:
                    content.append(display, style="dim")
                else:
                    content.append(display)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="content"),
            Layout(name="footer", size=3),
        )

        layout["header"].update(Panel(header_text, style="blue"))
        layout["content"].update(Panel(content, border_style="dim"))
        layout["footer"].update(Panel(footer_text, style="dim"))

        return layout

    def _get_log_path(self, filename: str) -> str:
        """Get full path to a log file."""
        from ..logging_config import _get_log_directory
        log_dir = _get_log_directory()
        if log_dir:
            return str(log_dir / filename)
        return f"(unknown)/{filename}"

    def _read_log_tail(self, filename: str, max_lines: int = 200) -> List[str]:
        """Read the last N lines from a log file efficiently.

        Reads from the end of the file to avoid loading the entire file.
        Returns empty list if file doesn't exist.
        """
        from ..logging_config import _get_log_directory
        log_dir = _get_log_directory()
        if not log_dir:
            return []
        log_file = log_dir / filename
        try:
            size = log_file.stat().st_size
            if size == 0:
                return []
            # Read a chunk from the end (generous: ~300 bytes per line)
            chunk_size = min(size, max_lines * 300)
            with open(log_file, "rb") as f:
                f.seek(max(0, size - chunk_size))
                data = f.read()
            lines = data.decode("utf-8", errors="replace").splitlines()
            # If we didn't read from the start, drop the first partial line
            if chunk_size < size:
                lines = lines[1:]
            return lines[-max_lines:]
        except FileNotFoundError:
            return []
        except Exception:
            return ["(error reading log file)"]

    def enter_log_view(self, mode: str) -> None:
        """Switch to log viewer mode.

        Args:
            mode: "debug_log" or "nlq_log"
        """
        self._view_mode = mode
        self._log_scroll_offset = 0

    def exit_log_view(self) -> None:
        """Return to dashboard mode."""
        self._view_mode = "dashboard"

    def scroll_log(self, delta: int) -> None:
        """Scroll log by delta lines (positive = scroll up, negative = scroll down).

        Args:
            delta: Number of lines to scroll. Positive scrolls up (older),
                   negative scrolls down (newer).
        """
        new_offset = self._log_scroll_offset + delta
        self._log_scroll_offset = max(0, new_offset)

    def scroll_log_home(self) -> None:
        """Scroll to top of log file."""
        # Large offset — _read_log_tail will clamp to what's available
        self._log_scroll_offset = 999999

    def scroll_log_end(self) -> None:
        """Scroll to bottom (follow mode)."""
        self._log_scroll_offset = 0

    @property
    def in_overlay_view(self) -> bool:
        """True if currently in an overlay view (not the dashboard)."""
        return self._view_mode in ("debug_log", "nlq_log", "source_tree")

    # =========================================================================
    # Source tree viewer
    # =========================================================================

    def enter_source_tree(self) -> None:
        """Switch to source tree view (fetches sources once)."""
        self._view_mode = "source_tree"
        self._log_scroll_offset = 0

        # Build tree cache once
        sources: List[str] = []
        if self.server._reter:
            try:
                raw_sources, _ = self.server._reter.get_all_sources()
                if raw_sources:
                    sources = raw_sources
            except Exception:
                pass

        self._source_tree_count = len(sources)

        if not sources:
            if self.server._reter is None:
                self._source_tree_cache = [("  (not initialized yet)", "dim italic")]
            else:
                self._source_tree_cache = [("  (no sources loaded)", "dim italic")]
        else:
            project_root = str(self.server.config.project_root) if self.server.config.project_root else ""
            root_prefix = project_root.replace("\\", "/").rstrip("/") + "/"
            rel_paths = []
            for s in sources:
                # Strip hash prefix (e.g. "abc123|path/to/file" → "path/to/file")
                if "|" in s:
                    s = s.split("|", 1)[1]
                normalized = s.replace("\\", "/")
                if normalized.startswith(root_prefix):
                    rel_paths.append(normalized[len(root_prefix):])
                else:
                    rel_paths.append(normalized)
            self._source_tree_cache = self._build_tree_lines(sorted(rel_paths))

    def _build_source_tree_layout(self) -> Layout:
        """Build layout for source tree view (renders from cache)."""
        import os

        try:
            term_size = os.get_terminal_size()
            term_height = term_size.lines
            term_width = term_size.columns - 4
        except OSError:
            term_height = 30
            term_width = 116

        tree_lines = self._source_tree_cache or [("  (no sources loaded)", "dim italic")]

        # Header
        header_text = Text()
        header_text.append("Source Tree", style="bold blue")
        header_text.append(f"  {self._source_tree_count} files", style="dim cyan")
        project_root = str(self.server.config.project_root) if self.server.config.project_root else ""
        if project_root:
            root_display = project_root
            if len(root_display) > 60:
                root_display = "..." + root_display[-57:]
            header_text.append(f"  [{root_display}]", style="dim")

        # Footer
        footer_text = Text()
        footer_text.append("[ESC]", style="bold")
        footer_text.append(" Back  ", style="dim")
        footer_text.append("[↑/↓]", style="bold")
        footer_text.append(" Scroll  ", style="dim")
        footer_text.append("[Home]", style="bold")
        footer_text.append(" Top  ", style="dim")
        footer_text.append("[End]", style="bold")
        footer_text.append(" Bottom", style="dim")

        # Visible window (top-anchored scroll)
        visible_lines = max(1, term_height - 9)
        total_lines = len(tree_lines)
        start = max(0, min(self._log_scroll_offset, max(0, total_lines - visible_lines)))
        end = min(total_lines, start + visible_lines)

        # Build content
        content = Text(no_wrap=True, overflow="ellipsis")
        visible = tree_lines[start:end]
        for i, (line_text, style) in enumerate(visible):
            if i > 0:
                content.append("\n")
            display = line_text[:term_width] if len(line_text) > term_width else line_text
            content.append(display, style=style)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="content"),
            Layout(name="footer", size=3),
        )
        layout["header"].update(Panel(header_text, style="blue"))
        layout["content"].update(Panel(content, border_style="dim"))
        layout["footer"].update(Panel(footer_text, style="dim"))
        return layout

    @staticmethod
    def _build_tree_lines(paths: List[str]) -> List[tuple]:
        """Convert sorted flat paths into tree-rendered lines with box-drawing chars.

        Returns list of (text, style) tuples.
        """
        # Extension → style map
        ext_styles = {
            ".py": "green",
            ".pyx": "green",
            ".pyi": "green",
            ".js": "yellow",
            ".ts": "yellow",
            ".tsx": "yellow",
            ".jsx": "yellow",
            ".md": "dim",
            ".txt": "dim",
            ".json": "cyan",
            ".yaml": "cyan",
            ".yml": "cyan",
            ".toml": "cyan",
            ".cfg": "cyan",
            ".ini": "cyan",
            ".html": "magenta",
            ".css": "magenta",
            ".scss": "magenta",
            ".c": "blue",
            ".cpp": "blue",
            ".h": "blue",
            ".hpp": "blue",
            ".rs": "red",
            ".go": "bright_cyan",
            ".java": "bright_red",
            ".cs": "bright_green",
            ".rb": "red",
            ".sh": "bright_yellow",
            ".bat": "bright_yellow",
            ".sql": "bright_blue",
        }

        if not paths:
            return []

        # Build a nested dict representing the directory tree
        tree: Dict[str, Any] = {}
        for p in paths:
            parts = p.split("/")
            node = tree
            for part in parts:
                if part not in node:
                    node[part] = {}
                node = node[part]

        # Walk the tree and produce lines
        result: List[tuple] = []

        def walk(node: dict, prefix: str, depth: int):
            entries = sorted(node.keys(), key=lambda k: (not bool(node[k]), k.lower()))
            for i, name in enumerate(entries):
                is_last = (i == len(entries) - 1)
                connector = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")

                children = node[name]
                if children:
                    # Directory
                    result.append((f"{prefix}{connector}{name}/", "bold cyan"))
                    walk(children, child_prefix, depth + 1)
                else:
                    # File — color by extension
                    ext = ""
                    dot_pos = name.rfind(".")
                    if dot_pos >= 0:
                        ext = name[dot_pos:]
                    style = ext_styles.get(ext.lower(), "white")
                    result.append((f"{prefix}{connector}{name}", style))

        walk(tree, "  ", 0)
        return result

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.0f}m"
        else:
            hours = seconds / 3600
            mins = (seconds % 3600) / 60
            return f"{hours:.0f}h {mins:.0f}m"

    def _format_file_path(self, path: str, max_width: int = 35) -> str:
        """Format file path for display, keeping most informative parts.

        Strategy:
        - Always show the filename
        - Show as much of the parent path as fits
        - Use ... to indicate truncation in the middle
        """
        import os.path

        # Normalize separators
        path = path.replace("\\", "/")

        # If it fits, show it all
        if len(path) <= max_width:
            return path

        # Split into parts
        parts = path.split("/")
        filename = parts[-1]

        # If just filename is too long, truncate it
        if len(filename) > max_width:
            return filename[:max_width - 3] + "..."

        # Try to fit parent folder + filename
        if len(parts) >= 2:
            parent = parts[-2]
            short_path = f"{parent}/{filename}"
            if len(short_path) <= max_width:
                return short_path

            # Try with ellipsis prefix
            short_path = f".../{parent}/{filename}"
            if len(short_path) <= max_width:
                return short_path

        # Fall back to just filename with ellipsis
        result = f".../{filename}"
        if len(result) <= max_width:
            return result

        # Last resort: truncate filename
        available = max_width - 4  # for ".../""
        return f".../{filename[:available]}"

    @staticmethod
    def _get_mcp_command(project_root: str) -> str:
        """Build the claude mcp add command with project root."""
        import shutil
        if shutil.which("reter_code"):
            return "claude mcp add reter -- reter_code --stdio"
        _UVX_FROM = "git+https://github.com/reter-ai/reter_code"
        _FIND_LINKS = "https://raw.githubusercontent.com/reter-ai/reter/main/reter_core/index.html"
        return f"claude mcp add reter -- uvx --from {_UVX_FROM} --find-links {_FIND_LINKS} reter_code --stdio"

    def log_query(
        self,
        method: str,
        duration_ms: float,
        result_count: int,
        error: Optional[str] = None
    ) -> None:
        """Log a query execution.

        Args:
            method: Method name
            duration_ms: Execution time
            result_count: Number of results
            error: Error message if failed
        """
        entry = QueryLogEntry(
            timestamp=time.time(),
            method=method,
            duration_ms=duration_ms,
            result_count=result_count,
            error=error
        )
        self.query_log.append(entry)
        self._refresh()

    def update_status(
        self,
        total_sources: Optional[int] = None,
        total_wmes: Optional[int] = None,
        total_vectors: Optional[int] = None,
        connected_clients: Optional[int] = None
    ) -> None:
        """Update server status.

        Args:
            total_sources: Number of loaded sources
            total_wmes: Number of facts
            total_vectors: Number of RAG vectors
            connected_clients: Number of connected clients
        """
        if total_sources is not None:
            self.status.total_sources = total_sources
        if total_wmes is not None:
            self.status.total_wmes = total_wmes
        if total_vectors is not None:
            self.status.total_vectors = total_vectors
        if connected_clients is not None:
            self.status.connected_clients = connected_clients
        self._refresh()

    def update_progress(
        self,
        operation: Optional[str],
        current: int = 0,
        total: int = 0
    ) -> None:
        """Update progress display.

        Args:
            operation: Current operation name (None to clear)
            current: Current progress value
            total: Total progress value
        """
        self.status.current_operation = operation
        self.status.progress_current = current
        self.status.progress_total = total
        self._refresh()

    # =========================================================================
    # ConsoleProgress-compatible methods for initialization callbacks
    # =========================================================================

    def set_phase(self, phase: str) -> None:
        """Set current phase description."""
        self.status.current_operation = phase
        self._refresh()

    def start_spinner(self, text: str) -> None:
        """Start spinner animation for indeterminate progress."""
        self.status.spinner_active = True
        self.status.spinner_text = text
        self.status.spinner_frame = 0
        self.status.current_operation = None  # Clear regular operation
        self._refresh()

    def stop_spinner(self, result_text: str = None) -> None:
        """Stop spinner and optionally show result."""
        self.status.spinner_active = False
        if result_text:
            self.status.current_operation = result_text
        self._refresh()

    def start_gitignore_loading(self) -> None:
        """Start gitignore loading phase."""
        self.status.current_operation = "Loading .gitignore patterns..."
        self.status.progress_current = 0
        self.status.progress_total = 0
        self._refresh()

    def update_gitignore_progress(self, path: str) -> None:
        """Update gitignore progress with current directory."""
        self.status.current_file = path
        self.status.progress_current += 1
        self._refresh()

    def end_gitignore_loading(self, pattern_count: int) -> None:
        """End gitignore loading."""
        self.status.current_file = None
        self.status.progress_total = pattern_count
        self.status.progress_current = pattern_count
        self._refresh()

    def start_scan(self, message: str = "Scanning files") -> None:
        """Start file scanning phase."""
        self.status.current_operation = message
        self.status.current_file = None
        self.status.progress_current = 0
        self.status.progress_total = 0
        self._refresh()

    def end_scan(self, file_count: int) -> None:
        """End scanning phase."""
        self.status.total_sources = file_count
        self.status.progress_total = file_count
        self.status.progress_current = file_count
        self._refresh()

    def start_file_loading(self, total: int) -> None:
        """Start file loading phase."""
        self.status.current_operation = "Loading files..."
        self.status.progress_total = total
        self.status.progress_current = 0
        self.status.current_file = "Starting..."  # Placeholder until first file
        self._refresh()

    def update_file_progress(self, current: int, total: int, filename: str) -> None:
        """Update file loading progress."""
        self.status.progress_current = current
        self.status.progress_total = total
        self.status.total_sources = current
        self.status.current_file = filename
        self._refresh()

    def end_file_loading(self) -> None:
        """End file loading phase."""
        self.status.current_operation = None
        self.status.current_file = None
        self._refresh()

    def start_entity_processing(self, total: int) -> None:
        """Start entity processing phase."""
        self.status.current_operation = "Processing entities..."
        self.status.progress_total = total
        self.status.progress_current = 0
        self._refresh()

    def update_entity_progress(self, current: int, total: int) -> None:
        """Update entity processing progress."""
        self.status.progress_current = current
        self.status.progress_total = total
        self.status.total_wmes = current
        self._refresh()

    def end_entity_processing(self) -> None:
        """End entity processing phase."""
        self.status.current_operation = None
        self.status.current_file = None
        self._refresh()

    def start_embedding_loading(self, model_name: str) -> None:
        """Start embedding model loading."""
        self.status.current_operation = f"Loading {model_name}..."
        self.status.current_file = None
        self.status.progress_current = 0
        self.status.progress_total = 0
        self._refresh()

    def end_embedding_loading(self) -> None:
        """End embedding model loading."""
        # Don't clear operation - next phase will set it
        self._refresh()

    def start_rag_indexing(self, total: int = 0) -> None:
        """Start RAG indexing phase."""
        self.status.current_operation = "Creating embeddings..."
        self.status.progress_total = total
        self.status.progress_current = 0
        self._refresh()

    def update_rag_progress(self, current: int, total: int) -> None:
        """Update RAG indexing progress."""
        self.status.progress_current = current
        self.status.progress_total = total
        self.status.total_vectors = current
        self._refresh()

    def end_rag_indexing(self, vectors_indexed: int) -> None:
        """End RAG indexing phase."""
        self.status.total_vectors = vectors_indexed
        self.status.current_operation = None
        self._refresh()

    def set_component_ready(self, name: str) -> None:
        """Mark component as ready."""
        pass  # Could add component status tracking

    def set_component_error(self, name: str) -> None:
        """Mark component as errored."""
        pass

    def mark_initialized(self) -> None:
        """Mark server as fully initialized and ready."""
        self.status.initialized = True
        self.status.current_operation = None
        self.status.current_file = None
        self.status.progress_current = 0
        self.status.progress_total = 0
        self._refresh()

    def _refresh(self) -> None:
        """Refresh the display with current state.

        With auto_refresh=True and __rich__(), Rich automatically rebuilds
        the layout 4 times per second. Manual refresh not needed.
        """
        pass


class NoOpConsoleUI:
    """No-op console UI when Rich is not available.

    ::: This is-in-layer Presentation-Layer.
    ::: This is a adapter.
    ::: This is stateless.
    """

    def __init__(self, server: Any):
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def log_query(self, *args, **kwargs) -> None:
        pass

    def update_status(self, *args, **kwargs) -> None:
        pass

    def update_progress(self, *args, **kwargs) -> None:
        pass

    # ConsoleProgress-compatible no-ops
    def set_phase(self, phase: str) -> None:
        pass

    def start_spinner(self, text: str) -> None:
        pass

    def stop_spinner(self, result_text: str = None) -> None:
        pass

    def start_gitignore_loading(self) -> None:
        pass

    def update_gitignore_progress(self, path: str) -> None:
        pass

    def end_gitignore_loading(self, pattern_count: int) -> None:
        pass

    def start_scan(self, message: str = "Scanning files") -> None:
        pass

    def end_scan(self, file_count: int) -> None:
        pass

    def start_file_loading(self, total: int) -> None:
        pass

    def update_file_progress(self, current: int, total: int, filename: str) -> None:
        pass

    def end_file_loading(self) -> None:
        pass

    def start_entity_processing(self, total: int) -> None:
        pass

    def update_entity_progress(self, current: int, total: int) -> None:
        pass

    def end_entity_processing(self) -> None:
        pass

    def start_embedding_loading(self, model_name: str) -> None:
        pass

    def end_embedding_loading(self) -> None:
        pass

    def start_rag_indexing(self, total: int = 0) -> None:
        pass

    def update_rag_progress(self, current: int, total: int) -> None:
        pass

    def end_rag_indexing(self, vectors_indexed: int) -> None:
        pass

    def set_component_ready(self, name: str) -> None:
        pass

    def set_component_error(self, name: str) -> None:
        pass

    def mark_initialized(self) -> None:
        pass

    # Log viewer no-ops
    def enter_log_view(self, mode: str) -> None:
        pass

    def exit_log_view(self) -> None:
        pass

    def scroll_log(self, delta: int) -> None:
        pass

    def scroll_log_home(self) -> None:
        pass

    def scroll_log_end(self) -> None:
        pass

    # Source tree no-op
    def enter_source_tree(self) -> None:
        pass

    @property
    def in_overlay_view(self) -> bool:
        return False


# Export appropriate class based on Rich availability
if RICH_AVAILABLE:
    __all__ = ["ConsoleUI", "QueryLogEntry", "ServerStatus"]
else:
    ConsoleUI = NoOpConsoleUI  # type: ignore
    __all__ = ["ConsoleUI"]
