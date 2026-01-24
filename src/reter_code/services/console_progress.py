"""
Console Progress Display for Reter Code Startup

Rich-based console UI showing multiple progress bars and active task status.
Used by sync_only() when running in interactive (TTY) mode.
"""

import io
import sys
import time
from typing import Optional, Callable, TextIO
from rich.console import Console, Group
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text


class StderrSuppressor:
    """Context manager to suppress stderr output during progress display."""

    def __init__(self):
        self._original_stderr: Optional[TextIO] = None
        self._devnull: Optional[TextIO] = None

    def suppress(self):
        """Start suppressing stderr."""
        self._original_stderr = sys.stderr
        self._devnull = io.StringIO()
        sys.stderr = self._devnull

    def restore(self):
        """Restore original stderr."""
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
            self._original_stderr = None
        if self._devnull is not None:
            self._devnull.close()
            self._devnull = None


class ConsoleProgress:
    """Rich-based console progress display for sync operations."""

    def __init__(self):
        self.console = Console()
        self._live: Optional[Live] = None
        self._start_time = time.time()
        self._stderr_suppressor = StderrSuppressor()

        # Progress bars
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=False,
        )

        # Task IDs
        self._scan_task = None
        self._file_task = None
        self._entity_task = None
        self._embedding_task = None
        self._rag_task = None

        # Current status
        self._current_file = ""
        self._phase = "Starting..."
        self._components = {
            "SQLite": "pending",
            "RETER": "pending",
            "Embedding": "pending",
            "RAG": "pending",
        }

        # Stats
        self._files_total = 0
        self._files_loaded = 0
        self._entities_total = 0
        self._entities_processed = 0
        self._vectors_indexed = 0

    def start(self):
        """Start the live display and suppress stderr."""
        self._start_time = time.time()
        # Suppress stderr to prevent log spam from interfering with progress UI
        self._stderr_suppressor.suppress()
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def stop(self, success: bool = True):
        """Stop the live display and restore stderr."""
        if self._live:
            self._live.stop()
            self._live = None

        # Restore stderr before printing final message
        self._stderr_suppressor.restore()

        elapsed = time.time() - self._start_time

        if success:
            self.console.print()
            self.console.print(
                Panel(
                    f"[bold green]Initialization complete[/bold green]\n\n"
                    f"  Files loaded: {self._files_loaded}\n"
                    f"  Entities: {self._entities_processed:,}\n"
                    f"  Vectors indexed: {self._vectors_indexed:,}\n"
                    f"  Time: {elapsed:.1f}s",
                    title="[bold]Reter Code Ready[/bold]",
                    border_style="green",
                )
            )
        else:
            self.console.print()
            self.console.print(
                Panel(
                    f"[bold red]Initialization failed[/bold red]\n\n"
                    f"  Time: {elapsed:.1f}s",
                    title="[bold]Error[/bold]",
                    border_style="red",
                )
            )

    def set_phase(self, phase: str):
        """Set the current phase description."""
        self._phase = phase
        self._refresh()

    def start_gitignore_loading(self):
        """Start gitignore loading phase."""
        self._phase = "Loading gitignore patterns"
        self._gitignore_task = self._progress.add_task(
            "[cyan]Loading .gitignore", total=None  # Indeterminate spinner
        )
        self._refresh()

    def update_gitignore_progress(self, path: str):
        """Update current gitignore file being loaded."""
        self._current_file = path
        self._refresh()

    def end_gitignore_loading(self, pattern_count: int):
        """End gitignore loading phase."""
        if hasattr(self, '_gitignore_task') and self._gitignore_task is not None:
            # Mark as complete instead of removing
            self._progress.update(
                self._gitignore_task,
                completed=pattern_count,
                total=pattern_count,
                description="[green]Gitignore loaded"
            )
        self._current_file = ""
        self._refresh()

    def start_scan(self, message: str = "Scanning files"):
        """Start the file scanning phase."""
        self._phase = message
        self._scan_task = self._progress.add_task(
            "[cyan]Scanning", total=None  # Indeterminate
        )
        self._refresh()

    def end_scan(self, file_count: int):
        """End scanning and mark as complete."""
        if self._scan_task is not None:
            # Mark as complete instead of removing
            self._progress.update(
                self._scan_task,
                completed=file_count,
                total=file_count,
                description="[green]Files scanned"
            )
        self._files_total = file_count
        self._refresh()

    def start_file_loading(self, total: int):
        """Start the file loading phase."""
        self._phase = "Loading code files"
        self._files_total = total
        self._files_loaded = 0
        self._file_task = self._progress.add_task(
            "[cyan]Loading files", total=total
        )
        self._refresh()

    def update_file_progress(self, current: int, total: int, filename: str):
        """Update file loading progress (callback for RETER)."""
        self._files_loaded = current
        self._files_total = total
        self._current_file = filename

        if self._file_task is None:
            self._file_task = self._progress.add_task(
                "[cyan]Loading files", total=total
            )

        self._progress.update(self._file_task, completed=current, total=total)
        self._refresh()

    def end_file_loading(self):
        """Mark file loading as complete."""
        if self._file_task is not None:
            self._progress.update(
                self._file_task,
                completed=self._files_total,
                description="[green]Files loaded"
            )
        self._current_file = ""
        self._refresh()

    def start_entity_processing(self, total: int):
        """Start entity accumulation phase."""
        self._phase = "Processing entities"
        self._entities_total = total
        self._entities_processed = 0
        self._entity_task = self._progress.add_task(
            "[cyan]Processing entities", total=total
        )
        self._refresh()

    def update_entity_progress(self, current: int, total: int):
        """Update entity processing progress."""
        self._entities_processed = current
        self._entities_total = total

        if self._entity_task is None:
            self._entity_task = self._progress.add_task(
                "[cyan]Processing entities", total=total
            )

        self._progress.update(self._entity_task, completed=current, total=total)
        self._refresh()

    def end_entity_processing(self):
        """Mark entity processing as complete."""
        if self._entity_task is not None:
            self._progress.update(
                self._entity_task,
                completed=self._entities_total,
                description="[green]Entities processed"
            )
        self._refresh()

    def start_embedding_loading(self, model_name: str):
        """Start embedding model loading phase."""
        self._phase = f"Loading embedding model ({model_name})"
        self._embedding_task = self._progress.add_task(
            f"[cyan]Loading {model_name}", total=None  # Indeterminate
        )
        self._refresh()

    def end_embedding_loading(self):
        """Mark embedding model as loaded."""
        if self._embedding_task is not None:
            # Mark as complete instead of removing
            self._progress.update(
                self._embedding_task,
                completed=1,
                total=1,
                description="[green]Model loaded"
            )
        self.set_component_ready("Embedding")
        self._refresh()

    def start_rag_indexing(self, total: int = 0):
        """Start RAG indexing phase."""
        self._phase = "Building RAG index"
        if total > 0:
            self._rag_task = self._progress.add_task(
                "[cyan]Creating embeddings", total=total
            )
        else:
            self._rag_task = self._progress.add_task(
                "[cyan]Creating embeddings", total=None  # Indeterminate
            )
        self._refresh()

    def update_rag_progress(self, current: int, total: int):
        """Update RAG indexing progress."""
        self._vectors_indexed = current

        if self._rag_task is None:
            self._rag_task = self._progress.add_task(
                "[cyan]Creating embeddings", total=total
            )

        self._progress.update(self._rag_task, completed=current, total=total)
        self._refresh()

    def end_rag_indexing(self, vectors_indexed: int):
        """Mark RAG indexing as complete."""
        self._vectors_indexed = vectors_indexed
        if self._rag_task is not None:
            self._progress.update(
                self._rag_task,
                completed=vectors_indexed,
                total=vectors_indexed,
                description="[green]Embeddings created"
            )
        self.set_component_ready("RAG")
        self._refresh()

    def set_component_ready(self, name: str):
        """Mark a component as ready."""
        if name in self._components:
            self._components[name] = "ready"
            self._refresh()

    def set_component_error(self, name: str):
        """Mark a component as errored."""
        if name in self._components:
            self._components[name] = "error"
            self._refresh()

    def _build_display(self) -> Group:
        """Build the complete display layout."""
        elements = []

        # Header
        header = Panel(
            "[bold white]Reter Code[/bold white] - AI Code Reasoning Server",
            border_style="blue",
            padding=(0, 1),
        )
        elements.append(header)
        elements.append("")

        # Phase indicator
        elements.append(Text(f"  {self._phase}", style="bold"))
        elements.append("")

        # Progress bars
        elements.append(self._progress)

        # Current file being processed
        if self._current_file:
            truncated = self._current_file
            if len(truncated) > 60:
                truncated = "..." + truncated[-57:]
            elements.append("")
            elements.append(Text.from_markup(f"  [dim]\u2192 {truncated}[/dim]"))

        elements.append("")

        # Component status
        components_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            expand=False,
        )
        components_table.add_column("status", width=3)
        components_table.add_column("name", width=12)

        row_items = []
        for name, status in self._components.items():
            if status == "ready":
                icon = "[green]\u2713[/green]"
            elif status == "error":
                icon = "[red]\u2717[/red]"
            else:
                icon = "[dim]\u25cb[/dim]"
            row_items.extend([icon, name])

        # Add all components in one row
        if row_items:
            components_table.add_row(*row_items[:8])  # Max 4 components

        comp_panel = Panel(
            components_table,
            title="Components",
            border_style="dim",
            padding=(0, 1),
        )
        elements.append(comp_panel)

        return Group(*elements)

    def _refresh(self):
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())


def create_progress_callback(progress: ConsoleProgress) -> Callable[[int, int, str], None]:
    """Create a progress callback function for RETER file loading."""
    def callback(current: int, total: int, filename: str):
        progress.update_file_progress(current, total, filename)
    return callback


def create_entity_callback(progress: ConsoleProgress) -> Callable[[int, int], None]:
    """Create a progress callback function for entity processing."""
    def callback(current: int, total: int):
        progress.update_entity_progress(current, total)
    return callback
