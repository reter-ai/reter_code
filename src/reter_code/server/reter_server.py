"""
RETER ZeroMQ Server.

Standalone RETER server process with ZeroMQ interface and rich console output.

::: This is-in-layer Server-Layer.
::: This is-in-component ZeroMQ-Server.
::: This depends-on pyzmq.
::: This depends-on rich.
"""

# Disable tqdm progress bars from sentence_transformers and other libraries
# We use our own console UI for progress display
import os
os.environ["TQDM_DISABLE"] = "1"

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import zmq

from .protocol import (
    ReterMessage,
    ReterError,
    serialize,
    deserialize,
    INTERNAL_ERROR,
    PARSE_ERROR,
)
from .config import ServerConfig, ServerDiscovery
from .handlers import HandlerContext, HandlerRegistry, create_handler_registry

# Configure logger to write to debug_trace.log
from ..logging_config import configure_logger_for_debug_trace
logger = configure_logger_for_debug_trace(__name__)


class ReterServer:
    """Standalone RETER server with ZeroMQ interface.

    ::: This is-in-layer Presentation-Layer.
    ::: This is a service.
    ::: This is stateful.
    ::: This holds-expensive-resource "rete-network".
    ::: This holds-expensive-resource "embedding-model".
    ::: This has-singleton-scope.

    The server:
    1. Binds to ZeroMQ REQ/REP socket for queries
    2. Binds to ZeroMQ PUB socket for events
    3. Routes requests to appropriate handlers
    4. Displays rich console output (optional)
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """Initialize RETER server.

        Args:
            config: Server configuration. If None, uses defaults.
        """
        self.config = config or ServerConfig()
        self._running = False
        self._start_time = time.time()

        # ZeroMQ context and sockets
        self._context: Optional[zmq.Context] = None
        self._query_socket: Optional[zmq.Socket] = None
        self._event_socket: Optional[zmq.Socket] = None

        # Actual bound ports (may differ from config if using dynamic ports)
        self._actual_query_port: int = 0
        self._actual_event_port: int = 0

        # RETER components (lazy initialized)
        self._reter = None
        self._rag_manager = None
        self._instance_manager = None

        # Handler registry
        self._handler_registry: Optional[HandlerRegistry] = None

        # Console UI (optional)
        self._console = None

        # Discovery info
        self._discovery: Optional[ServerDiscovery] = None

        # Statistics
        self._stats = {
            "requests_handled": 0,
            "errors": 0,
            "total_time_ms": 0,
        }

    def _init_reter(self) -> None:
        """Initialize RETER components."""
        logger.info("Initializing RETER components...")

        # Import here to avoid circular imports
        from ..reter_wrapper import ReterWrapper
        from ..services.rag_index_manager import RAGIndexManager
        from ..services.instance_manager import InstanceManager
        from ..services.state_persistence import StatePersistenceService
        from ..services.default_instance_manager import DefaultInstanceManager
        from ..services.config_loader import load_config, get_config_loader
        from ..reter_utils import set_initialization_in_progress, set_initialization_complete

        # Load config from reter_code.json FIRST (before other services read env vars)
        # This sets env vars from config file if not already set
        load_config()

        # Mark initialization in progress (allows internal code to access RETER)
        set_initialization_in_progress(True)

        try:
            # Initialize in correct order (respecting dependencies)
            # Pass console as progress_callback for live UI updates
            self._instance_manager = InstanceManager()
            self._persistence = StatePersistenceService(self._instance_manager)
            self._default_manager = DefaultInstanceManager(
                self._persistence,
                progress_callback=self._console
            )

            # Link components together
            self._instance_manager.set_persistence_service(self._persistence)
            self._instance_manager.set_default_manager(self._default_manager)

            # Get the default RETER instance with progress callback
            self._reter = self._instance_manager.get_or_create_instance(
                "default",
                progress_callback=self._console
            )
            self._rag_manager = RAGIndexManager(self._persistence)

            # Link RAG manager to default instance manager for sync
            self._default_manager.set_rag_manager(self._rag_manager)

            # Initialize RAG index after RETER is loaded
            from ..reter_wrapper import debug_log
            debug_log(f"[RAG-INIT] RAG manager is_enabled: {self._rag_manager.is_enabled}")
            if self._rag_manager.is_enabled:
                # Load embedding model first (required before sync_sources)
                from ..services.config_loader import get_config_loader
                rag_config = get_config_loader().get_rag_config()
                model_name = rag_config.get(
                    "rag_embedding_model",
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                debug_log(f"[RAG-INIT] Loading embedding model: {model_name}")
                if self._console:
                    self._console.start_embedding_loading(model_name)

                try:
                    # Load the SentenceTransformer model
                    import os
                    from sentence_transformers import SentenceTransformer
                    cache_dir = os.environ.get('TRANSFORMERS_CACHE', None)
                    debug_log(f"[RAG-INIT] Creating SentenceTransformer...")
                    preloaded_model = SentenceTransformer(model_name, cache_folder=cache_dir)
                    debug_log(f"[RAG-INIT] Calling set_preloaded_model...")
                    self._rag_manager.set_preloaded_model(preloaded_model)
                    debug_log("[RAG-INIT] Embedding model loaded successfully")

                    if self._console:
                        self._console.end_embedding_loading()
                        self._console.start_rag_indexing()

                    # Now sync sources (model is loaded)
                    debug_log("[RAG-INIT] Starting RAG sync_sources...")

                    def rag_progress_callback(current: int, total: int, phase: str):
                        if self._console:
                            # Show phase in operation text
                            phase_text = {
                                "generating_embeddings": "Generating embeddings",
                                "embeddings_complete": "Embeddings complete",
                                "indexing": "Indexing vectors",
                            }.get(phase, phase.replace("_", " ").title())
                            self._console.update_progress(phase_text, current, total)

                    rag_stats = self._rag_manager.sync_sources(
                        reter=self._reter,
                        project_root=self._default_manager.project_root,
                        progress_callback=rag_progress_callback,
                    )
                    debug_log(f"[RAG-INIT] RAG sync complete: {rag_stats}")

                    # Mark RAG components as ready
                    from ..services.initialization_progress import get_component_readiness
                    get_component_readiness().set_rag_code_ready(True)
                    get_component_readiness().set_rag_docs_ready(True)
                    debug_log("[RAG-INIT] RAG component readiness flags set")

                    if self._console:
                        self._console.end_rag_indexing(rag_stats.get('total_vectors', 0))
                except Exception as e:
                    import traceback
                    debug_log(f"[RAG-INIT] ERROR: {e}\n{traceback.format_exc()}")
                    if self._console:
                        self._console.end_rag_indexing(0)
            else:
                debug_log("[RAG-INIT] RAG is disabled, skipping initialization")

            # Mark initialization complete
            set_initialization_complete(True)
        finally:
            # Clear in-progress flag
            set_initialization_in_progress(False)

        # Create handler context and registry
        context = HandlerContext(
            reter=self._reter,
            rag_manager=self._rag_manager,
            instance_manager=self._instance_manager,
            event_publisher=self._publish_event
        )
        self._handler_registry = create_handler_registry(context)

        logger.info("RETER components initialized")

    def _init_zmq(self) -> None:
        """Initialize ZeroMQ sockets."""
        logger.info("Initializing ZeroMQ sockets...")

        self._context = zmq.Context()

        # Query socket (REQ/REP pattern)
        self._query_socket = self._context.socket(zmq.REP)

        # Bind to configured endpoint
        if self.config.query_port == 0:
            # Dynamic port allocation
            self._query_socket.bind("tcp://*:0")
            endpoint = self._query_socket.getsockopt_string(zmq.LAST_ENDPOINT)
            self._actual_query_port = int(endpoint.split(":")[-1])
        else:
            self._query_socket.bind(self.config.query_bind_endpoint)
            self._actual_query_port = self.config.query_port

        # Event socket (PUB/SUB pattern)
        self._event_socket = self._context.socket(zmq.PUB)

        if self.config.event_port == 0:
            # Dynamic port allocation
            self._event_socket.bind("tcp://*:0")
            endpoint = self._event_socket.getsockopt_string(zmq.LAST_ENDPOINT)
            self._actual_event_port = int(endpoint.split(":")[-1])
        else:
            self._event_socket.bind(self.config.event_bind_endpoint)
            self._actual_event_port = self.config.event_port

        logger.info(f"Query socket bound to tcp://*:{self._actual_query_port}")
        logger.info(f"Event socket bound to tcp://*:{self._actual_event_port}")

    def _init_console(self) -> None:
        """Initialize console UI if enabled and running in a TTY."""
        if not self.config.console_enabled:
            return

        # Check if we're in a TTY - rich console only works in interactive terminals
        if not (sys.stdout.isatty() and sys.stderr.isatty()):
            return

        try:
            from .console_ui import ConsoleUI

            # Suppress ALL stderr logging to prevent interference with rich console
            # This must happen before console starts
            try:
                from ..logging_config import suppress_stderr_logging
                suppress_stderr_logging()
            except ImportError:
                pass

            # Also suppress root logger stderr output
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                    root_logger.removeHandler(handler)

            self._console = ConsoleUI(self)
        except ImportError:
            pass  # Rich not available

    def _write_discovery(self) -> None:
        """Write discovery file for clients."""
        if self.config.project_root:
            from .config import DISCOVERY_DIR, DISCOVERY_FILE
            self._discovery = self.config.write_discovery(
                self._actual_query_port,
                self._actual_event_port
            )
            logger.info(f"Discovery file written to {self.config.project_root / DISCOVERY_DIR / DISCOVERY_FILE}")

    def _cleanup_discovery(self) -> None:
        """Remove discovery file on shutdown."""
        if self.config.project_root:
            self.config.cleanup_discovery()
            logger.info("Discovery file removed")

    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish event via PUB socket.

        Args:
            event_type: Type of event (progress, file_loaded, etc.)
            data: Event data
        """
        if self._event_socket:
            message = ReterMessage.event(event_type, data)
            self._event_socket.send(serialize(message))

    def _handle_request(self, raw_message: bytes) -> bytes:
        """Handle incoming request and return response.

        Args:
            raw_message: Raw message bytes

        Returns:
            Serialized response message
        """
        start_time = time.time()

        try:
            # Deserialize request
            request = deserialize(raw_message)
        except Exception as e:
            logger.error(f"Failed to parse request: {e}")
            response = ReterMessage.response(
                "",
                error=ReterError(PARSE_ERROR, f"Failed to parse request: {e}")
            )
            return serialize(response)

        try:
            # Log request
            if self.config.console_log_queries:
                logger.debug(f"Request: {request.method} {request.params}")

            # Route to handler
            response = self._handler_registry.handle(request)

            # Update stats
            self._stats["requests_handled"] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self._stats["total_time_ms"] += elapsed_ms

            # Log to console
            if self._console:
                result_count = 0
                if response.result and isinstance(response.result, dict):
                    result_count = response.result.get("count", 0)
                self._console.log_query(request.method, elapsed_ms, result_count)

            return serialize(response)

        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            self._stats["errors"] += 1

            response = ReterMessage.response(
                request.id,
                error=ReterError.from_exception(e)
            )
            return serialize(response)

    def _update_console_stats(self) -> None:
        """Update console with current RETER stats."""
        if not self._console or not self._reter:
            return

        try:
            # Get sources count
            sources, _ = self._reter.get_all_sources()
            total_sources = len(sources) if sources else 0

            # Get facts/WMEs count - use hybrid stats if available
            total_wmes = 0
            if self._reter.is_hybrid_mode():
                stats = self._reter.get_hybrid_stats()
                total_wmes = stats.get("base_facts", 0) + stats.get("delta_facts", 0)
            else:
                # Fallback to network stats
                total_wmes = self._reter.reasoner.network.fact_count()

            # Get RAG vectors count
            total_vectors = 0
            if self._rag_manager and self._rag_manager.is_initialized:
                rag_stats = self._rag_manager.get_status()
                total_vectors = rag_stats.get("total_vectors", 0)

            # Update console
            self._console.update_status(
                total_sources=total_sources,
                total_wmes=total_wmes,
                total_vectors=total_vectors
            )
        except Exception as e:
            logger.debug(f"Error updating console stats: {e}")

    def _handle_keyboard(self) -> None:
        """Handle keyboard input for manual actions."""
        try:
            import msvcrt  # Windows-specific
            if msvcrt.kbhit():
                key = msvcrt.getch()
                logger.debug(f"[Keyboard] Key pressed: {key}")
                if key in (b'k', b'K'):
                    # Manual compaction
                    logger.info("[Keyboard] K pressed - triggering compaction")
                    self._trigger_compaction()
                elif key in (b'r', b'R'):
                    # Refresh stats
                    logger.info("[Keyboard] R pressed - refreshing stats")
                    self._update_console_stats()
        except ImportError:
            # Not on Windows, keyboard handling not available
            pass
        except Exception as e:
            logger.debug(f"[Keyboard] Error: {e}")

    def _trigger_compaction(self) -> None:
        """Trigger manual compaction in background with progress."""
        import threading

        logger.info("[Compaction] _trigger_compaction called")
        if not self._reter:
            logger.info("[Compaction] No reter instance")
            return
        if not self._reter.is_hybrid_mode():
            logger.info("[Compaction] Not in hybrid mode")
            if self._console:
                self._console.set_phase("Not in hybrid mode")
            return

        # Check if already compacting
        if hasattr(self, '_compacting') and self._compacting:
            logger.info("[Compaction] Already compacting")
            return

        self._compacting = True
        logger.info("[Compaction] Starting compaction...")

        if self._console:
            self._console.update_progress("Compacting", 0, 100)

        def progress_callback(percent: int) -> None:
            """Called by C++ at 1% intervals."""
            if self._console:
                self._console.update_progress("Compacting", percent, 100)

        def do_compact():
            try:
                success, time_ms = self._reter.compact(progress_callback)
                logger.info(f"[Compaction] Completed: success={success}, time={time_ms:.0f}ms")
                if self._console:
                    self._console.update_progress(None)  # Clear progress
                    self._console.set_phase(f"Compacted in {time_ms:.0f}ms")
                    self._update_console_stats()
                    # Clear phase after a moment
                    def clear_phase():
                        import time
                        time.sleep(2)
                        if self._console:
                            self._console.mark_initialized()
                    threading.Thread(target=clear_phase, daemon=True).start()
            except Exception as e:
                logger.error(f"[Compaction] Failed: {e}")
                if self._console:
                    self._console.update_progress(None)  # Clear progress
                    self._console.set_phase(f"Compact failed: {e}")
            finally:
                self._compacting = False

        threading.Thread(target=do_compact, daemon=True).start()

    def _run_event_loop(self) -> None:
        """Main event loop processing requests."""
        logger.info("Starting event loop...")

        poller = zmq.Poller()
        poller.register(self._query_socket, zmq.POLLIN)

        while self._running:
            try:
                # Poll with timeout for graceful shutdown
                events = dict(poller.poll(timeout=100))

                if self._query_socket in events:
                    raw_message = self._query_socket.recv()
                    response = self._handle_request(raw_message)
                    self._query_socket.send(response)

                # Handle keyboard input (Windows only)
                if self._console:
                    self._handle_keyboard()

            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break  # Context terminated
                logger.error(f"ZMQ error: {e}")

            except Exception as e:
                logger.exception(f"Event loop error: {e}")

        logger.info("Event loop stopped")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start(self) -> None:
        """Start the RETER server."""
        # Initialize and START console FIRST so we can show progress during init
        self._init_console()
        if self._console:
            self._console.start()

        try:
            # Initialize components - progress callbacks update console automatically
            self._init_reter()
            self._init_zmq()
            self._write_discovery()
            self._setup_signal_handlers()

            self._running = True

            # Mark initialization complete and update stats
            if self._console:
                self._console.mark_initialized()
                # Update stats from loaded RETER instance
                self._update_console_stats()

            # Run event loop with keyboard handling
            self._run_event_loop()

        except Exception as e:
            logger.exception(f"Failed to start server: {e}")
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the RETER server."""
        if not self._running:
            return

        self._running = False

        # Stop console UI and restore stderr logging
        if self._console:
            self._console.stop()
            try:
                from ..logging_config import restore_stderr_logging
                restore_stderr_logging()
            except ImportError:
                pass

        logger.info("Stopping RETER server...")

        # Cleanup discovery file
        self._cleanup_discovery()

        # Close ZeroMQ sockets
        if self._query_socket:
            self._query_socket.close()
        if self._event_socket:
            self._event_socket.close()
        if self._context:
            self._context.term()

        logger.info("RETER server stopped")

    @property
    def query_endpoint(self) -> str:
        """Get the actual query endpoint."""
        return f"tcp://127.0.0.1:{self._actual_query_port}"

    @property
    def event_endpoint(self) -> str:
        """Get the actual event endpoint."""
        return f"tcp://127.0.0.1:{self._actual_event_port}"

    @property
    def stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        uptime = time.time() - self._start_time
        avg_time = 0
        if self._stats["requests_handled"] > 0:
            avg_time = self._stats["total_time_ms"] / self._stats["requests_handled"]

        return {
            **self._stats,
            "uptime_seconds": uptime,
            "avg_request_time_ms": avg_time,
        }


def main():
    """Main entry point for RETER server."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="RETER ZeroMQ Server")
    parser.add_argument("--project", "-p", type=Path, help="Project root directory")
    parser.add_argument("--port", type=int, help="Query socket port (0 for dynamic)")
    parser.add_argument("--no-console", action="store_true", help="Disable console UI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Set project root env var EARLY so all components use correct path
    if args.project:
        os.environ["RETER_PROJECT_ROOT"] = str(args.project.resolve())
        # Reconfigure loggers to use the correct directory
        try:
            from ..logging_config import reconfigure_log_directory
            reconfigure_log_directory()
        except ImportError:
            pass

    # Determine if we'll use rich console (TTY detection)
    is_tty = sys.stdout.isatty() and sys.stderr.isatty()
    use_console = not args.no_console and is_tty

    # Configure logging - only add stderr handler if NOT using rich console
    # This prevents log duplication when rich console is active
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Clear any existing handlers on root logger to prevent duplication
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    if use_console:
        # Rich console mode - suppress ALL stderr logging immediately
        try:
            from ..logging_config import suppress_stderr_logging
            suppress_stderr_logging()
        except ImportError:
            pass
    else:
        # No rich console - use standard logging to stderr
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        root_logger.addHandler(handler)

    # Build config
    if args.project:
        config = ServerConfig.for_project(args.project)
    else:
        config = ServerConfig()

    if args.port is not None:
        config.query_port = args.port
        config.event_port = args.port + 1

    if args.no_console:
        config.console_enabled = False

    # Start server
    server = ReterServer(config)
    server.start()


if __name__ == "__main__":
    main()
