"""
Codeine MCP Server

AI-powered code reasoning server using RETER engine.
"""

# Import timing - measure how long module imports take
import time as _import_time
_import_start = _import_time.time()

import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from fastmcp import FastMCP  # Use fastmcp, not mcp.server.fastmcp, for proper Context injection

from .logging_config import configure_logger_for_debug_trace
logger = configure_logger_for_debug_trace(__name__)

# Sampling handler imports
from mcp.types import SamplingMessage, CreateMessageRequestParams, CreateMessageResult, TextContent

from .models import (
    LogicalThinkingInput,
    LogicalThinkingOutput,
    LogicalThought,
    LogicalSession,
    ThoughtType,
    AddKnowledgeOutput,
    QueryOutput,
    ForgetSourceOutput,
    StateOperationOutput,
    ConsistencyCheckOutput
)
from .reter_wrapper import ReterWrapper
from .services import (
    InstanceManager,
    DocumentationProvider,
    ReterOperations,
    StatePersistenceService,
    ResourceRegistrar,
    ToolRegistrar
)
from .services.default_instance_manager import DefaultInstanceManager
from .services.config_loader import load_config, get_config_loader
from .services.rag_index_manager import RAGIndexManager
from .services.initialization_progress import get_component_readiness

from sentence_transformers import SentenceTransformer

# Use stderr for import timing since logger might not be configured yet
print(f"[TIMING] Module imports completed in {_import_time.time() - _import_start:.3f}s", file=sys.stderr)


async def anthropic_sampling_handler(
    messages: List[SamplingMessage],
    params: CreateMessageRequestParams,
    context: Any
) -> CreateMessageResult:
    """
    Sampling handler that uses Anthropic's API.

    Environment variables:
        ANTHROPIC_API_KEY: Required. Your Anthropic API key.
        ANTHROPIC_MODEL_NAME: Optional. Model to use (default: claude-sonnet-4-20250514)
        ANTHROPIC_MAX_TOKENS: Optional. Max tokens for response (default: 1024)
    """
    try:
        import anthropic
    except ImportError:
        raise ValueError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Get configurable settings from environment
    model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-20250514")
    default_max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))

    client = anthropic.Anthropic(api_key=api_key)

    # Convert MCP messages to Anthropic format
    anthropic_messages = []
    for msg in messages:
        content = msg.content
        if hasattr(content, 'text'):
            text = content.text
        else:
            text = str(content)
        anthropic_messages.append({
            "role": msg.role,
            "content": text
        })

    # Call Anthropic API
    response = client.messages.create(
        model=model_name,
        max_tokens=params.maxTokens or default_max_tokens,
        system=params.systemPrompt or "",
        messages=anthropic_messages
    )

    # Extract response text
    response_text = response.content[0].text if response.content else ""

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model=response.model,
        stopReason=response.stop_reason
    )


class CodeineServer:
    """
    RETER Logical Thinking MCP Server with thread-safe RETER access.

    This class has been refactored from a God Class (62+ methods) into a
    clean orchestrator that delegates to specialized service classes:
    - ResourceRegistrar: Handles documentation resource registration
    - ToolRegistrar: Handles MCP tool registration (including all plugin tools)
    - InstanceManager: Manages RETER instances
    - DocumentationProvider: Provides documentation content
    - ReterOperations: Handles RETER operations
    - StatePersistenceService: Manages state persistence
    """

    def __init__(self):
        import time as _time
        _init_start = _time.time()

        # Load config from reter.json FIRST (before other services read env vars)
        # This sets env vars from config file if not already set
        load_config()
        logger.info("[TIMING] Config loaded in %.3fs", _time.time() - _init_start)

        # Service classes extracted as part of God Class refactoring (Fowler's "Extract Class" pattern)
        _t = _time.time()
        self.instance_manager = InstanceManager()
        self.doc_provider = DocumentationProvider()
        self.reter_ops = ReterOperations(self.instance_manager)
        self.persistence = StatePersistenceService(self.instance_manager)
        logger.info("[TIMING] Services created in %.3fs", _time.time() - _t)

        # Connect persistence service to instance manager for lazy loading
        self.instance_manager.set_persistence_service(self.persistence)

        # Initialize default instance manager
        _t = _time.time()
        self.default_manager = DefaultInstanceManager(self.persistence)
        self.instance_manager.set_default_manager(self.default_manager)
        logger.info("[TIMING] DefaultInstanceManager created in %.3fs", _time.time() - _t)

        # Initialize RAG index manager (if enabled)
        _t = _time.time()
        rag_config = get_config_loader().get_rag_config()
        self._rag_config = rag_config  # Store for background init
        if rag_config.get("rag_enabled", True):
            try:
                # Create RAG manager immediately (no model yet)
                self.rag_manager = RAGIndexManager(self.persistence, rag_config)
                self.default_manager.set_rag_manager(self.rag_manager, rag_config)
                logger.info("[TIMING] RAGIndexManager created in %.3fs", _time.time() - _t)
            except Exception as e:
                logger.warning("RAG initialization failed: %s", e)
                self.rag_manager = None
        else:
            self.rag_manager = None
            logger.info("RAG semantic search disabled via configuration")

        # Initialize registrars (Extract Class pattern - Fowler Ch. 7)
        # ToolRegistrar now directly registers all plugin tools (no complex plugin infrastructure)
        _t = _time.time()
        self.resource_registrar = ResourceRegistrar(self.doc_provider)
        self.tool_registrar = ToolRegistrar(
            self.reter_ops,
            self.persistence,
            self.instance_manager,
            self.default_manager
        )
        logger.info("[TIMING] Registrars created in %.3fs", _time.time() - _t)

        # Shutdown tracking to prevent duplicate state saves
        self._shutdown_complete = False

        # Initialize FastMCP with Anthropic sampling handler
        # Claude Code doesn't support MCP sampling, so we use Anthropic API as fallback
        _t = _time.time()
        self.app = FastMCP(
            "codeine",
            sampling_handler=anthropic_sampling_handler,
            sampling_handler_behavior="fallback"
        )
        logger.info("[TIMING] FastMCP created in %.3fs", _time.time() - _t)



    # All RETER operations, state persistence, and instance management methods
    # have been extracted to service classes (God Class refactoring)

    def _start_background_initialization(self):
        """
        Start background thread for combined initialization.

        This thread:
        1. Sets _initialization_in_progress = True (allows internal RETER access)
        2. Loads embedding model (if RAG enabled)
        3. Initializes default RETER instance (loads Python files)
        4. Initializes RAG index (indexes code for semantic search)
        5. Sets _initialization_complete = True, _initialization_in_progress = False

        All MCP tool calls are BLOCKED until this completes (via check_initialization()).
        """
        import threading
        import time

        from .reter_wrapper import (
            set_initialization_in_progress,
            set_initialization_complete,
            debug_log
        )
        from .services.initialization_progress import (
            get_instance_progress,
            get_component_readiness,
            InitStatus,
            InitPhase
        )

        def _background_init():
            progress = get_instance_progress()
            components = get_component_readiness()
            try:
                init_start = time.time()

                # SQLite/persistence is ready - mark component as ready
                components.set_sql_ready(True)
                logger.info("SQLite/persistence layer ready")

                # CRITICAL: Set flag to allow internal RETER access during init
                set_initialization_in_progress(True)
                progress.update(
                    init_status=InitStatus.INITIALIZING,
                    init_phase=InitPhase.PENDING,
                    init_started_at=datetime.now(),
                    init_message="Starting initialization..."
                )
                logger.info("Background initialization started...")

                # Step 1: Initialize default RETER instance (loads Python files)
                # This must happen FIRST so code_inspection tools work while embedding loads
                if self.default_manager.is_configured():
                    progress.update(
                        init_phase=InitPhase.LOADING_PYTHON,
                        init_message=f"Loading Python files from {self.default_manager._project_root}..."
                    )
                    logger.info("Loading Python files from %s...", self.default_manager._project_root)
                    reter_start = time.time()
                    try:
                        # This triggers full initialization including Python file loading
                        reter = self.instance_manager.get_or_create_instance("default")
                        if reter:
                            # Sync to load any new/changed files
                            rebuilt = self.default_manager.ensure_default_instance_synced(reter)
                            if rebuilt is not None:
                                # Instance was rebuilt from scratch to compact RETE network
                                # Update the instance manager's cache
                                self.instance_manager._instances["default"] = rebuilt
                                reter = rebuilt
                            # Mark RETER as ready - Python files are loaded
                            # code_inspection tools can now be used!
                            components.set_reter_ready(True)
                            logger.info("RETER instance initialized in %.1fs", time.time() - reter_start)
                    except Exception as e:
                        components.set_reter_ready(False, error=str(e))
                        logger.warning("RETER initialization failed: %s", e)
                        logger.exception("RETER initialization traceback")
                else:
                    # No default instance configured - mark as ready anyway
                    components.set_reter_ready(True)

                # Step 2: Load embedding model (if RAG enabled)
                # This can take time but doesn't block code_inspection tools
                if self.rag_manager is not None and self._rag_config.get("rag_enabled", True):
                    model_name = self._rag_config.get('rag_embedding_model', 'all-MiniLM-L6-v2')
                    progress.update(
                        init_phase=InitPhase.BUILDING_RAG_INDEX,
                        init_message=f"Loading embedding model '{model_name}'..."
                    )
                    logger.info("Loading embedding model '%s'...", model_name)
                    model_start = time.time()
                    try:
                        cache_dir = os.environ.get('TRANSFORMERS_CACHE', None)
                        preloaded_model = SentenceTransformer(model_name, cache_folder=cache_dir)
                        self.rag_manager.set_preloaded_model(preloaded_model)
                        # Mark embedding model as ready
                        components.set_embedding_ready(True)
                        logger.info("Embedding model loaded in %.1fs", time.time() - model_start)
                    except Exception as e:
                        logger.warning("Embedding model loading failed: %s", e)

                # Step 3: Initialize RAG index (if enabled and model loaded)
                if self.rag_manager is not None and self.rag_manager.is_model_loaded:
                    progress.update(
                        init_phase=InitPhase.BUILDING_RAG_INDEX,
                        init_message="Building RAG code index..."
                    )
                    logger.info("Building RAG index...")
                    rag_start = time.time()
                    try:
                        self.rag_manager.initialize(self.default_manager._project_root)
                        # Sync RAG with RETER to index all Python entities
                        if self.default_manager.is_configured():
                            reter = self.instance_manager.get_or_create_instance("default")
                            if reter:
                                self.rag_manager.sync_sources(
                                    reter=reter,
                                    project_root=self.default_manager._project_root
                                )
                        # Mark RAG code index as ready
                        components.set_rag_code_ready(True)
                        logger.info("RAG code index built in %.1fs", time.time() - rag_start)

                        # Step 4: Index Markdown documents
                        progress.update(
                            init_message="Indexing Markdown documents..."
                        )
                        # Markdown indexing happens during sync() above - mark as ready
                        components.set_rag_docs_ready(True)
                        logger.info("RAG document index ready")

                    except Exception as e:
                        components.set_rag_code_ready(False, error=str(e))
                        components.set_rag_docs_ready(False, error=str(e))
                        logger.warning("RAG indexing failed: %s", e)
                        logger.exception("RAG indexing traceback")
                else:
                    # RAG not enabled - mark as ready so tools don't block
                    components.set_rag_code_ready(True)
                    components.set_rag_docs_ready(True)

                # CRITICAL: Mark initialization as complete
                set_initialization_complete(True)
                set_initialization_in_progress(False)
                progress.update(
                    init_status=InitStatus.READY,
                    init_phase=InitPhase.COMPLETE,
                    init_progress=1.0,
                    init_message="Initialization complete",
                    init_completed_at=datetime.now()
                )

                total_time = time.time() - init_start
                logger.info("Background initialization complete in %.1fs - all tools ready", total_time)

            except Exception as e:
                logger.error("Background initialization failed: %s", e)
                logger.exception("Background initialization traceback")
                # Still mark as complete so tools don't hang forever
                set_initialization_complete(True)
                set_initialization_in_progress(False)
                progress.update(
                    init_status=InitStatus.ERROR,
                    init_error=str(e),
                    init_completed_at=datetime.now()
                )

        # Start the background thread
        logger.info("Background initialization thread started...")
        init_thread = threading.Thread(target=_background_init, daemon=True, name="reter-init")
        init_thread.start()

    def run(self):
        """
        Start the MCP server with graceful shutdown support.

        Handles SIGINT (Ctrl+C) and SIGTERM to ensure state is saved on exit.
        """
        def signal_handler(signum, frame):
            """Handle shutdown signals gracefully"""
            sig_name = signal.Signals(signum).name
            logger.warning("Received %s, initiating graceful shutdown...", sig_name)
            # Save all instances before exit
            if not self._shutdown_complete:
                logger.info("Saving all RETER instances...")
                self.persistence.save_all_instances()
                self._shutdown_complete = True
            sys.exit(0)

        # Register signal handlers (SIGINT = Ctrl+C, SIGTERM = kill command)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start server
        try:
            import time as _time
            _init_start = _time.time()

            logger.info("RETER Logical Thinking Server starting... (total: %.3fs)", _time.time() - _init_start)

            # Discover available snapshots (lazy loading - will load on first use)
            explicit_snapshots = os.getenv("RETER_SNAPSHOTS_DIR")
            if explicit_snapshots:
                logger.info("Snapshots directory: %s", self.persistence.snapshots_dir)
            else:
                logger.info("Snapshots directory (auto): %s", self.persistence.snapshots_dir)
            available = self.persistence.discover_snapshots()
            if available:
                logger.info("Discovered %d snapshot(s) (will load on first use)", len(available))

            # Report default instance configuration and start background initialization
            if self.default_manager.is_configured():
                explicit_root = os.getenv('RETER_PROJECT_ROOT')
                if explicit_root:
                    logger.info("Default instance configured: %s", self.default_manager._project_root)
                else:
                    logger.info("Default instance auto-detected from CWD: %s", self.default_manager._project_root)
                if self.default_manager._include_patterns:
                    logger.info("Including: %s", ', '.join(self.default_manager._include_patterns))
                if self.default_manager._exclude_patterns:
                    logger.info("Excluding: %s", ', '.join(self.default_manager._exclude_patterns))

                # Schedule background initialization to start 2 seconds after MCP server starts
                # This allows the MCP server to be responsive immediately
                import threading
                def delayed_init():
                    logger.info("Starting delayed background initialization...")
                    self._start_background_initialization()
                init_timer = threading.Timer(2.0, delayed_init)
                init_timer.daemon = True
                init_timer.start()
                logger.info("Background initialization scheduled (2s delay)")
            else:
                # No default instance - just mark as complete so tools work
                from .reter_wrapper import set_initialization_complete
                set_initialization_complete(True)
                logger.info("Default instance not configured (set RETER_PROJECT_ROOT or run from project dir)")

            # Use registrars to register resources and tools
            _t = _time.time()
            self.resource_registrar.register_all_resources(self.app)
            self.tool_registrar.register_all_tools(self.app)
            logger.info("[TIMING] Tools registered in %.3fs", _time.time() - _t)

            logger.info("All tools registered successfully (total init: %.3fs)", _time.time() - _init_start)

            # Run the FastMCP app (blocking)
            self.app.run()

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            logger.warning("Keyboard interrupt received, shutting down...")

        except Exception as e:
            # Log unexpected errors
            logger.error("Unexpected error during server operation: %s", e)
            raise

        finally:
            # Ensure cleanup happens even on unexpected exit
            if not self._shutdown_complete:
                try:
                    logger.info("Emergency save of RETER instances...")
                    # Save all instances synchronously
                    self.persistence.save_all_instances()
                except Exception as e:
                    try:
                        logger.error("Error during emergency save: %s", e)
                    except:
                        pass  # logger may be unavailable

            try:
                logger.info("Server shutdown complete")
            except:
                pass  # logger may be unavailable during MCP shutdown


def create_server():
    """Factory function to create server instance"""
    return CodeineServer()


def sync_only():
    """Sync default instance and exit (for pre-warming cache).

    Runs synchronously in the main thread - sets initialization flags
    to allow RETER operations without the background init thread.
    """
    import time
    from .reter_wrapper import set_initialization_in_progress, set_initialization_complete

    print("[codeine] Sync-only mode - syncing default instance...", file=sys.stderr)
    start = time.time()

    # CRITICAL: Set initialization flag to allow RETER operations in main thread
    # This bypasses the check_initialization() guard that normally blocks
    # until background init completes
    set_initialization_in_progress(True)

    try:
        # Initialize just enough to sync
        load_config()
        instance_manager = InstanceManager()
        persistence = StatePersistenceService(instance_manager)
        instance_manager.set_persistence_service(persistence)
        default_manager = DefaultInstanceManager(persistence)
        instance_manager.set_default_manager(default_manager)

        # Get or create the default instance and sync it
        if default_manager.is_configured():
            reter = instance_manager.get_or_create_instance("default")
            if reter:
                default_manager.ensure_default_instance_synced(reter)
                print(f"[codeine] Sync complete in {time.time() - start:.2f}s", file=sys.stderr)
            else:
                print("[codeine] Failed to create default instance", file=sys.stderr)
        else:
            print("[codeine] No project root configured (set RETER_PROJECT_ROOT or run from project dir)", file=sys.stderr)
    finally:
        # Mark initialization complete (sync finished)
        set_initialization_in_progress(False)
        set_initialization_complete(True)


def main():
    """Main entry point.

    Behavior:
    - If stdin is a TTY (interactive terminal): sync default instance and exit
    - If stdin is piped (MCP mode): run the MCP server normally
    """
    if sys.stdin.isatty():
        # Interactive mode - sync and exit
        sync_only()
    else:
        # MCP mode - run server
        server = create_server()
        server.run()


if __name__ == "__main__":
    main()