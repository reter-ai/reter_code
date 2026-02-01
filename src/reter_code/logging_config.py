"""
Logging Configuration for RETER MCP Server.

Provides centralized logger setup for debug trace and NLQ debug logs.
All loggers write to files in the auto-detected snapshots directory.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Snapshot directory priority (same as state_persistence.py):
# 1. RETER_SNAPSHOTS_DIR (explicit)
# 2. RETER_PROJECT_ROOT/.reter_code (if set)
# 3. CWD/.reter_code (fallback)
# This ensures logs go to the same directory as .default.reter
def _get_log_directory() -> Path:
    """Get the log directory path (same as snapshots_dir in state_persistence)."""
    log_dir = os.getenv("RETER_SNAPSHOTS_DIR")
    if not log_dir:
        project_root = os.getenv("RETER_PROJECT_ROOT")
        if project_root:
            log_dir = str(Path(project_root) / ".reter_code")
        else:
            log_dir = str(Path.cwd() / ".reter_code")
    return Path(log_dir)


def _ensure_log_directory() -> Path:
    """Ensure log directory exists and return its path."""
    log_dir = _get_log_directory()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# Check if debug logging is enabled via environment variable
# Set RETER_DEBUG_LOG="" to disable
_debug_log_env = os.getenv("RETER_DEBUG_LOG")
DEBUG_LOG_ENABLED = _debug_log_env is None or _debug_log_env != ""


def _create_file_handler(log_filename: str) -> Optional[logging.FileHandler]:
    """
    Create a file handler for the specified log file.

    Args:
        log_filename: Name of the log file (e.g., 'debug_trace.log')

    Returns:
        Configured FileHandler, or None if logging is disabled
    """
    if not DEBUG_LOG_ENABLED:
        return None

    try:
        log_dir = _ensure_log_directory()
        log_path = log_dir / log_filename
        handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        return handler
    except Exception:
        return None


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit.

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a handler.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()


def _create_stderr_handler() -> logging.StreamHandler:
    """Create a stderr handler for console output with auto-flush."""
    handler = FlushingStreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    return handler


def get_debug_trace_logger() -> logging.Logger:
    """
    Get the debug trace logger for RETER wrapper operations.

    This logger is used for tracing C++ calls and debugging hangs.
    Output goes to .reter_code/debug_trace.log and stderr

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("reter.debug_trace")

    # Only configure once
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Don't propagate to root logger

        # File handler
        file_handler = _create_file_handler("debug_trace.log")
        if file_handler:
            logger.addHandler(file_handler)

        # Stderr handler
        stderr_handler = _create_stderr_handler()
        logger.addHandler(stderr_handler)

    return logger


def get_nlq_debug_logger() -> logging.Logger:
    """
    Get the NLQ debug logger for natural language query operations.

    This logger is used for tracing NLQ query generation and execution.
    Output goes to .reter_code/nlq_debug.log and stderr.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("reter.nlq_debug")

    # Only configure once
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Don't propagate to root logger

        # File handler
        file_handler = _create_file_handler("nlq_debug.log")
        if file_handler:
            logger.addHandler(file_handler)

        # Stderr handler for real-time visibility
        stderr_handler = _create_stderr_handler()
        logger.addHandler(stderr_handler)

    return logger


# Track configured log directory for reconfiguration
_configured_log_dir: Optional[Path] = None


def ensure_nlq_logger_configured() -> logging.Logger:
    """
    Ensure NLQ logger is configured with the correct directory.

    Call this before logging to ensure the logger points to the right directory,
    especially after RETER_PROJECT_ROOT is set.

    Returns:
        Configured logger instance
    """
    global _configured_log_dir

    current_dir = _get_log_directory()
    logger = logging.getLogger("reter.nlq_debug")

    # Reconfigure if directory changed or no handlers
    if _configured_log_dir != current_dir or not logger.handlers:
        # Remove old handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # File handler
        file_handler = _create_file_handler("nlq_debug.log")
        if file_handler:
            logger.addHandler(file_handler)

        # Stderr handler for real-time visibility
        stderr_handler = _create_stderr_handler()
        logger.addHandler(stderr_handler)

        _configured_log_dir = current_dir

    return logger


# Pre-create loggers for import convenience
debug_trace_logger = get_debug_trace_logger()
nlq_debug_logger = get_nlq_debug_logger()


def configure_logger_for_debug_trace(logger_name: str) -> logging.Logger:
    """
    Configure a logger to also write to debug_trace.log.

    Args:
        logger_name: Name of the logger to configure (e.g., __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Enable all log levels
    for handler in debug_trace_logger.handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)
    return logger


def suppress_stderr_logging():
    """
    Suppress stderr logging for all debug loggers.

    Call this when using Rich progress UI to avoid log spam in the console.
    File logging continues to work normally.
    """
    for logger in [debug_trace_logger, nlq_debug_logger]:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.CRITICAL + 1)  # Effectively disable


def restore_stderr_logging():
    """
    Restore stderr logging for all debug loggers.

    Call this after Rich progress UI is done.
    """
    for logger in [debug_trace_logger, nlq_debug_logger]:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG)
