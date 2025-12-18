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

# Snapshot directory priority:
# 1. RETER_SNAPSHOTS_DIR (explicit)
# 2. RETER_PROJECT_ROOT/.codeine (if set)
# 3. CWD/.codeine (auto-detection from Claude Code)
def _get_log_directory() -> Path:
    """Get the log directory path."""
    log_dir = os.getenv("RETER_SNAPSHOTS_DIR")
    if not log_dir:
        project_root = os.getenv("RETER_PROJECT_ROOT")
        if project_root:
            log_dir = os.path.join(project_root, ".codeine")
        else:
            log_dir = os.path.join(os.getcwd(), ".codeine")
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
    """StreamHandler that flushes after every emit."""
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
    Output goes to .codeine/debug_trace.log and stderr

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
    Output goes to .codeine/nlq_debug.log

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("reter.nlq_debug")

    # Only configure once
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Don't propagate to root logger

        handler = _create_file_handler("nlq_debug.log")
        if handler:
            logger.addHandler(handler)

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
