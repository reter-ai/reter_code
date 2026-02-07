"""
Logging Configuration for RETER MCP Server.

Provides centralized logger setup for debug trace and NLQ debug logs.
All loggers write ONLY to files — never to stdout/stderr.
MCP client uses stdout for protocol, server uses Rich console UI.
"""

import logging
import os
from pathlib import Path
from typing import Optional

# Snapshot directory priority (same as state_persistence.py):
# 1. RETER_SNAPSHOTS_DIR (explicit)
# 2. RETER_PROJECT_ROOT/.reter_code (if set)
# 3. None (don't create file handlers until project root is known)
def _get_log_directory() -> Optional[Path]:
    """Get the log directory path (same as snapshots_dir in state_persistence).

    Returns None if RETER_PROJECT_ROOT is not set, to avoid creating logs
    in the wrong directory during early module imports.
    """
    log_dir = os.getenv("RETER_SNAPSHOTS_DIR")
    if not log_dir:
        project_root = os.getenv("RETER_PROJECT_ROOT")
        if project_root:
            log_dir = str(Path(project_root) / ".reter_code")
        else:
            return None
    return Path(log_dir)


def _ensure_log_directory() -> Optional[Path]:
    """Ensure log directory exists and return its path."""
    log_dir = _get_log_directory()
    if log_dir is None:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# Check if debug logging is enabled via environment variable
# Set RETER_DEBUG_LOG="" to disable
_debug_log_env = os.getenv("RETER_DEBUG_LOG")
DEBUG_LOG_ENABLED = _debug_log_env is None or _debug_log_env != ""


def _create_file_handler(log_filename: str) -> Optional[logging.FileHandler]:
    """Create a file handler for the specified log file."""
    if not DEBUG_LOG_ENABLED:
        return None

    try:
        log_dir = _ensure_log_directory()
        if log_dir is None:
            return None
        log_path = log_dir / log_filename
        handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        return handler
    except Exception:
        return None


def _setup_logger(name: str, log_filename: str) -> logging.Logger:
    """Create a file-only logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        file_handler = _create_file_handler(log_filename)
        if file_handler:
            logger.addHandler(file_handler)
    return logger


# Track configured log directory for reconfiguration
_configured_log_dir: Optional[Path] = None

# Track loggers configured via configure_logger_for_debug_trace()
_configured_loggers: list = []


def get_debug_trace_logger() -> logging.Logger:
    """Get the debug trace logger. Writes to .reter_code/debug_trace.log only."""
    return _setup_logger("reter.debug_trace", "debug_trace.log")


def get_nlq_debug_logger() -> logging.Logger:
    """Get the NLQ debug logger. Writes to .reter_code/nlq_debug.log only."""
    return _setup_logger("reter.nlq_debug", "nlq_debug.log")


def ensure_nlq_logger_configured() -> logging.Logger:
    """Ensure NLQ logger is configured with the correct directory."""
    global _configured_log_dir

    current_dir = _get_log_directory()
    logger = logging.getLogger("reter.nlq_debug")

    if _configured_log_dir != current_dir or not logger.handlers:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        file_handler = _create_file_handler("nlq_debug.log")
        if file_handler:
            logger.addHandler(file_handler)

        _configured_log_dir = current_dir

    return logger


# Pre-create loggers for import convenience
debug_trace_logger = get_debug_trace_logger()
nlq_debug_logger = get_nlq_debug_logger()


def reconfigure_log_directory() -> None:
    """Reconfigure all loggers to use the correct directory.

    Call this after setting RETER_PROJECT_ROOT to ensure logs go to the right place.
    """
    global _configured_log_dir, debug_trace_logger, nlq_debug_logger

    current_dir = _get_log_directory()
    if current_dir is None:
        return
    if _configured_log_dir == current_dir:
        return

    _configured_log_dir = current_dir

    # Reconfigure debug_trace_logger
    for handler in debug_trace_logger.handlers[:]:
        handler.close()
        debug_trace_logger.removeHandler(handler)
    file_handler = _create_file_handler("debug_trace.log")
    if file_handler:
        debug_trace_logger.addHandler(file_handler)

    # Reconfigure nlq_debug_logger
    for handler in nlq_debug_logger.handlers[:]:
        handler.close()
        nlq_debug_logger.removeHandler(handler)
    file_handler = _create_file_handler("nlq_debug.log")
    if file_handler:
        nlq_debug_logger.addHandler(file_handler)

    # Update all loggers configured via configure_logger_for_debug_trace()
    debug_file_handler = None
    for handler in debug_trace_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            debug_file_handler = handler
            break

    if debug_file_handler:
        for logger in _configured_loggers:
            has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
            if not has_file_handler:
                logger.addHandler(debug_file_handler)


def configure_logger_for_debug_trace(logger_name: str) -> logging.Logger:
    """Configure a logger to write to debug_trace.log (file only)."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in debug_trace_logger.handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)
    if logger not in _configured_loggers:
        _configured_loggers.append(logger)
    return logger


def configure_logger_for_nlq_debug(logger_name: str) -> logging.Logger:
    """Configure a logger to write to nlq_debug.log (file only)."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in nlq_debug_logger.handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)
    if logger not in _configured_loggers:
        _configured_loggers.append(logger)
    return logger
