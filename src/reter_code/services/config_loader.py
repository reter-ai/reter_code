"""
Configuration Loader Service

Loads RETER configuration from reter_code.json in the project root.
Environment variables always take precedence over config file values.

Config file location (in order of precedence):
1. RETER_PROJECT_ROOT/reter_code.json (if RETER_PROJECT_ROOT is set)
2. CWD/reter_code.json (auto-detection from Claude Code)

Supported settings in reter_code.json:
{
    "project_include": "src/*,lib/*",      // -> RETER_PROJECT_INCLUDE
    "project_exclude": "test_*.py,**/__pycache__/*",  // -> RETER_PROJECT_EXCLUDE
    "anthropic_model": "claude-opus-4-5-20251101",    // -> ANTHROPIC_MODEL_NAME
    "anthropic_max_tokens": 1024,          // -> ANTHROPIC_MAX_TOKENS
    "tools_available": "full",             // -> TOOLS_AVAILABLE ("default" or "full")

    // RAG Configuration
    "rag_enabled": true,                   // -> RETER_RAG_ENABLED
    "rag_embedding_model": "sentence-transformers/all-mpnet-base-v2",  // -> RETER_RAG_MODEL
    "rag_embedding_cache_size": 1000,      // -> RETER_RAG_CACHE_SIZE
    "rag_max_body_lines": 50,              // -> RETER_RAG_MAX_BODY_LINES
    "rag_batch_size": 32,                  // -> RETER_RAG_BATCH_SIZE
    "rag_index_markdown": true,            // -> RETER_RAG_INDEX_MARKDOWN
    "rag_markdown_include": "**/*.md",     // -> RETER_RAG_MARKDOWN_INCLUDE
    "rag_markdown_exclude": "node_modules/**",  // -> RETER_RAG_MARKDOWN_EXCLUDE

    // Code Chunking Configuration
    "rag_code_chunk_enabled": true,        // -> RETER_RAG_CODE_CHUNK_ENABLED
    "rag_code_chunk_size": 30,             // -> RETER_RAG_CODE_CHUNK_SIZE (lines)
    "rag_code_chunk_overlap": 10,          // -> RETER_RAG_CODE_CHUNK_OVERLAP (lines)
    "rag_code_chunk_min_size": 15          // -> RETER_RAG_CODE_CHUNK_MIN_SIZE (lines)
}
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from ..logging_config import is_stderr_suppressed


class ConfigLoader:
    """
    Loads configuration from reter_code.json file.

    ::: This is-in-layer Service-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    Priority: Environment variables > reter_code.json > defaults
    """

    # Mapping from reter_code.json keys to environment variable names
    CONFIG_KEY_TO_ENV = {
        "project_include": "RETER_PROJECT_INCLUDE",
        "project_exclude": "RETER_PROJECT_EXCLUDE",
        "anthropic_model": "ANTHROPIC_MODEL_NAME",
        "anthropic_max_tokens": "ANTHROPIC_MAX_TOKENS",
        "snapshots_dir": "RETER_SNAPSHOTS_DIR",
        "debug_log": "RETER_DEBUG_LOG",
        # Tools configuration
        "tools_available": "TOOLS_AVAILABLE",  # "default" or "full"
        # RAG configuration
        "rag_enabled": "RETER_RAG_ENABLED",
        "rag_embedding_model": "RETER_RAG_MODEL",
        "rag_embedding_cache_size": "RETER_RAG_CACHE_SIZE",
        "rag_max_body_lines": "RETER_RAG_MAX_BODY_LINES",
        "rag_batch_size": "RETER_RAG_BATCH_SIZE",
        "rag_index_markdown": "RETER_RAG_INDEX_MARKDOWN",
        "rag_markdown_include": "RETER_RAG_MARKDOWN_INCLUDE",
        "rag_markdown_exclude": "RETER_RAG_MARKDOWN_EXCLUDE",
        "rag_markdown_max_chunk_words": "RETER_RAG_MARKDOWN_MAX_CHUNK_WORDS",
        "rag_markdown_min_chunk_words": "RETER_RAG_MARKDOWN_MIN_CHUNK_WORDS",
        # Code chunking configuration
        "rag_code_chunk_enabled": "RETER_RAG_CODE_CHUNK_ENABLED",
        "rag_code_chunk_size": "RETER_RAG_CODE_CHUNK_SIZE",
        "rag_code_chunk_overlap": "RETER_RAG_CODE_CHUNK_OVERLAP",
        "rag_code_chunk_min_size": "RETER_RAG_CODE_CHUNK_MIN_SIZE",
    }

    # Default values for RAG configuration
    RAG_DEFAULTS = {
        "rag_enabled": True,
        "rag_embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "rag_embedding_cache_size": 1000,
        "rag_max_body_lines": 50,
        "rag_batch_size": 32,
        "rag_index_markdown": True,
        "rag_markdown_include": "**/*.md",
        "rag_markdown_exclude": "node_modules/**",
        "rag_markdown_max_chunk_words": 500,
        "rag_markdown_min_chunk_words": 50,
        # Code chunking defaults
        "rag_code_chunk_enabled": True,
        "rag_code_chunk_size": 30,
        "rag_code_chunk_overlap": 10,
        "rag_code_chunk_min_size": 15,
    }

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None
        self._loaded = False

    def load(self, project_root: Optional[Path] = None) -> bool:
        """
        Load configuration from reter_code.json.

        Args:
            project_root: Project root directory. If None, uses RETER_PROJECT_ROOT or CWD.

        Returns:
            True if config file was found and loaded, False otherwise.
        """
        if self._loaded:
            return self._config_path is not None

        # Determine project root
        if project_root is None:
            env_root = os.getenv("RETER_PROJECT_ROOT")
            if env_root:
                project_root = Path(env_root)
            else:
                project_root = Path.cwd()

        # Look for reter_code.json
        config_path = project_root / "reter_code.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                self._config_path = config_path
                if not is_stderr_suppressed():
                    print(f"📋 Loaded config from: {config_path}", file=sys.stderr, flush=True)
                self._apply_config()
            except json.JSONDecodeError as e:
                if not is_stderr_suppressed():
                    print(f"⚠️  Invalid JSON in {config_path}: {e}", file=sys.stderr, flush=True)
            except Exception as e:
                if not is_stderr_suppressed():
                    print(f"⚠️  Error loading {config_path}: {e}", file=sys.stderr, flush=True)

        self._loaded = True
        return self._config_path is not None

    def _apply_config(self) -> None:
        """
        Apply config values as environment variables (only if not already set).
        This allows env vars to override config file values.
        """
        for config_key, env_var in self.CONFIG_KEY_TO_ENV.items():
            if config_key in self._config:
                # Only set if env var is not already set
                if not os.getenv(env_var):
                    value = self._config[config_key]
                    # Convert non-string values to strings for env vars
                    if isinstance(value, (int, float)):
                        value = str(value)
                    elif isinstance(value, bool):
                        value = "true" if value else "false"
                    elif isinstance(value, list):
                        value = ",".join(str(v) for v in value)

                    os.environ[env_var] = value
                    if not is_stderr_suppressed():
                        print(f"   {env_var}={value} (from reter_code.json)", file=sys.stderr, flush=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self._config.get(key, default)

    def get_rag_config(self) -> Dict[str, Any]:
        """
        Get RAG configuration with defaults applied.

        Returns:
            Dictionary with all RAG settings, using defaults where not specified.
        """
        rag_config = {}

        for key, default_value in self.RAG_DEFAULTS.items():
            # Check environment variable first
            env_var = self.CONFIG_KEY_TO_ENV.get(key)
            if env_var:
                env_value = os.getenv(env_var)
                if env_value is not None:
                    # Convert string to appropriate type
                    if isinstance(default_value, bool):
                        rag_config[key] = env_value.lower() in ('true', '1', 'yes')
                    elif isinstance(default_value, int):
                        try:
                            rag_config[key] = int(env_value)
                        except ValueError:
                            rag_config[key] = default_value
                    else:
                        rag_config[key] = env_value
                    continue

            # Check config file
            if key in self._config:
                rag_config[key] = self._config[key]
            else:
                rag_config[key] = default_value

        return rag_config

    @property
    def config_path(self) -> Optional[Path]:
        """Path to the loaded config file, or None if not loaded."""
        return self._config_path

    @property
    def config(self) -> Dict[str, Any]:
        """The loaded configuration dictionary."""
        return self._config.copy()


# Global singleton instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(project_root: Optional[Path] = None) -> bool:
    """
    Load configuration from reter_code.json.

    This should be called early in server startup, before other services
    read environment variables.

    Args:
        project_root: Project root directory. If None, auto-detects.

    Returns:
        True if config was loaded, False otherwise.
    """
    return get_config_loader().load(project_root)
