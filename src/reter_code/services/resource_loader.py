"""
Shared resource loading utilities.

Consolidates duplicate _load_resource implementations from:
- hybrid_query_engine.py
- nlq_constants.py
- agent_sdk_client.py
"""

from pathlib import Path

# Resource directory (relative to package root)
_RESOURCES_DIR = Path(__file__).parent.parent / "resources"


def load_resource(filename: str) -> str:
    """Load a resource file from the resources directory.

    Args:
        filename: Name of the resource file to load

    Returns:
        Contents of the file, or an error message if not found
    """
    resource_path = _RESOURCES_DIR / filename
    if resource_path.exists():
        with open(resource_path, 'r', encoding='utf-8') as f:
            return f.read()
    return f"# Resource file not found: {filename}"


def get_resources_dir() -> Path:
    """Get the path to the resources directory."""
    return _RESOURCES_DIR
