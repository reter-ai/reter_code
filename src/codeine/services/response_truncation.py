"""
Response Truncation Utility

Limits MCP response sizes to avoid filling up context windows.
When responses exceed the limit, full results are saved to .codeine/results/
and a reference is returned in the response.

Configurable via RETER_MCP_MAX_RESPONSE_SIZE environment variable.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default max response size in bytes (5KB)
DEFAULT_MAX_RESPONSE_SIZE = 5000

# Results directory name
RESULTS_DIR_NAME = ".codeine/results"


def get_max_response_size() -> int:
    """Get the maximum response size from environment variable."""
    try:
        return int(os.getenv("RETER_MCP_MAX_RESPONSE_SIZE", str(DEFAULT_MAX_RESPONSE_SIZE)))
    except ValueError:
        return DEFAULT_MAX_RESPONSE_SIZE


def get_results_dir() -> Path:
    """Get or create the results directory for storing full responses."""
    # Use current working directory as project root
    project_root = Path.cwd()
    results_dir = project_root / RESULTS_DIR_NAME

    # Create directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir


def save_full_response(response: Dict[str, Any], query_hint: str = "") -> str:
    """
    Save the full response to a file and return the file path.

    Args:
        response: The full response dictionary to save
        query_hint: Optional hint about what query produced this (for filename)

    Returns:
        The path to the saved file
    """
    results_dir = get_results_dir()

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]

    # Clean query hint for filename
    if query_hint:
        hint = "".join(c if c.isalnum() or c in "_-" else "_" for c in query_hint[:30])
        filename = f"{timestamp}_{hint}_{unique_id}.json"
    else:
        filename = f"{timestamp}_{unique_id}.json"

    filepath = results_dir / filename

    # Save with pretty formatting
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(response, f, indent=2, default=str)

    return str(filepath)


def estimate_json_size(obj: Any) -> int:
    """Estimate the JSON-serialized size of an object."""
    try:
        return len(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        # Fallback for non-serializable objects
        return len(str(obj))


def create_summary(response: Dict[str, Any], truncatable_fields: List[str]) -> Dict[str, Any]:
    """
    Create a summary of the response showing counts and first few items.

    Args:
        response: The full response
        truncatable_fields: List of field names that contain result lists

    Returns:
        A summary dictionary with counts and previews
    """
    summary = {}

    for field in truncatable_fields:
        if field in response and isinstance(response[field], list):
            items = response[field]
            count = len(items)

            if count > 0:
                # Show first 2 items as preview
                preview_count = min(2, count)
                preview = items[:preview_count]

                # Truncate long strings in preview items
                def truncate_strings(obj, max_len=100):
                    if isinstance(obj, str) and len(obj) > max_len:
                        return obj[:max_len] + "..."
                    elif isinstance(obj, dict):
                        return {k: truncate_strings(v, max_len) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [truncate_strings(item, max_len) for item in obj[:3]]
                    return obj

                preview = [truncate_strings(item) for item in preview]

                summary[field] = {
                    "count": count,
                    "preview": preview,
                    "showing": f"{preview_count} of {count}"
                }

    return summary


def truncate_response(response: Dict[str, Any], query_hint: str = "") -> Dict[str, Any]:
    """
    Truncate a response dict to fit within RETER_MCP_MAX_RESPONSE_SIZE.

    If the response exceeds the size limit, the full response is saved to
    .codeine/results/ and a reference is returned along with a summary.

    Args:
        response: The response dictionary to potentially truncate
        query_hint: Optional hint about the query (used in filename)

    Returns:
        The response, potentially truncated with a file reference
    """
    max_size = get_max_response_size()

    # Quick check - if response is already under limit, return as-is
    current_size = estimate_json_size(response)
    if current_size <= max_size:
        return response

    # Response is too large - save full results to file
    truncatable_fields = ["results", "items", "classes", "functions", "methods",
                         "modules", "usages", "callers", "callees", "subclasses",
                         "dependencies", "findings", "recommendations", "tests",
                         "thoughts", "clusters", "pairs", "relevant_docs", "orphaned_docs",
                         "similar_tools", "rag_matches", "similar_tested_code",
                         "well_designed_classes", "well_structured_alternatives"]

    # Save full response to file
    try:
        full_results_file = save_full_response(response, query_hint)
    except Exception as e:
        # If we can't save, fall back to aggressive truncation
        full_results_file = None

    # Build truncated response with file reference AT THE TOP
    result = {}

    # FIRST: Add truncation info and file reference at the beginning
    if full_results_file:
        result["full_results_file"] = full_results_file
        result["message"] = (
            f"Response too large ({current_size} bytes > {max_size} limit). "
            f"Full results saved to: {full_results_file}"
        )
    else:
        result["message"] = (
            f"Response too large ({current_size} bytes > {max_size} limit). "
            f"Could not save full results. Use more specific filters."
        )

    result["truncated"] = True
    result["response_size_bytes"] = current_size
    result["max_response_size_bytes"] = max_size

    # Create summary of truncatable fields
    summary = create_summary(response, truncatable_fields)

    # Calculate total counts
    total_items = sum(
        len(response.get(field, []))
        for field in truncatable_fields
        if isinstance(response.get(field), list)
    )

    result["total_results"] = total_items
    result["summary"] = summary

    # Copy non-truncatable fields
    for key, value in response.items():
        if key not in truncatable_fields:
            result[key] = value

    # Include first few items from main results field if possible
    main_fields = ["findings", "results", "items"]
    for field in main_fields:
        if field in response and isinstance(response[field], list) and response[field]:
            # Include up to 3 items in the truncated response
            preview_items = response[field][:3]

            # Further truncate each item to remove large nested arrays
            def slim_item(item):
                if not isinstance(item, dict):
                    return item
                slimmed = {}
                for k, v in item.items():
                    if isinstance(v, list) and len(v) > 2:
                        slimmed[k] = v[:2]
                        slimmed[f"{k}_truncated"] = len(v) - 2
                    elif isinstance(v, str) and len(v) > 200:
                        slimmed[k] = v[:200] + "..."
                    else:
                        slimmed[k] = v
                return slimmed

            result[field] = [slim_item(item) for item in preview_items]
            result[f"{field}_preview_count"] = len(preview_items)
            result[f"{field}_total_count"] = len(response[field])
            break

    return result


def cleanup_old_results(max_age_hours: int = 24) -> int:
    """
    Clean up old result files older than max_age_hours.

    Args:
        max_age_hours: Maximum age of files to keep

    Returns:
        Number of files deleted
    """
    from datetime import timedelta

    results_dir = get_results_dir()
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    deleted = 0

    for filepath in results_dir.glob("*.json"):
        try:
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            if mtime < cutoff:
                filepath.unlink()
                deleted += 1
        except Exception:
            pass

    return deleted
