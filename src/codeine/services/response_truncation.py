"""
Response Truncation Utility

Limits MCP response sizes to avoid filling up context windows.
Configurable via RETER_MCP_MAX_RESPONSE_SIZE environment variable.
"""

import json
import os
from typing import Any, Dict, List, Union

# Default max response size in bytes (5KB)
DEFAULT_MAX_RESPONSE_SIZE = 5000

def get_max_response_size() -> int:
    """Get the maximum response size from environment variable."""
    try:
        return int(os.getenv("RETER_MCP_MAX_RESPONSE_SIZE", str(DEFAULT_MAX_RESPONSE_SIZE)))
    except ValueError:
        return DEFAULT_MAX_RESPONSE_SIZE


def estimate_json_size(obj: Any) -> int:
    """Estimate the JSON-serialized size of an object."""
    try:
        return len(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        # Fallback for non-serializable objects
        return len(str(obj))


def truncate_results_list(
    results: List[Any],
    max_size: int,
    base_response: Dict[str, Any]
) -> tuple[List[Any], bool]:
    """
    Truncate a results list to fit within size limit.

    Args:
        results: The list of results to potentially truncate
        max_size: Maximum allowed size in bytes
        base_response: The base response dict (without results) for size calculation

    Returns:
        Tuple of (truncated_results, was_truncated)
    """
    if not results:
        return results, False

    # Calculate size of base response without results
    test_response = dict(base_response)
    test_response["results"] = []
    base_size = estimate_json_size(test_response)

    # Available space for results
    available_size = max_size - base_size - 100  # 100 bytes buffer for warning field

    if available_size <= 0:
        return [], True

    # Binary search for the right number of results
    truncated = []
    current_size = 2  # Start with "[]"

    for item in results:
        item_size = estimate_json_size(item) + 1  # +1 for comma
        if current_size + item_size > available_size:
            return truncated, True
        truncated.append(item)
        current_size += item_size

    return truncated, False


def truncate_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Truncate a response dict to fit within RETER_MCP_MAX_RESPONSE_SIZE.

    If the response exceeds the size limit, results are truncated and a
    warning field is added to indicate truncation.

    Args:
        response: The response dictionary to potentially truncate

    Returns:
        The response, potentially truncated with a warning added
    """
    max_size = get_max_response_size()

    # Quick check - if response is already under limit, return as-is
    current_size = estimate_json_size(response)
    if current_size <= max_size:
        return response

    # Find truncatable fields (lists that can be shortened)
    truncatable_fields = ["results", "items", "classes", "functions", "methods",
                         "modules", "usages", "callers", "callees", "subclasses",
                         "dependencies", "findings", "recommendations", "tests",
                         "thoughts", "clusters", "pairs", "relevant_docs", "orphaned_docs"]

    result = dict(response)
    was_truncated = False
    truncated_fields = []
    original_counts = {}

    for field in truncatable_fields:
        if field in result and isinstance(result[field], list) and len(result[field]) > 0:
            original_count = len(result[field])
            original_counts[field] = original_count

            # Create base response without this field for size calculation
            base = {k: v for k, v in result.items() if k != field}

            truncated_list, field_truncated = truncate_results_list(
                result[field],
                max_size,
                base
            )

            if field_truncated:
                was_truncated = True
                truncated_fields.append(f"{field}: {len(truncated_list)}/{original_count}")
                result[field] = truncated_list

                # Update count field if present (try multiple naming patterns)
                count_fields = [
                    "count",  # Most common
                    f"{field}_count",  # e.g., functions_count
                    f"total_{field}",  # e.g., total_functions
                ]
                for count_field in count_fields:
                    if count_field in result and result[count_field] == original_count:
                        result[count_field] = len(truncated_list)
                        break

    # Check if still too large after truncation
    final_size = estimate_json_size(result)

    if was_truncated:
        # Calculate how many more items are available
        more_available = {}
        for field in truncatable_fields:
            if field in result and field in original_counts:
                returned = len(result[field])
                total = original_counts[field]
                if total > returned:
                    more_available[field] = total - returned

        result["truncated"] = True
        result["more_available"] = more_available
        result["warning"] = (
            f"Results truncated ({max_size} bytes limit). "
            f"{', '.join(f'{v} more {k}' for k, v in more_available.items())} available. "
            "Use limit/offset for pagination or more specific filters."
        )
        result["response_size_bytes"] = final_size
        result["max_response_size_bytes"] = max_size

    # If still too large, do aggressive truncation
    if final_size > max_size:
        for field in truncatable_fields:
            if field in result and isinstance(result[field], list):
                # Keep only first few items
                keep = max(1, len(result[field]) // 4)
                if len(result[field]) > keep:
                    original = original_counts.get(field, len(result[field]))
                    result[field] = result[field][:keep]
                    truncated_fields = [f for f in truncated_fields if not f.startswith(f"{field}:")]
                    truncated_fields.append(f"{field}: {keep}/{original}")

        # Recalculate more_available after aggressive truncation
        more_available = {}
        for field in truncatable_fields:
            if field in result and field in original_counts:
                returned = len(result[field])
                total = original_counts[field]
                if total > returned:
                    more_available[field] = total - returned

        result["truncated"] = True
        result["more_available"] = more_available
        result["warning"] = (
            f"Results aggressively truncated ({max_size} bytes limit). "
            f"{', '.join(f'{v} more {k}' for k, v in more_available.items())} available. "
            "Use limit/offset for pagination or increase RETER_MCP_MAX_RESPONSE_SIZE."
        )
        result["response_size_bytes"] = estimate_json_size(result)

    return result
