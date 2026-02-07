"""
Helper functions for Natural Language Query (NLQ) tool.

Extracted from tool_registrar.py to improve maintainability and testability.
"""

import re
from typing import Dict, Any, List, Optional

# Import constants
from .nlq_constants import REQL_SYNTAX_HELP

# Debug logger - use centralized configuration
from ..logging_config import configure_logger_for_nlq_debug

logger = configure_logger_for_nlq_debug(__name__)


def query_instance_schema(reter) -> str:
    """
    Query the actual schema from a RETER instance.

    Args:
        reter: RETER wrapper instance

    Returns:
        Schema info as formatted string, or empty string if failed
    """
    try:
        schema_query = """SELECT ?concept ?pred (COUNT(*) AS ?count)
            WHERE { ?s type ?concept . ?s ?pred ?o }
            GROUP BY ?concept ?pred
            ORDER BY ?concept DESC(?count)"""
        schema_result = reter.reql(schema_query)

        if schema_result is not None and len(schema_result) > 0:
            # Group by concept
            concepts = {}
            for row in schema_result.to_pydict().values():
                # Convert columns to lists
                concept_col = schema_result.column('?concept').to_pylist()
                pred_col = schema_result.column('?pred').to_pylist()
                count_col = schema_result.column('?count').to_pylist()

                for i in range(len(concept_col)):
                    concept = concept_col[i]
                    pred = pred_col[i]
                    count = count_col[i]
                    if concept not in concepts:
                        concepts[concept] = []
                    concepts[concept].append(f"{pred} ({count})")
                break  # Only need one iteration

            schema_lines = ["## Actual Schema in This Instance\n"]
            for concept, preds in concepts.items():
                schema_lines.append(f"### {concept}")
                schema_lines.append(f"Predicates: {', '.join(preds[:15])}")  # Top 15 predicates
                schema_lines.append("")
            schema_info = "\n".join(schema_lines)
            logger.debug(f"SCHEMA INFO:\n{schema_info}")
            return schema_info
    except Exception as e:
        logger.debug(f"Failed to query schema: {e}")
    return ""


def build_nlq_prompt(
    question: str,
    schema_info: str,
    attempt: int,
    generated_query: Optional[str] = None,
    last_error: Optional[str] = None
) -> str:
    """
    Build the prompt for the LLM to generate a REQL query.

    Args:
        question: Natural language question
        schema_info: Schema info from instance
        attempt: Current attempt number (1-based)
        generated_query: Previous query (for retries)
        last_error: Previous error message (for retries)

    Returns:
        Formatted prompt string
    """
    if attempt == 1:
        return f"{schema_info}\nQuestion: {question}\n\nGenerate a REQL query:"
    else:
        # Include previous error for retry with syntax help
        return f"""Question: {question}

Previous query attempt:
```
{generated_query}
```

Error received:
{last_error}

{REQL_SYNTAX_HELP}

Please fix the REQL query to correct the syntax error. Return ONLY the corrected query:"""


def extract_reql_from_response(response_text: str) -> str:
    """
    Extract REQL query from LLM response.

    Handles code blocks and raw query responses.

    Args:
        response_text: Raw LLM response

    Returns:
        Extracted REQL query string
    """
    # Try to find query in code blocks first
    code_block_match = re.search(
        r'```(?:reql|sparql|sql)?\s*\n?(.*?)\n?```',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    if code_block_match:
        query = code_block_match.group(1).strip()
    else:
        # Assume entire response is the query
        query = response_text.strip()

    # Remove any markdown artifacts
    query = re.sub(r'^```\w*\s*', '', query)
    query = re.sub(r'\s*```$', '', query)

    return query


def execute_reql_query(reter, query: str) -> List[Dict[str, Any]]:
    """
    Execute a REQL query and convert results to list of dicts.

    Args:
        reter: RETER wrapper instance
        query: REQL query string

    Returns:
        List of result dictionaries

    Raises:
        Exception: If query execution fails
    """
    result = reter.reql(query)

    # Convert result to list of dicts
    if result.num_rows == 0:
        return []

    column_names = result.column_names
    rows = []
    for i in range(result.num_rows):
        row = {}
        for col_name in column_names:
            value = result.column(col_name)[i].as_py()
            row[col_name] = value
        rows.append(row)
    return rows


def is_retryable_error(error_message: str) -> bool:
    """
    Check if an error is worth retrying with a modified query.

    Args:
        error_message: Error message string

    Returns:
        True if the error should trigger a retry
    """
    error_lower = error_message.lower()
    retryable_patterns = [
        "parse", "syntax", "unexpected", "failed to apply",
        "filter", "regex", "invalid", "unknown", "error",
        "expected", "unrecognized", "unsupported"
    ]
    return any(pattern in error_lower for pattern in retryable_patterns)
