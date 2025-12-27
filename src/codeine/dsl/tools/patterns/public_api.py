"""
find_public_api - Find all public API entities in the codebase.

Identifies public classes and functions (non-underscore prefixed).
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("find_public_api", category="patterns", severity="info")
@param("limit", int, default=100, description="Maximum results to return")
def find_public_api() -> Pipeline:
    """
    Find all public API entities in the codebase.

    Returns:
        findings: List of public API entities
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?e ?name ?file ?line ?docstring
            WHERE {
                { ?e type {Class} }
                UNION
                { ?e type {Function} }
                ?e name ?name .
                ?e inFile ?file .
                ?e atLine ?line .
                OPTIONAL { ?e docstring ?docstring }
                FILTER ( !REGEX(?name, "^_") && REGEX(?file, "\\.py$") )
            }
            ORDER BY ?file ?name
            LIMIT {limit}
        ''')
        .select("name", "file", "line", "docstring")
        .emit("findings")
    )
