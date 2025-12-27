"""
move_statements - Detect functions called from many places (candidates for refactoring).

Original intent: Detect repeated statement sequences before/after function calls.
Current implementation: Find heavily-called functions that may need statement consolidation.

Note: Statement-level tracking is not available in the current parser.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("move_statements", category="refactoring", severity="medium")
@param("min_callers", int, default=5, description="Minimum callers to report")
@param("limit", int, default=100, description="Maximum results to return")
def move_statements() -> Pipeline:
    """
    Detect heavily-called functions - candidates for statement consolidation.

    Since statement-level tracking is not available, this finds functions
    called from many places, which may benefit from consolidating
    common setup/teardown patterns.

    Returns:
        findings: List of heavily-called functions
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?func ?func_name ?file ?line (COUNT(?caller) AS ?caller_count)
            WHERE {
                ?func type {Function} .
                ?func name ?func_name .
                ?func inFile ?file .
                ?func atLine ?line .
                ?caller calls ?func .
                FILTER ( REGEX(?file, "\\.py$") )
            }
            GROUP BY ?func ?func_name ?file ?line
            HAVING ( ?caller_count >= {min_callers} )
            ORDER BY DESC(?caller_count)
            LIMIT {limit}
        ''')
        .select("func_name", "file", "line", "caller_count")
        .map(lambda r: {
            **r,
            "refactoring": "move_statements_into_function",
            "message": f"Function '{r['func_name']}' called from {r['caller_count']} places",
            "suggestion": "Review call sites for common patterns that could be moved into the function"
        })
        .emit("findings")
    )
