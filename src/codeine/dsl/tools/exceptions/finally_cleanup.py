"""
finally_without_context_manager - Detect try/finally needing 'with' statement.

Identifies try/finally blocks used for resource cleanup that
could be replaced with context managers.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("finally_without_context_manager", category="exception_handling", severity="low")
@param("limit", int, default=100, description="Maximum results to return")
def finally_without_context_manager() -> Pipeline:
    """
    Detect try/finally blocks that should use context managers.

    Returns:
        findings: List of finally blocks needing context managers
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?t ?func_name ?file ?line ?cleanup_type
            WHERE {
                ?t type {TryBlock} .
                ?t inFile ?file .
                ?t atLine ?line .
                ?t hasFinally true .
                ?t cleanupPattern ?cleanup_type .
                OPTIONAL { ?t inFunction ?f . ?f name ?func_name }
                FILTER ( ?cleanup_type = "file_close" || ?cleanup_type = "lock_release" || ?cleanup_type = "connection_close" )
            }
            ORDER BY ?file ?line
        ''')
        .select("func_name", "file", "line", "cleanup_type")
        .map(lambda r: {
            **r,
            "issue": "finally_without_context_manager",
            "message": f"try/finally for {r.get('cleanup_type', 'cleanup')} in '{r.get('func_name', 'unknown')}'",
            "suggestion": "Use a context manager ('with' statement) instead"
        })
        .emit("findings")
    )
