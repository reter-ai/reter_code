"""
too_general_exceptions - Detect catching overly broad exceptions.

Identifies except blocks that catch Exception, BaseException,
or use bare except clauses.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("too_general_exceptions", category="exception_handling", severity="high")
@param("limit", int, default=100, description="Maximum results to return")
def too_general_exceptions() -> Pipeline:
    """
    Detect catching overly broad exception types.

    Returns:
        findings: List of too-general exception handlers
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?h ?exception_type ?func_name ?file ?line
            WHERE {
                ?h type {CatchClause} .
                ?h inFile ?file .
                ?h atLine ?line .
                ?h exceptionType ?exception_type .
                OPTIONAL { ?h inFunction ?f . ?f name ?func_name }
                FILTER ( ?exception_type = "Exception" || ?exception_type = "BaseException" || ?exception_type = "" )
            }
            ORDER BY ?file ?line
        ''')
        .select("exception_type", "func_name", "file", "line")
        .map(lambda r: {
            **r,
            "issue": "too_general_exception",
            "message": f"Catching too-general '{r.get('exception_type', 'bare except')}' in '{r.get('func_name', 'unknown')}'",
            "suggestion": "Catch specific exception types to avoid masking bugs"
        })
        .emit("findings")
    )
