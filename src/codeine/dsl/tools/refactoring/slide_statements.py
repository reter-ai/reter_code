"""
slide_statements - Detect long methods that may have scattered variable usage.

Original intent: Detect statements accessing same data but separated by unrelated code.
Current implementation: Find long methods where code organization may need review.

Note: Statement-level variable tracking is not available in the current parser.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("slide_statements", category="refactoring", severity="low")
@param("min_lines", int, default=30, description="Minimum method length to report")
@param("limit", int, default=100, description="Maximum results to return")
def slide_statements() -> Pipeline:
    """
    Detect long methods that may benefit from statement reorganization.

    Since statement-level tracking is not available, this finds long methods
    where related statements may be scattered and could be grouped together.

    Returns:
        findings: List of long methods to review
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?m ?name ?class_name ?file ?line ?loc
            WHERE {
                ?m type {Method} .
                ?m name ?name .
                ?m inFile ?file .
                ?m atLine ?line .
                ?m lineCount ?loc .
                OPTIONAL { ?m definedIn ?c . ?c name ?class_name }
                FILTER ( REGEX(?file, "\\.py$") )
                FILTER ( ?loc >= {min_lines} )
            }
            ORDER BY DESC(?loc)
            LIMIT {limit}
        ''')
        .select("name", "class_name", "file", "line", "loc")
        .map(lambda r: {
            **r,
            "refactoring": "slide_statements",
            "message": f"Method '{r['name']}' is {r['loc']} lines - may have scattered variable usage",
            "suggestion": "Review and group related statements together"
        })
        .emit("findings")
    )
