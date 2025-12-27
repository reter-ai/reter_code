"""
split_loop - Detect complex methods that may contain loops needing splitting.

Original intent: Detect loops doing multiple independent things.
Current implementation: Find complex methods that may contain loops to review.

Note: Loop-level tracking is not available in the current parser.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("split_loop", category="refactoring", severity="low")
@param("min_lines", int, default=25, description="Minimum method length")
@param("limit", int, default=100, description="Maximum results to return")
def split_loop() -> Pipeline:
    """
    Detect complex methods that may contain loops worth splitting.

    Since loop tracking is not available, this finds long methods
    that may contain complex loops doing multiple things.

    Returns:
        findings: List of complex methods to review for loop splitting
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
                FILTER ( !REGEX(?name, "^test_|^_") )
            }
            ORDER BY DESC(?loc)
            LIMIT {limit}
        ''')
        .select("name", "class_name", "file", "line", "loc")
        .map(lambda r: {
            **r,
            "refactoring": "split_loop",
            "message": f"Method '{r['name']}' ({r['loc']} lines) may contain loops to review",
            "suggestion": "Check for loops doing multiple things that could be split"
        })
        .emit("findings")
    )
