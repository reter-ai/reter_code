"""
pipeline_conversion - Detect functions that may benefit from pipeline patterns.

Original intent: Detect loops that could be replaced with collection pipelines.
Current implementation: Find functions using list operations that could use comprehensions.

Note: Loop-level tracking is not available in the current parser.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("pipeline_conversion", category="refactoring", severity="low")
@param("limit", int, default=100, description="Maximum results to return")
def pipeline_conversion() -> Pipeline:
    """
    Detect functions that may benefit from pipeline/comprehension patterns.

    Since loop tracking is not available, this finds functions
    that call list operations like append, extend, filter.

    Returns:
        findings: List of functions to review for pipeline conversion
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?m ?name ?class_name ?file ?line ?callee
            WHERE {
                ?m type {Method} .
                ?m name ?name .
                ?m inFile ?file .
                ?m atLine ?line .
                ?m calls ?callee .
                OPTIONAL { ?m definedIn ?c . ?c name ?class_name }
                FILTER ( REGEX(?file, "\\.py$") )
                FILTER ( REGEX(?callee, "append|extend|filter|map|reduce") )
                FILTER ( !REGEX(?name, "^test_|^_") )
            }
            ORDER BY ?file ?name
            LIMIT {limit}
        ''')
        .select("name", "class_name", "file", "line", "callee")
        .map(lambda r: {
            **r,
            "refactoring": "replace_loop_with_pipeline",
            "message": f"Method '{r['name']}' uses '{r['callee']}' - may benefit from comprehension",
            "suggestion": "Consider using list comprehension or generator expression"
        })
        .emit("findings")
    )
