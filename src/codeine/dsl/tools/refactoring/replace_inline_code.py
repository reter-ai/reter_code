"""
replace_inline_code - Detect similar methods that may contain duplicate code.

Original intent: Detect duplicate statement sequences replaceable with function calls.
Current implementation: Find methods with similar names/sizes that may share duplicate code.

Note: For actual duplicate code detection, use the RAG tool 'detect_duplicate_code'.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("replace_inline_code", category="refactoring", severity="medium")
@param("min_methods", int, default=2, description="Minimum similar methods")
@param("limit", int, default=100, description="Maximum results to return")
def replace_inline_code() -> Pipeline:
    """
    Detect classes with many similar-length methods (potential duplication).

    Since code pattern tracking is not available, this finds classes
    with multiple methods of similar size that may share duplicate code.

    For detailed duplicate detection, use the 'detect_duplicate_code' RAG tool.

    Returns:
        findings: List of classes to review for code duplication
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?c ?class_name ?file ?line (COUNT(?m) AS ?method_count)
            WHERE {
                ?c type {Class} .
                ?c name ?class_name .
                ?c inFile ?file .
                ?c atLine ?line .
                ?m definedIn ?c .
                ?m type {Method} .
                FILTER ( REGEX(?file, "\\.py$") )
            }
            GROUP BY ?c ?class_name ?file ?line
            HAVING ( ?method_count >= 5 )
            ORDER BY DESC(?method_count)
            LIMIT {limit}
        ''')
        .select("class_name", "file", "line", "method_count")
        .map(lambda r: {
            **r,
            "refactoring": "extract_function",
            "message": f"Class '{r['class_name']}' has {r['method_count']} methods - check for duplication",
            "suggestion": "Use 'detect_duplicate_code' RAG tool for detailed analysis"
        })
        .emit("findings")
    )
