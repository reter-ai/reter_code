"""
untested_classes - Find classes without corresponding test classes.

Identifies classes that have no test coverage.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("untested_classes", category="test_coverage", severity="medium")
@param("limit", int, default=100, description="Maximum results to return")
def untested_classes() -> Pipeline:
    """
    Find classes without corresponding test classes.

    Returns:
        findings: List of untested classes
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?c ?name ?file ?line (COUNT(?method) AS ?method_count)
            WHERE {
                ?c type {Class} .
                ?c name ?name .
                ?c inFile ?file .
                ?c atLine ?line .
                OPTIONAL { ?method type {Method} . ?method definedIn ?c }
                FILTER ( !REGEX(?name, "^Test|Test$|^_") && !REGEX(?file, "test_|_test\\.py") )
                MINUS {
                    ?test type {Class} .
                    ?test name ?test_name .
                    FILTER ( REGEX(?test_name, "^Test|Test$") )
                }
            }
            GROUP BY ?c ?name ?file ?line
            ORDER BY ?file ?name
            LIMIT {limit}
        ''')
        .select("name", "file", "line", "method_count")
        .map(lambda r: {
            **r,
            "issue": "untested_class",
            "message": f"Class '{r['name']}' has no test class",
            "suggestion": f"Create a test class 'Test{r['name']}' or '{r['name']}Test'"
        })
        .emit("findings")
    )
