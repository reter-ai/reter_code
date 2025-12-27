"""
untested_functions - Find public functions without tests.

Identifies public functions that have no test coverage.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("untested_functions", category="test_coverage", severity="medium")
@param("limit", int, default=100, description="Maximum results to return")
def untested_functions() -> Pipeline:
    """
    Find public functions without test coverage.

    Returns:
        findings: List of untested functions
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?f ?name ?file ?line ?line_count
            WHERE {
                ?f type {Function} .
                ?f name ?name .
                ?f inFile ?file .
                ?f atLine ?line .
                OPTIONAL { ?f lineCount ?line_count }
                FILTER ( !REGEX(?name, "^_") && !REGEX(?file, "test_|_test\\.py") )
                MINUS { ?test calls ?f . ?test inFile ?test_file . FILTER ( REGEX(?test_file, "test_") ) }
            }
            ORDER BY DESC(?line_count)
        ''')
        .select("name", "file", "line", "line_count")
        .map(lambda r: {
            **r,
            "issue": "untested_function",
            "message": f"Function '{r['name']}' has no tests",
            "suggestion": f"Create test function 'test_{r['name']}'"
        })
        .emit("findings")
    )
