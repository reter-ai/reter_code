"""
untested_methods - Find public methods without test coverage.

Identifies public methods that have no corresponding tests.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("untested_methods", category="test_coverage", severity="medium")
@param("limit", int, default=100, description="Maximum results to return")
def untested_methods() -> Pipeline:
    """
    Find public methods without test coverage.

    Returns:
        findings: List of untested methods
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?m ?name ?class_name ?file ?line ?line_count
            WHERE {
                ?m type {Method} .
                ?m name ?name .
                ?m inFile ?file .
                ?m atLine ?line .
                ?m definedIn ?c .
                ?c name ?class_name .
                OPTIONAL { ?m lineCount ?line_count }
                FILTER ( !REGEX(?name, "^_") && !REGEX(?file, "test_|_test\\.py") )
                MINUS { ?test calls ?m . ?test inFile ?test_file . FILTER ( REGEX(?test_file, "test_") ) }
            }
            ORDER BY DESC(?line_count)
        ''')
        .select("name", "class_name", "file", "line", "line_count")
        .map(lambda r: {
            **r,
            "issue": "untested_method",
            "message": f"Method '{r.get('class_name', '')}.{r['name']}' has no tests",
            "suggestion": f"Add test for method '{r['name']}'"
        })
        .emit("findings")
    )
