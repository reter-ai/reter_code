"""
public_api_untested - Find public API without tests.

Identifies public API entities that have no test coverage.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("public_api_untested", category="test_coverage", severity="high")
@param("limit", int, default=100, description="Maximum results to return")
def public_api_untested() -> Pipeline:
    """
    Find public API without test coverage.

    Returns:
        findings: List of untested public API
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?e ?name ?entity_type ?file ?line
            WHERE {
                { ?e type {Class} } UNION { ?e type {Function} }
                ?e name ?name .
                ?e inFile ?file .
                ?e atLine ?line .
                ?e isExported true .
                FILTER ( !REGEX(?name, "^_") && !REGEX(?file, "test_") )
                MINUS { ?test calls ?e . ?test inFile ?test_file . FILTER ( REGEX(?test_file, "test_") ) }
            }
            ORDER BY ?entity_type ?name
        ''')
        .select("name", "entity_type", "file", "line")
        .map(lambda r: {
            **r,
            "issue": "public_api_untested",
            "message": f"Public API {r['entity_type']} '{r['name']}' has no tests",
            "suggestion": "Critical: public API must have test coverage"
        })
        .emit("findings")
    )
