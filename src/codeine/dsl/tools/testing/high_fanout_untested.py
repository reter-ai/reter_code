"""
high_fanout_untested - Find high fan-out functions without tests.

Identifies functions with many dependencies that have no test coverage.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("high_fanout_untested", category="test_coverage", severity="high")
@param("min_fanout", int, default=5, description="Minimum fan-out count")
@param("limit", int, default=100, description="Maximum results to return")
def high_fanout_untested() -> Pipeline:
    """
    Find high fan-out functions without test coverage.

    Returns:
        findings: List of high fan-out untested code
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?e ?name ?class_name ?file ?line (COUNT(?callee) AS ?fanout)
            WHERE {
                { ?e type {Method} } UNION { ?e type {Function} }
                ?e name ?name .
                ?e inFile ?file .
                ?e atLine ?line .
                ?e calls ?callee .
                OPTIONAL { ?e definedIn ?c . ?c name ?class_name }
                FILTER ( !REGEX(?file, "test_") )
                MINUS { ?test calls ?e . ?test inFile ?test_file . FILTER ( REGEX(?test_file, "test_") ) }
            }
            GROUP BY ?e ?name ?class_name ?file ?line
            HAVING (?fanout >= {min_fanout})
            ORDER BY DESC(?fanout)
            LIMIT {limit}
        ''')
        .select("name", "class_name", "file", "line", "fanout")
        .map(lambda r: {
            **r,
            "issue": "high_fanout_untested",
            "message": f"High fan-out code '{r['name']}' (calls {r.get('fanout', 0)} functions) has no tests",
            "suggestion": "Integration risk: add tests covering all called functions"
        })
        .emit("findings")
    )
