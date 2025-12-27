"""
large_untested_modules - Find modules with many classes but few tests.

Identifies large modules that have inadequate test coverage.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("large_untested_modules", category="test_coverage", severity="medium")
@param("min_classes", int, default=5, description="Minimum classes for large module")
@param("limit", int, default=100, description="Maximum results to return")
def large_untested_modules() -> Pipeline:
    """
    Find modules with many classes but few tests.

    Returns:
        findings: List of large untested modules
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?file (COUNT(?class) AS ?class_count)
            WHERE {
                ?class type {Class} .
                ?class inFile ?file .
                FILTER ( !REGEX(?file, "test_|tests/|_test\\.py") )
            }
            GROUP BY ?file
            HAVING ( ?class_count >= {min_classes} )
            ORDER BY DESC(?class_count)
            LIMIT {limit}
        ''')
        .select("file", "class_count")
        .map(lambda r: {
            **r,
            "issue": "large_untested_module",
            "message": f"Module '{r['file']}' has {r.get('class_count', 0)} classes - review test coverage",
            "suggestion": "Ensure adequate test coverage for large modules"
        })
        .emit("findings")
    )
