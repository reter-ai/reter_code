"""
find_external_dependencies - Find external package dependencies.

Identifies all external (pip) packages used in the codebase.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("find_external_dependencies", category="dependencies", severity="info")
@param("limit", int, default=100, description="Maximum results to return")
def find_external_dependencies() -> Pipeline:
    """
    Find external package dependencies.

    Returns:
        findings: List of external packages imported by modules
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?m ?module_name ?package_name ?file
            WHERE {
                ?m type {Module} .
                ?m name ?module_name .
                ?m inFile ?file .
                ?m imports ?package_name
            }
            ORDER BY ?package_name ?module_name
            LIMIT {limit}
        ''')
        .select("package_name", "module_name", "file")
        .unique(lambda r: r.get("package_name"))
        .map(lambda r: {
            **r,
            "issue": "external_dependency",
            "message": f"Package '{r['package_name']}' is imported"
        })
        .emit("findings")
    )
