"""
find_interface_implementations - Find classes implementing interfaces/ABCs.

Identifies classes that implement abstract base classes or protocols.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("find_interface_implementations", category="patterns", severity="info")
@param("interface_name", str, required=False, default=None, description="Filter by interface name")
@param("limit", int, default=100, description="Maximum results to return")
def find_interface_implementations() -> Pipeline:
    """
    Find classes implementing abstract base classes or interfaces.

    Detects classes that inherit from base classes (classes with names
    containing Base, Abstract, ABC, Interface, or Protocol).

    Returns:
        findings: List of interface implementations
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?c ?class_name ?parent_name ?file ?line
            WHERE {
                ?c type {Class} .
                ?c name ?class_name .
                ?c inFile ?file .
                ?c atLine ?line .
                ?c inheritsFrom ?parent_name .
                FILTER ( REGEX(?parent_name, "Base|Abstract|ABC|Interface|Protocol") )
            }
            ORDER BY ?parent_name ?class_name
            LIMIT {limit}
        ''')
        .select("class_name", "file", "line", parent="parent_name")
        .map(lambda r: {
            **r,
            "interface_name": r.get("parent", ""),
            "message": f"Class '{r['class_name']}' implements '{r.get('parent', '')}'"
        })
        .emit("findings")
    )
