"""
get_complexity - Calculate complexity metrics for the codebase.

Returns various complexity metrics for classes, methods, and functions.
"""

from codeine.dsl import query, param, reql, Pipeline


@query("get_complexity")
@param("limit", int, default=100, description="Maximum results to return")
def get_complexity() -> Pipeline:
    """
    Calculate complexity metrics for the codebase.

    Returns:
        class_complexity: Complexity metrics for classes
        method_complexity: Complexity metrics for methods
        inheritance_complexity: Inheritance depth metrics
        call_complexity: Call graph complexity metrics
    """
    return (
        reql('''
            SELECT ?c ?name ?file (COUNT(?method) AS ?method_count) (COUNT(?attr) AS ?attr_count) (COUNT(?parent) AS ?parent_count)
            WHERE {
                ?c type {Class} .
                ?c name ?name .
                ?c inFile ?file .
                OPTIONAL { ?method type {Method} . ?method definedIn ?c }
                OPTIONAL { ?attr type {Field} . ?attr definedIn ?c }
                OPTIONAL { ?c inheritsFrom ?parent }
            }
            GROUP BY ?c ?name ?file
            ORDER BY DESC(?method_count)
            LIMIT {limit}
        ''')
        .select("name", "file", "method_count", "attr_count", "parent_count", qualified_name="c")
        .emit("class_complexity")
    )
