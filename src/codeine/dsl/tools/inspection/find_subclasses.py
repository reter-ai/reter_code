"""
find_subclasses - Find all subclasses of a class (direct and indirect).

Returns the complete list of classes that inherit from the target.
"""

from codeine.dsl import query, param, reql, Pipeline


@query("find_subclasses")
@param("target", str, required=True, description="Class name to find subclasses of")
def find_subclasses() -> Pipeline:
    """
    Find all subclasses of a class (direct subclasses).

    Note: inheritsFrom returns qualified name strings, so we use CONTAINS
    to match the target class name within the parent string.

    Returns:
        subclasses: List of subclass info with name, file, line
        count: Number of subclasses found
    """
    return (
        reql('''
            SELECT ?sub ?name ?file ?line ?parent_name
            WHERE {
                ?sub type {Class} .
                ?sub inheritsFrom ?parent_name .
                ?sub name ?name .
                ?sub inFile ?file .
                ?sub atLine ?line .
                FILTER CONTAINS(?parent_name, "{target}") .
            }
        ''')
        .select("name", "file", "line", "parent_name", qualified_name="sub")
        .order_by("file")
        .emit("subclasses")
    )
