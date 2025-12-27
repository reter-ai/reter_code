"""
find_callees - Find all functions/methods called by the target.

Returns the complete list of functions/methods called by the target.
"""

from codeine.dsl import query, param, reql, Pipeline


@query("find_callees")
@param("target", str, required=True, description="Function or method name to find callees of")
def find_callees() -> Pipeline:
    """
    Find all functions/methods called by the target (direct callees).

    Returns:
        callees: List of callee info with name, file, line
        count: Number of callees found
    """
    return (
        reql('''
            SELECT ?callee ?callee_name ?callee_file ?callee_line ?callee_type
            WHERE {
                { ?caller type {Method} } UNION { ?caller type {Function} }
                ?caller name "{target}" .
                ?caller calls ?callee .
                { ?callee type {Method} } UNION { ?callee type {Function} }
                ?callee type ?callee_type .
                ?callee name ?callee_name .
                ?callee inFile ?callee_file .
                ?callee atLine ?callee_line .
            }
        ''')
        .select("callee_name", "callee_file", "callee_line", "callee_type", qualified_name="callee")
        .order_by("callee_file")
        .emit("callees")
    )
