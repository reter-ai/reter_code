"""
find_callers - Find all functions/methods that call the target.

Returns the complete call hierarchy for a function or method.
"""

from codeine.dsl import query, param, reql, Pipeline


@query("find_callers")
@param("target", str, required=True, description="Function or method name to find callers of")
def find_callers() -> Pipeline:
    """
    Find all functions/methods that call the target (direct callers).

    Returns:
        callers: List of caller info with name, file, line
        count: Number of callers found
    """
    return (
        reql('''
            SELECT ?caller ?caller_name ?caller_file ?caller_line ?caller_type
            WHERE {
                { ?caller type {Method} } UNION { ?caller type {Function} }
                ?caller calls ?target .
                ?target name "{target}" .
                ?caller type ?caller_type .
                ?caller name ?caller_name .
                ?caller inFile ?caller_file .
                ?caller atLine ?caller_line .
            }
        ''')
        .select("caller_name", "caller_file", "caller_line", "caller_type", qualified_name="caller")
        .order_by("caller_file")
        .emit("callers")
    )
