"""
flag_arguments - Detect boolean parameters that control function behavior.

Flag arguments indicate that a function does more than one thing.
The function should typically be split into separate functions.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("flag_arguments", category="code_smell", severity="medium")
@param("limit", int, default=100, description="Maximum results to return")
def flag_arguments() -> Pipeline:
    """
    Detect flag arguments - boolean params controlling function behavior.

    Returns:
        findings: List of functions with flag arguments
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?func ?name ?file ?line ?param_name ?class_name
            WHERE {
                {
                    ?func type {Function} .
                    ?func name ?name .
                    ?func inFile ?file .
                    ?func atLine ?line .
                    ?param parameterOf ?func .
                    ?param name ?param_name .
                    ?param parameterType ?ptype .
                    FILTER ( ?ptype = "bool" || ?ptype = "boolean" )
                    OPTIONAL { ?func definedIn ?c . ?c name ?class_name }
                }
                UNION
                {
                    ?func type {Method} .
                    ?func name ?name .
                    ?func inFile ?file .
                    ?func atLine ?line .
                    ?param parameterOf ?func .
                    ?param name ?param_name .
                    ?param parameterType ?ptype .
                    FILTER ( ?ptype = "bool" || ?ptype = "boolean" )
                    OPTIONAL { ?func definedIn ?c . ?c name ?class_name }
                }
            }
            ORDER BY ?file ?line
            LIMIT {limit}
        ''')
        .select("name", "file", "line", "param_name", "class_name")
        .map(lambda r: {
            **r,
            "issue": "flag_argument",
            "message": f"Function '{r['name']}' has boolean flag parameter '{r['param_name']}'",
            "suggestion": "Split into two separate functions or use polymorphism"
        })
        .emit("findings")
    )
