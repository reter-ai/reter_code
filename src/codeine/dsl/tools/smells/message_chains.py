"""
message_chains - Detect long method call chains (Law of Demeter violations).

Message chains occur when a client asks one object for another object,
then asks that object for yet another object, etc. (a.getB().getC().getD())

Note: Method chain tracking is not currently supported by the parser.
This detector returns methods with many outgoing calls as a proxy.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("message_chains", category="code_smell", severity="medium")
@param("min_calls", int, default=5, description="Minimum outgoing calls to report")
@param("limit", int, default=100, description="Maximum results to return")
def message_chains() -> Pipeline:
    """
    Detect potential message chain issues - methods with many outgoing calls.

    Since method chain tracking is not available, this uses methods
    with high call counts as a proxy for potential coupling issues.

    Returns:
        findings: List of methods with many calls
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?m ?name ?class_name ?file ?line (COUNT(?callee) AS ?call_count)
            WHERE {
                ?m type {Method} .
                ?m name ?name .
                ?m inFile ?file .
                ?m atLine ?line .
                ?m calls ?callee .
                OPTIONAL { ?m definedIn ?c . ?c name ?class_name }
                FILTER ( REGEX(?file, "\\.py$") )
            }
            GROUP BY ?m ?name ?class_name ?file ?line
            HAVING ( ?call_count >= {min_calls} )
            ORDER BY DESC(?call_count)
            LIMIT {limit}
        ''')
        .select("name", "class_name", "file", "line", "call_count")
        .map(lambda r: {
            **r,
            "issue": "high_coupling",
            "message": f"Method '{r['name']}' makes {r['call_count']} calls - may indicate coupling issues",
            "suggestion": "Review for potential message chains or excessive dependencies"
        })
        .emit("findings")
    )
