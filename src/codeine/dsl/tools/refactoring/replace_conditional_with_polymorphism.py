"""
replace_conditional_with_polymorphism - Suggest polymorphism over conditionals.

Identifies switch statements or cascading if-else chains that could
be replaced with polymorphism using inheritance or strategy pattern.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("replace_conditional_with_polymorphism", category="refactoring", severity="medium")
@param("min_branches", int, default=4, description="Minimum branches to suggest")
@param("limit", int, default=100, description="Maximum results to return")
def replace_conditional_with_polymorphism() -> Pipeline:
    """
    Suggest Replace Conditional with Polymorphism opportunities.

    Returns:
        findings: List of conditionals that could use polymorphism
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?m ?name ?class_name ?file ?line ?branch_count
            WHERE {
                ?m type {Method} .
                ?m name ?name .
                ?m inFile ?file .
                ?m atLine ?line .
                OPTIONAL { ?m definedIn ?c . ?c name ?class_name }
                ?m branchCount ?branch_count .
                FILTER ( ?branch_count >= {min_branches} )
            }
            ORDER BY DESC(?branch_count)
            LIMIT {limit}
        ''')
        .select("name", "class_name", "file", "line", "branch_count", qualified_name="m")
        .map(lambda r: {
            **r,
            "refactoring": "replace_conditional_with_polymorphism",
            "message": f"Method '{r['name']}' has {r.get('branch_count', 0)} branches",
            "suggestion": "Consider using polymorphism (strategy pattern) instead"
        })
        .emit("findings")
    )
