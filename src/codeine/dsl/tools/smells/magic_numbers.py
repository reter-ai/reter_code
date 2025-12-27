"""
magic_numbers - Detect numeric literals that should be named constants.

Magic numbers are unexplained numeric literals in code that make it harder
to understand intent and maintain consistency.

Implementation: Uses parameter default values as a proxy for magic numbers.
"""

from codeine.dsl import detector, param, reql, Pipeline


@detector("magic_numbers", category="code_smell", severity="medium")
@param("exclude_common", bool, default=True, description="Exclude common values like 0, 1, -1")
@param("min_occurrences", int, default=2, description="Minimum occurrences to report")
@param("limit", int, default=100, description="Maximum results to return")
def magic_numbers() -> Pipeline:
    """
    Detect magic numbers - numeric default values that appear multiple times.

    Uses parameter default values as a proxy since numeric literals
    are not directly tracked in the ontology.

    Returns:
        findings: List of repeated numeric default values
        count: Number of findings
    """
    return (
        reql('''
            SELECT ?value (COUNT(?p) AS ?occurrences)
            WHERE {
                ?p type {Parameter} .
                ?p defaultValue ?value .
                FILTER ( REGEX(?value, "^-?[0-9]+\\.?[0-9]*$") )
                FILTER ( !{exclude_common} || (?value != "0" && ?value != "1" && ?value != "-1" && ?value != "2") )
            }
            GROUP BY ?value
            HAVING ( ?occurrences >= {min_occurrences} )
            ORDER BY DESC(?occurrences)
            LIMIT {limit}
        ''')
        .select("value", "occurrences")
        .map(lambda r: {
            **r,
            "issue": "magic_number",
            "message": f"Numeric value '{r['value']}' used as default in {r['occurrences']} parameters",
            "suggestion": "Consider extracting to a named constant"
        })
        .emit("findings")
    )
