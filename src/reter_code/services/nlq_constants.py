"""
Constants for Natural Language Query (NLQ) tool.

Extracted from tool_registrar.py to reduce method size and improve maintainability.
Prompts and grammar are loaded from external files for easier maintenance.
"""

from pathlib import Path

# Resource directory
_RESOURCES_DIR = Path(__file__).parent.parent / "resources"


def _load_resource(filename: str) -> str:
    """Load a resource file from the resources directory."""
    resource_path = _RESOURCES_DIR / filename
    if resource_path.exists():
        with open(resource_path, 'r', encoding='utf-8') as f:
            return f.read()
    return f"# Resource file not found: {filename}"


def _build_prompts():
    """Load grammar and prompts, injecting grammar into prompt templates."""
    # Load the REQL Lark grammar
    grammar = _load_resource("REQL_GRAMMAR.lark")

    # Load prompt templates
    system_prompt_template = _load_resource("REQL_SYSTEM.prompt")
    syntax_help_template = _load_resource("REQL_SYNTAX_HELP.prompt")

    # Inject grammar into prompts
    system_prompt = system_prompt_template.replace("{GRAMMAR}", grammar)
    syntax_help = syntax_help_template.replace("{GRAMMAR}", grammar)

    return grammar, system_prompt, syntax_help


# Load all resources at module import
REQL_LARK_GRAMMAR, REQL_SYSTEM_PROMPT, REQL_SYNTAX_HELP = _build_prompts()
