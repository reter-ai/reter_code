"""
Resource Registrar Service

Handles registration and management of documentation resources.
Extracted from LogicalThinkingServer as part of Extract Class refactoring (Fowler Ch. 7).
"""

from typing import Dict, Any, Callable
from mcp.server.fastmcp import FastMCP
from .documentation_provider import DocumentationProvider


class ResourceRegistrar:
    """
    Manages MCP resource registration for documentation.
    Single Responsibility: Register and serve documentation resources.
    """

    def __init__(self, doc_provider: DocumentationProvider):
        """
        Initialize ResourceRegistrar with a documentation provider.

        Args:
            doc_provider: Documentation provider service
        """
        self.doc_provider = doc_provider
        self._resources: Dict[str, Callable] = {}

    def register_all_resources(self, app: FastMCP) -> None:
        """
        Register all documentation resources with the MCP app.

        Args:
            app: FastMCP application instance
        """
        # Core documentation resources
        self._register_core_resources(app)

        # Refactoring recipe resources
        self._register_recipe_resources(app)

        # System documentation resources
        self._register_system_resources(app)

    def _register_core_resources(self, app: FastMCP) -> None:
        """Register core documentation resources."""

        @app.resource("guide://logical-thinking/usage")
        def get_usage_guide() -> str:
            """AI Agent Usage Guide - READ THIS FIRST"""
            return self.doc_provider.get_usage_guide()

        @app.resource("guide://logical-thinking/refactoring")
        def get_refactoring_summary() -> str:
            """v2.0.0 Refactoring Summary"""
            return self.doc_provider.get_refactoring_summary()

        @app.resource("grammar://reter/dl-reql")
        def get_grammar_reference() -> str:
            """Complete ANTLR4 Grammar Reference"""
            return self.doc_provider.get_grammar_reference()

        @app.resource("python://reter/analysis")
        def get_python_analysis_reference() -> str:
            """Python Code Analysis Capabilities Reference"""
            return self.doc_provider.get_python_analysis_reference()

        @app.resource("python://reter/query-patterns")
        def get_python_query_patterns() -> str:
            """Common Python Query Patterns"""
            return self.doc_provider.get_python_query_patterns()

        @app.resource("python://reter/tools")
        def get_python_tools_reference() -> str:
            """Python Analysis Tools API Reference"""
            return self.doc_provider.get_python_tools_reference()

        @app.resource("guide://reter/custom-knowledge")
        def get_custom_knowledge_guide() -> str:
            """Creating Custom Knowledge in RETER"""
            return self.doc_provider.get_custom_knowledge_guide()

        @app.resource("reference://reter/syntax-quick")
        def get_syntax_quick_reference() -> str:
            """RETER Syntax Quick Reference Card"""
            return self.doc_provider.get_syntax_quick_reference()

    def _register_recipe_resources(self, app: FastMCP) -> None:
        """Register refactoring recipe resources."""

        @app.resource("recipe://refactoring/index")
        def get_refactoring_recipes_index() -> str:
            """Refactoring Recipes Index"""
            return self.doc_provider.get_refactoring_recipes_index()

        @app.resource("recipe://refactoring/access-guide")
        def get_refactoring_recipes_access() -> str:
            """How to Access Refactoring Recipes"""
            return self.doc_provider.get_refactoring_recipes_access()

        # Register chapter resources
        chapters = [
            ("01", "First Example"),
            ("02", "Refactoring Principles"),
            ("03", "Bad Smells in Code"),
            ("04", "Building Tests"),
            ("05", "Catalog Reference"),
            ("06", "First Refactorings"),
            ("07", "Encapsulation"),
            ("08", "Moving Features"),
            ("09", "Organizing Data"),
            ("10", "Simplifying Conditional Logic"),
            ("11", "Refactoring APIs"),
            ("12", "Dealing with Inheritance")
        ]

        for chapter_num, chapter_title in chapters:
            self._register_chapter_resource(app, chapter_num, chapter_title)

    def _register_chapter_resource(self, app: FastMCP, chapter_num: str, chapter_title: str) -> None:
        """Register a single chapter resource."""

        @app.resource(f"recipe://refactoring/chapter-{chapter_num}")
        def get_recipe_chapter() -> str:
            f"""Chapter {int(chapter_num)}: {chapter_title}"""
            return self.doc_provider.get_refactoring_recipe(f"chapter-{chapter_num}")

        # Store reference to avoid closure issues
        get_recipe_chapter.__name__ = f"get_recipe_chapter_{chapter_num}"

    def _register_system_resources(self, app: FastMCP) -> None:
        """Register system documentation resources."""

        @app.resource("system://reter/snapshots")
        def get_automatic_snapshots_doc() -> str:
            """RETER Automatic Snapshots Documentation"""
            return self.doc_provider.get_automatic_snapshots_doc()

        @app.resource("system://reter/multiple-instances")
        def get_multiple_instances_doc() -> str:
            """RETER Multiple Instances Documentation"""
            return self.doc_provider.get_multiple_instances_doc()

        @app.resource("system://reter/source-management")
        def get_source_management_doc() -> str:
            """RETER Source Management Documentation"""
            return self.doc_provider.get_source_management_doc()

        @app.resource("system://reter/thread-safety")
        def get_thread_safety_doc() -> str:
            """RETER Thread Safety Documentation"""
            return self.doc_provider.get_thread_safety_doc()

        @app.resource("guide://plugins/system")
        def get_plugin_system_guide() -> str:
            """Plugin System Guide (Phase 5.1-5.7 Complete)"""
            return self.doc_provider.get_plugin_system_guide()

        @app.resource("guide://reter/default-instance")
        def get_default_instance_guide() -> str:
            """Default Instance - Auto-Syncing Project Analysis"""
            return self.doc_provider.get_default_instance_guide()

        @app.resource("guide://reter/session-context")
        def get_session_context_guide() -> str:
            """Session Context - MUST Call at Session Start"""
            return self.doc_provider.get_session_context_guide()

        @app.resource("guide://reter/recommendations")
        def get_recommendations_plugin_guide() -> str:
            """Recommendations Plugin - Session Continuity & Progress Tracking"""
            return self.doc_provider.get_recommendations_plugin_guide()