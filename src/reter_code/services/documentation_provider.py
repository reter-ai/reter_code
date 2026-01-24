"""
Documentation Provider Service

Provides all documentation resources for the logical-thinking MCP server.
Extracted from LogicalThinkingServer as part of God Class refactoring.
"""

import os
from pathlib import Path
from typing import Dict


class DocumentationProvider:
    """
    Provides documentation resources for RETER, Python analysis, and refactoring recipes.

    This is a stateless service that serves documentation content.
    Uses a resource registry pattern to avoid code duplication.

    Responsibilities:
    - Serve usage guides and reference documentation
    - Provide refactoring recipe chapters
    - Offer syntax and grammar references
    """

    # Resource registry mapping resource names to file paths and descriptions
    RESOURCES: Dict[str, Dict[str, str]] = {
        "usage_guide": {
            "file": "AI_AGENT_USAGE_GUIDE.md",
            "desc": "AI Agent Usage Guide - READ THIS FIRST"
        },
        "refactoring_summary": {
            "file": "AI_AGENT_USAGE_GUIDE.md",  # Redirects to usage guide
            "desc": "v2.0.0 Refactoring Summary"
        },
        "grammar_reference": {
            "file": "GRAMMAR_REFERENCE.md",
            "desc": "Complete ANTLR4 Grammar Reference"
        },
        "python_analysis_reference": {
            "file": "python/PYTHON_ANALYSIS_REFERENCE.md",
            "desc": "Python Code Analysis Capabilities Reference"
        },
        "python_tools_reference": {
            "file": "PYTHON_TOOLS_REFERENCE.md",
            "desc": "Python Analysis Tools API Reference"
        },
        "python_query_patterns": {
            "file": "PYTHON_QUERY_PATTERNS.md",
            "desc": "Common Python Query Patterns"
        },
        "custom_knowledge_guide": {
            "file": "CUSTOM_KNOWLEDGE_GUIDE.md",
            "desc": "Adding Custom Domain Knowledge"
        },
        "syntax_quick_reference": {
            "file": "SYNTAX_QUICK_REFERENCE.md",
            "desc": "Syntax Quick Reference Card"
        },
        "refactoring_recipes_index": {
            "file": "rtp/REFACTORING_RECIPES_INDEX.md",
            "desc": "Martin Fowler's Refactoring Recipes - Index"
        },
        "refactoring_recipes_access": {
            "file": "rtp/REFACTORING_RECIPES_ACCESS.md",
            "desc": "How to Access Refactoring Recipe Chapters"
        },
        "automatic_snapshots_doc": {
            "file": "system/AUTOMATIC_SNAPSHOTS.md",
            "desc": "Automatic Snapshot Persistence"
        },
        "multiple_instances_doc": {
            "file": "system/MULTIPLE_INSTANCES.md",
            "desc": "Managing Multiple RETER Instances"
        },
        "source_management_doc": {
            "file": "system/SOURCE_MANAGEMENT.md",
            "desc": "Knowledge Source Management"
        },
        "thread_safety_doc": {
            "file": "system/THREAD_SAFETY.md",
            "desc": "Thread Safety and Concurrency"
        },
        "plugin_system_guide": {
            "file": "PLUGIN_SYSTEM_GUIDE.md",
            "desc": "Plugin System Guide (Phase 5.1-5.6)"
        },
        "default_instance_guide": {
            "file": "system/DEFAULT_INSTANCE.md",
            "desc": "Default Instance - Auto-Syncing Project Analysis"
        },
        "session_context_guide": {
            "file": "SESSION_CONTEXT_GUIDE.md",
            "desc": "Session Context - MUST Call at Session Start"
        },
        "recommendations_plugin_guide": {
            "file": "RECOMMENDATIONS_PLUGIN_GUIDE.md",
            "desc": "Recommendations Plugin - Session Continuity & Progress Tracking"
        },
    }


    def __init__(self):
        """Initialize the documentation provider (stateless)."""
        # Calculate resources directory relative to this file
        # Resources are now inside the package: src/reter_code/resources/
        self.resources_dir = Path(__file__).parent.parent / "resources"

    def load_resource(self, resource_name: str) -> str:
        """
        Load a documentation resource by name.

        Args:
            resource_name: Name of the resource (key from RESOURCES dict)

        Returns:
            Resource content as string
        """
        if resource_name not in self.RESOURCES:
            return f"Resource '{resource_name}' not found."

        resource_info = self.RESOURCES[resource_name]
        file_path = self.resources_dir / resource_info["file"]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"{resource_info['desc']} not found. Please ensure {resource_info['file']} exists in resources/"
        except Exception as e:
            return f"Error loading {resource_info['file']}: {e}"

    # Individual resource methods for backward compatibility
    # These delegate to the generic load_resource method

    def get_usage_guide(self) -> str:
        """Get the AI Agent Usage Guide."""
        return self.load_resource("usage_guide")

    def get_refactoring_summary(self) -> str:
        """Get the refactoring summary document."""
        return self.load_resource("refactoring_summary")

    def get_grammar_reference(self) -> str:
        """Get the complete ANTLR4 grammar reference."""
        return self.load_resource("grammar_reference")

    def get_python_analysis_reference(self) -> str:
        """Get the Python analysis capabilities reference."""
        return self.load_resource("python_analysis_reference")

    def get_python_tools_reference(self) -> str:
        """Get the Python tools API reference."""
        return self.load_resource("python_tools_reference")

    def get_python_query_patterns(self) -> str:
        """Get common Python query patterns."""
        return self.load_resource("python_query_patterns")

    def get_custom_knowledge_guide(self) -> str:
        """Get the custom knowledge guide."""
        return self.load_resource("custom_knowledge_guide")

    def get_syntax_quick_reference(self) -> str:
        """Get the syntax quick reference card."""
        return self.load_resource("syntax_quick_reference")

    def get_refactoring_recipes_index(self) -> str:
        """Get the refactoring recipes index."""
        return self.load_resource("refactoring_recipes_index")

    def get_refactoring_recipes_access(self) -> str:
        """Get instructions for accessing recipe chapters."""
        return self.load_resource("refactoring_recipes_access")

    def get_automatic_snapshots_doc(self) -> str:
        """Get documentation on automatic snapshots."""
        return self.load_resource("automatic_snapshots_doc")

    def get_multiple_instances_doc(self) -> str:
        """Get documentation on managing multiple instances."""
        return self.load_resource("multiple_instances_doc")

    def get_source_management_doc(self) -> str:
        """Get documentation on knowledge source management."""
        return self.load_resource("source_management_doc")

    def get_thread_safety_doc(self) -> str:
        """Get documentation on thread safety."""
        return self.load_resource("thread_safety_doc")

    def get_plugin_system_guide(self) -> str:
        """Get comprehensive plugin system guide (Phase 5.1-5.6)."""
        return self.load_resource("plugin_system_guide")

    def get_default_instance_guide(self) -> str:
        """Get the default instance guide."""
        return self.load_resource("default_instance_guide")

    def get_session_context_guide(self) -> str:
        """Get the session context guide - MUST call at session start."""
        return self.load_resource("session_context_guide")

    def get_recommendations_plugin_guide(self) -> str:
        """Get the recommendations plugin guide."""
        return self.load_resource("recommendations_plugin_guide")

    def get_refactoring_recipe(self, chapter_id: str) -> str:
        """
        Get a refactoring recipe chapter by ID.

        Args:
            chapter_id: Chapter identifier like 'chapter-01', 'chapter-02', etc.

        Returns:
            Chapter content as string
        """
        # Map chapter-XX to actual file names
        chapter_files = {
            "chapter-01": "rtp/recipe_chapter_01_first_example.md",
            "chapter-02": "rtp/recipe_chapter_02_principles.md",
            "chapter-03": "rtp/recipe_chapter_03_bad_smells.md",
            "chapter-04": "rtp/recipe_chapter_04_building_tests.md",
            "chapter-05": "rtp/recipe_chapter_05_catalog.md",
            "chapter-06": "rtp/recipe_chapter_06_first_refactorings.md",
            "chapter-07": "rtp/recipe_chapter_07_encapsulation.md",
            "chapter-08": "rtp/recipe_chapter_08_moving_features.md",
            "chapter-09": "rtp/recipe_chapter_09_organizing_data.md",
            "chapter-10": "rtp/recipe_chapter_10_conditional_logic.md",
            "chapter-11": "rtp/recipe_chapter_11_refactoring_apis.md",
            "chapter-12": "rtp/recipe_chapter_12_inheritance.md",
        }

        if chapter_id not in chapter_files:
            return f"Chapter '{chapter_id}' not found. Valid: chapter-01 to chapter-12"

        file_path = self.resources_dir / chapter_files[chapter_id]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Chapter file not found: {chapter_files[chapter_id]}"
        except Exception as e:
            return f"Error loading chapter: {e}"
