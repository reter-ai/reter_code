"""
Recommender Tools Registrar

Generic recommender tool that dispatches to specific recommender types.

Usage:
- recommender("refactoring") - lists all refactoring detectors
- recommender("refactoring", "god_class") - runs specific detector
- recommender("test_coverage") - lists all test coverage detectors
- recommender("test_coverage", "untested_classes") - runs specific detector
- recommender("documentation_maintenance") - lists all documentation detectors
- recommender("documentation_maintenance", "orphaned_sections") - runs specific detector
"""

from typing import Dict, Any, Optional, List
from fastmcp import FastMCP
from .base import ToolRegistrarBase, truncate_mcp_response
from ...reter_wrapper import DefaultInstanceNotInitialised, is_initialization_complete
from ..initialization_progress import (
    get_initializing_response,
    require_default_instance,
    ComponentNotReadyError,
)


# Available recommender types
RECOMMENDER_TYPES = ["refactoring", "test_coverage", "documentation_maintenance"]


class RecommenderToolsRegistrar(ToolRegistrarBase):
    """Registers the unified recommender tool with FastMCP."""

    def __init__(self, instance_manager, persistence_service, default_manager=None):
        super().__init__(instance_manager, persistence_service)
        self._default_manager = default_manager

    def register(self, app: FastMCP) -> None:
        """Register the recommender tool."""
        from ...tools.refactoring_improving.tool import RefactoringTool, DETECTORS
        from ...tools.refactoring_to_patterns.tool import RefactoringToPatternsTool, PATTERN_DETECTORS
        from ...tools.test_coverage.tool import TestCoverageTool, DETECTORS as TEST_COVERAGE_DETECTORS
        from ...tools.documentation_maintenance.tool import (
            DocumentationMaintenanceTool, DETECTORS as DOC_MAINTENANCE_DETECTORS
        )

        improving_tool = RefactoringTool(self.instance_manager)
        patterns_tool = RefactoringToPatternsTool(self.instance_manager)
        test_coverage_tool = TestCoverageTool(self.instance_manager)
        doc_maintenance_tool = DocumentationMaintenanceTool(
            self.instance_manager, self._default_manager
        )

        @app.tool()
        @truncate_mcp_response
        def recommender(
            recommender_type: str,
            detector_name: Optional[str] = None,
            instance_name: str = "default",
            session_instance: str = "default",
            categories: Optional[List[str]] = None,
            severities: Optional[List[str]] = None,
            detector_type: str = "all",
            params: Optional[Dict[str, Any]] = None,
            create_tasks: bool = False,
            link_to_thought: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Unified recommender tool for code analysis and recommendations.

            **See: recipe://refactoring/index for refactoring recipes**

            **recommender("refactoring")**: Lists all refactoring detectors
            **recommender("refactoring", "god_class")**: Runs specific detector

            Args:
                recommender_type: Type of recommender ("refactoring", "test_coverage", "documentation_maintenance")
                detector_name: Specific detector to run. If None, lists available.
                instance_name: RETER instance to analyze
                session_instance: Session instance for storing recommendations
                categories: Filter detectors by category
                severities: Filter by severity: low, medium, high
                detector_type: "all", "improving", or "patterns" (refactoring only)
                params: Override detector defaults
                create_tasks: Auto-create tasks from high-priority findings
                link_to_thought: Link recommendations to a thought ID

            Returns:
                Without detector_name: List of available detectors
                With detector_name: Detection results and recommendations
            """
            # Recommender requires RETER to be ready
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            # Validate recommender type
            if recommender_type not in RECOMMENDER_TYPES:
                return {
                    "success": False,
                    "error": f"Unknown recommender type: '{recommender_type}'",
                    "available_types": RECOMMENDER_TYPES
                }

            # REFACTORING RECOMMENDER
            if recommender_type == "refactoring":
                return _run_refactoring_recommender(
                    improving_tool=improving_tool,
                    patterns_tool=patterns_tool,
                    detectors=DETECTORS,
                    pattern_detectors=PATTERN_DETECTORS,
                    detector_name=detector_name,
                    instance_name=instance_name,
                    session_instance=session_instance,
                    categories=categories,
                    severities=severities,
                    detector_type=detector_type,
                    params=params,
                    create_tasks=create_tasks,
                    link_to_thought=link_to_thought
                )

            # TEST COVERAGE RECOMMENDER
            if recommender_type == "test_coverage":
                return _run_test_coverage_recommender(
                    tool=test_coverage_tool,
                    detectors=TEST_COVERAGE_DETECTORS,
                    detector_name=detector_name,
                    instance_name=instance_name,
                    session_instance=session_instance,
                    categories=categories,
                    severities=severities,
                    params=params,
                    create_tasks=create_tasks,
                    link_to_thought=link_to_thought
                )

            # DOCUMENTATION MAINTENANCE RECOMMENDER
            if recommender_type == "documentation_maintenance":
                return _run_documentation_maintenance_recommender(
                    tool=doc_maintenance_tool,
                    detectors=DOC_MAINTENANCE_DETECTORS,
                    detector_name=detector_name,
                    instance_name=instance_name,
                    session_instance=session_instance,
                    categories=categories,
                    severities=severities,
                    params=params,
                    create_tasks=create_tasks,
                    link_to_thought=link_to_thought
                )

            # Future recommender types go here
            return {"success": False, "error": f"Recommender '{recommender_type}' not implemented"}


def _run_refactoring_recommender(
    improving_tool,
    patterns_tool,
    detectors,
    pattern_detectors,
    detector_name: Optional[str],
    instance_name: str,
    session_instance: str,
    categories: Optional[List[str]],
    severities: Optional[List[str]],
    detector_type: str,
    params: Optional[Dict[str, Any]],
    create_tasks: bool,
    link_to_thought: Optional[str]
) -> Dict[str, Any]:
    """Run the refactoring recommender."""

    # PREPARE MODE: List available detectors
    if detector_name is None:
        results = {
            "success": True,
            "recommender_type": "refactoring",
            "mode": "prepare",
            "detector_type": detector_type,
            "improving": None,
            "patterns": None,
            "total_detectors": 0,
            "total_recommendations": 0,
            "session_instance": session_instance,
            "_resources": {
                "recipe://refactoring/index": "Refactoring Recipes Index",
                "recipe://refactoring/chapter-03": "Bad Smells in Code"
            }
        }

        # Get improving detectors
        if detector_type in ("all", "improving"):
            improving_result = improving_tool.prepare(
                instance_name=instance_name,
                categories=categories,
                severities=severities,
                session_instance=session_instance
            )
            results["improving"] = {
                "detectors": improving_result.get("detectors", []),
                "count": improving_result.get("detector_count", 0),
                "recommendations_created": improving_result.get("recommendations_created", 0)
            }
            results["total_detectors"] += results["improving"]["count"]
            results["total_recommendations"] += results["improving"]["recommendations_created"]

        # Get pattern detectors
        if detector_type in ("all", "patterns"):
            patterns_result = patterns_tool.prepare(
                instance_name=instance_name,
                categories=categories,
                severities=severities,
                session_instance=session_instance
            )
            results["patterns"] = {
                "detectors": patterns_result.get("detectors", []),
                "count": patterns_result.get("detector_count", 0),
                "recommendations_created": patterns_result.get("recommendations_created", 0)
            }
            results["total_detectors"] += results["patterns"]["count"]
            results["total_recommendations"] += results["patterns"]["recommendations_created"]

        return results

    # DETECTOR MODE: Run specific detector
    if detector_name in detectors:
        return improving_tool.detector(
            detector_name=detector_name,
            instance_name=instance_name,
            params=params,
            session_instance=session_instance,
            create_tasks=create_tasks,
            link_to_thought=link_to_thought
        )
    elif detector_name in pattern_detectors:
        return patterns_tool.detector(
            detector_name=detector_name,
            instance_name=instance_name,
            params=params,
            session_instance=session_instance,
            create_tasks=create_tasks,
            link_to_thought=link_to_thought
        )
    else:
        # Unknown detector
        all_detectors = list(detectors.keys()) + list(pattern_detectors.keys())
        return {
            "success": False,
            "error": f"Unknown detector: '{detector_name}'",
            "available_detectors": all_detectors,
            "hint": "Call recommender('refactoring') to see all available detectors"
        }


def _run_test_coverage_recommender(
    tool,
    detectors,
    detector_name: Optional[str],
    instance_name: str,
    session_instance: str,
    categories: Optional[List[str]],
    severities: Optional[List[str]],
    params: Optional[Dict[str, Any]],
    create_tasks: bool,
    link_to_thought: Optional[str]
) -> Dict[str, Any]:
    """Run the test coverage recommender."""

    # PREPARE MODE: List available detectors
    if detector_name is None:
        result = tool.prepare(
            instance_name=instance_name,
            categories=categories,
            severities=severities,
            session_instance=session_instance
        )
        result["recommender_type"] = "test_coverage"
        result["mode"] = "prepare"
        result["_resources"] = {
            "python://reter/tools": "Python Analysis Tools Reference",
            "recipe://refactoring/chapter-04": "Building Tests"
        }
        result["_tip"] = "Use code_inspection(action='find_tests', target='entity_name') to find existing tests"
        return result

    # DETECTOR MODE: Run specific detector
    if detector_name in detectors:
        result = tool.detector(
            detector_name=detector_name,
            instance_name=instance_name,
            params=params,
            session_instance=session_instance,
            create_tasks=create_tasks,
            link_to_thought=link_to_thought
        )
        result["recommender_type"] = "test_coverage"
        return result
    else:
        return {
            "success": False,
            "error": f"Unknown detector: '{detector_name}'",
            "available_detectors": list(detectors.keys()),
            "hint": "Call recommender('test_coverage') to see all available detectors"
        }


def _run_documentation_maintenance_recommender(
    tool,
    detectors,
    detector_name: Optional[str],
    instance_name: str,
    session_instance: str,
    categories: Optional[List[str]],
    severities: Optional[List[str]],
    params: Optional[Dict[str, Any]],
    create_tasks: bool,
    link_to_thought: Optional[str]
) -> Dict[str, Any]:
    """Run the documentation maintenance recommender."""

    # PREPARE MODE: List available detectors
    if detector_name is None:
        result = tool.prepare(
            instance_name=instance_name,
            categories=categories,
            severities=severities,
            session_instance=session_instance
        )
        result["recommender_type"] = "documentation_maintenance"
        result["mode"] = "prepare"
        result["_resources"] = {
            "python://reter/tools": "Python Analysis Tools Reference",
        }
        result["_tip"] = "Use analyze_documentation_relevance() for detailed analysis"
        return result

    # DETECTOR MODE: Run specific detector
    if detector_name in detectors:
        result = tool.detector(
            detector_name=detector_name,
            instance_name=instance_name,
            params=params,
            session_instance=session_instance,
            create_tasks=create_tasks,
            link_to_thought=link_to_thought
        )
        result["recommender_type"] = "documentation_maintenance"
        return result
    else:
        return {
            "success": False,
            "error": f"Unknown detector: '{detector_name}'",
            "available_detectors": list(detectors.keys()),
            "hint": "Call recommender('documentation_maintenance') to see all available detectors"
        }
