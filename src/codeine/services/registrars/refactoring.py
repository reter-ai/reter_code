"""
Recommender Tools Registrar

Generic recommender tool that dispatches to CADSL detector tools.
Uses the Registry to discover available detectors.

Usage:
- recommender() - lists all recommender categories
- recommender("smells") - queues tasks for all smell detectors
- recommender("smells", "large_classes") - runs specific detector

Available categories (from CADSL tools):
- smells: Code smell detection (large_classes, god_class, etc.)
- refactoring: Refactoring opportunities (extract_method, etc.)
- inheritance: Inheritance refactoring (pull_up_method, etc.)
- testing: Test coverage analysis (untested_classes, etc.)
- dependencies: Dependency analysis (circular_imports, etc.)
- exceptions: Exception handling issues
- patterns: Design pattern detection
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


# Map recommender types to CADSL tool module categories
# Some types map to multiple categories
RECOMMENDER_TYPES = {
    "smells": ["size", "complexity", "design", "maintenance", "code_smell"],  # All smell categories
    "refactoring": ["refactoring"],
    "inheritance": ["inheritance"],
    "testing": ["test_coverage"],
    "dependencies": ["dependencies"],
    "exceptions": ["exception_handling"],
    "patterns": ["patterns"],
    "duplication": ["duplication"],
}


class RecommenderToolsRegistrar(ToolRegistrarBase):
    """Registers the unified recommender tool with FastMCP."""

    def __init__(self, instance_manager, persistence_service, default_manager=None):
        super().__init__(instance_manager, persistence_service)
        self._default_manager = default_manager

    def register(self, app: FastMCP) -> None:
        """Register the recommender tool."""
        from ...dsl.core import Context
        from ...dsl.registry import Registry

        instance_manager = self.instance_manager
        default_manager = self._default_manager

        @app.tool()
        @truncate_mcp_response
        def recommender(
            recommender_type: Optional[str] = None,
            detector_name: Optional[str] = None,
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

            **recommender()**: Queues tasks for each recommender type
            **recommender("refactoring")**: Queues tasks for all detectors
            **recommender("refactoring", "god_class")**: Runs specific detector

            Args:
                recommender_type: Type of recommender. If None, queues tasks for all types.
                detector_name: Specific detector to run. If None, queues all detectors as tasks.
                session_instance: Session instance for storing tasks
                categories: Filter detectors by category
                severities: Filter by severity: low, medium, high
                detector_type: "all", "improving", or "patterns" (refactoring only)
                params: Override detector defaults
                create_tasks: Auto-create tasks from high-priority findings
                link_to_thought: Link recommendations to a thought ID

            Returns:
                Without recommender_type: Tasks created for each recommender type
                Without detector_name: Tasks created for each detector
                With detector_name: Detection results and recommendations
            """
            # Recommender requires RETER to be ready
            try:
                require_default_instance()
            except ComponentNotReadyError as e:
                return e.to_response()

            # NO TYPE: List all recommender categories
            if recommender_type is None:
                return {
                    "success": True,
                    "mode": "list_types",
                    "available_types": list(RECOMMENDER_TYPES.keys()),
                    "descriptions": {
                        "smells": "Code smell detection (large classes, god class, etc.)",
                        "refactoring": "Refactoring opportunities (extract method, etc.)",
                        "inheritance": "Inheritance refactoring (pull up/push down method)",
                        "testing": "Test coverage analysis (untested classes/functions)",
                        "dependencies": "Dependency analysis (circular imports, unused imports)",
                        "exceptions": "Exception handling issues",
                        "patterns": "Design pattern detection",
                        "duplication": "Code duplication detection (RAG-based)",
                    },
                    "_tip": "Call recommender('smells') to see available detectors",
                }

            # Validate recommender type
            if recommender_type not in RECOMMENDER_TYPES:
                return {
                    "success": False,
                    "error": f"Unknown recommender type: '{recommender_type}'",
                    "available_types": list(RECOMMENDER_TYPES.keys())
                }

            # Get RETER instance
            try:
                reter = instance_manager.get_or_create_instance("default")
            except DefaultInstanceNotInitialised as e:
                return {"success": False, "error": str(e), "status": "initializing"}
            except Exception as e:
                return {"success": False, "error": f"Failed to get RETER instance: {str(e)}"}

            # Ensure CADSL tools are loaded (triggers Registry registration)
            _ensure_cadsl_tools_loaded(recommender_type)

            # Get detectors for this category from Registry
            categories = RECOMMENDER_TYPES[recommender_type]
            detectors = []
            for category in categories:
                detectors.extend(Registry.get_by_category(category))

            # Filter by severity if specified
            if severities:
                detectors = [d for d in detectors if d.meta.get("severity") in severities]

            detector_names = [d.name for d in detectors]

            # NO DETECTOR: List available detectors for this type
            if detector_name is None:
                detector_info = []
                for spec in detectors:
                    detector_info.append({
                        "name": spec.name,
                        "description": spec.description,
                        "severity": spec.meta.get("severity", "medium"),
                        "command": f"recommender('{recommender_type}', '{spec.name}')"
                    })

                return {
                    "success": True,
                    "recommender_type": recommender_type,
                    "mode": "list_detectors",
                    "detectors": detector_info,
                    "count": len(detector_info),
                    "_tip": f"Call recommender('{recommender_type}', 'detector_name') to run a specific detector",
                }

            # DETECTOR MODE: Run specific detector
            if detector_name not in detector_names:
                return {
                    "success": False,
                    "error": f"Unknown detector: '{detector_name}'",
                    "available_detectors": detector_names,
                    "hint": f"Call recommender('{recommender_type}') to see all available detectors"
                }

            # Find the detector tool and execute it
            detector_tool = None
            for spec in detectors:
                if spec.name == detector_name:
                    detector_tool = spec
                    break

            if not detector_tool:
                return {
                    "success": False,
                    "error": f"Detector '{detector_name}' not found in registry"
                }

            # Build context with rag_manager if available
            extra = params or {}
            if default_manager:
                rag_mgr = default_manager.get_rag_manager()
                if rag_mgr:
                    extra = {**extra, "rag_manager": rag_mgr}
            ctx = Context(
                reter=reter,
                params=extra,
                instance_name=session_instance
            )

            try:
                # Get the tool function from the appropriate CADSL module
                tool_module = _get_tool_module(recommender_type)
                if tool_module and hasattr(tool_module, detector_name):
                    tool_func = getattr(tool_module, detector_name)
                    result = tool_func(ctx)

                    # Add metadata
                    result["recommender_type"] = recommender_type
                    result["detector"] = detector_name

                    return result
                else:
                    return {
                        "success": False,
                        "error": f"Tool function '{detector_name}' not found in module"
                    }

            except Exception as e:
                import traceback
                return {
                    "success": False,
                    "error": str(e),
                    "detector": detector_name,
                    "traceback": traceback.format_exc()
                }


def _get_tool_module(recommender_type: str):
    """Get the CADSL tool module for a recommender type."""
    from ...cadsl.tools_bridge import (
        smells,
        refactoring,
        inheritance,
        testing,
        dependencies,
        exceptions,
        patterns,
        rag,
    )

    modules = {
        "smells": smells,
        "refactoring": refactoring,
        "inheritance": inheritance,
        "testing": testing,
        "dependencies": dependencies,
        "exceptions": exceptions,
        "patterns": patterns,
        "duplication": rag,  # RAG tools include duplication detection
    }

    return modules.get(recommender_type)


def _ensure_cadsl_tools_loaded(recommender_type: str):
    """
    Ensure CADSL tool modules are loaded and registered with Registry.

    This triggers the lazy loading of tool modules which registers them
    with the global Registry for discovery.
    """
    from ...cadsl.tools_bridge import (
        smells,
        refactoring,
        inheritance,
        testing,
        dependencies,
        exceptions,
        patterns,
        rag,
    )

    # Map recommender types to modules that need to be loaded
    type_to_modules = {
        "smells": [smells],
        "refactoring": [refactoring],
        "inheritance": [inheritance],
        "testing": [testing],
        "dependencies": [dependencies],
        "exceptions": [exceptions],
        "patterns": [patterns],
        "duplication": [rag],
    }

    modules = type_to_modules.get(recommender_type, [])
    for module in modules:
        # Accessing list_tools triggers _ensure_loaded
        module.list_tools()
