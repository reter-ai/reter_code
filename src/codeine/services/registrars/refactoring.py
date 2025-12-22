"""
Recommender Tools Registrar

Generic recommender tool that dispatches to specific recommender types.

Usage:
- recommender() - queues tasks for all 4 recommender types
- recommender("refactoring") - queues tasks for all refactoring detectors
- recommender("refactoring", "god_class") - runs specific detector

Available types: refactoring, test_coverage, documentation_maintenance, redundancy_reduction

Queue mode creates tasks in SQL storage that can be viewed with:
  items(action='list', source_tool='recommender')
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
RECOMMENDER_TYPES = ["refactoring", "test_coverage", "documentation_maintenance", "redundancy_reduction"]


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
        from ...tools.redundancy_reduction.tool import (
            RedundancyReductionTool, DETECTORS as REDUNDANCY_DETECTORS
        )

        improving_tool = RefactoringTool(self.instance_manager)
        patterns_tool = RefactoringToPatternsTool(self.instance_manager)
        test_coverage_tool = TestCoverageTool(self.instance_manager)
        doc_maintenance_tool = DocumentationMaintenanceTool(
            self.instance_manager, self._default_manager
        )
        redundancy_tool = RedundancyReductionTool(
            self.instance_manager, self._default_manager
        )

        @app.tool()
        @truncate_mcp_response
        def recommender(
            recommender_type: Optional[str] = None,
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

            **recommender()**: Queues tasks for each recommender type
            **recommender("refactoring")**: Queues tasks for all detectors
            **recommender("refactoring", "god_class")**: Runs specific detector

            Args:
                recommender_type: Type of recommender. If None, queues tasks for all types.
                detector_name: Specific detector to run. If None, queues all detectors as tasks.
                instance_name: RETER instance to analyze
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

            # NO TYPE: Queue tasks for each recommender type
            if recommender_type is None:
                from ...tools.unified.store import UnifiedStore
                store = UnifiedStore()
                session_id = store.get_or_create_session(session_instance)

                results = {
                    "success": True,
                    "mode": "queue_types",
                    "tasks_created": [],
                    "total_tasks": 0,
                    "session_instance": session_instance,
                    "_tip": "Run items(action='list', source_tool='recommender') to see queued tasks",
                }

                for rtype in RECOMMENDER_TYPES:
                    task_id = store.add_item(
                        session_id=session_id,
                        item_type="task",
                        content=f"Run {rtype} recommender",
                        category="research",
                        priority="medium",
                        status="pending",
                        source_tool=f"recommender:{rtype}",
                        metadata={
                            "recommender_type": rtype,
                            "command": f"recommender('{rtype}')"
                        }
                    )
                    results["tasks_created"].append({
                        "task_id": task_id,
                        "recommender_type": rtype,
                        "command": f"recommender('{rtype}')"
                    })
                    results["total_tasks"] += 1

                return results

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

            # REDUNDANCY REDUCTION RECOMMENDER
            if recommender_type == "redundancy_reduction":
                return _run_redundancy_reduction_recommender(
                    tool=redundancy_tool,
                    detectors=REDUNDANCY_DETECTORS,
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

    # QUEUE MODE: Create tasks for each detector when no specific detector given
    if detector_name is None:
        store = improving_tool._get_unified_store()
        if not store:
            return {"success": False, "error": "Could not access unified store"}

        session_id = improving_tool._get_or_create_session(store, session_instance)
        if not session_id:
            return {"success": False, "error": "Could not create session"}

        # Mark parent task as completed (created by recommender())
        _complete_recommender_task(store, session_id, "refactoring")

        results = {
            "success": True,
            "recommender_type": "refactoring",
            "mode": "queue",
            "detector_type": detector_type,
            "tasks_created": [],
            "total_tasks": 0,
            "session_instance": session_instance,
            "_tip": "Run items(action='list', source_tool='recommender:refactoring') to see queued tasks",
            "_resources": {
                "recipe://refactoring/index": "Refactoring Recipes Index",
                "recipe://refactoring/chapter-03": "Bad Smells in Code"
            }
        }

        # Queue improving detectors
        if detector_type in ("all", "improving"):
            for det_name, det_info in detectors.items():
                task_id = store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=f"Run {det_name} detector: {det_info.get('description', '')}",
                    category="research",
                    priority="medium",
                    status="pending",
                    source_tool=f"recommender:refactoring:{det_name}",
                    metadata={
                        "recommender_type": "refactoring",
                        "detector_name": det_name,
                        "detector_type": "improving",
                        "instance_name": instance_name,
                        "command": f"recommender('refactoring', '{det_name}')"
                    }
                )
                results["tasks_created"].append({
                    "task_id": task_id,
                    "detector": det_name,
                    "type": "improving",
                    "description": det_info.get("description", "")
                })
                results["total_tasks"] += 1

        # Queue pattern detectors
        if detector_type in ("all", "patterns"):
            for det_name, det_info in pattern_detectors.items():
                task_id = store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=f"Run {det_name} detector: {det_info.get('description', '')}",
                    category="research",
                    priority="medium",
                    status="pending",
                    source_tool=f"recommender:refactoring:{det_name}",
                    metadata={
                        "recommender_type": "refactoring",
                        "detector_name": det_name,
                        "detector_type": "patterns",
                        "instance_name": instance_name,
                        "command": f"recommender('refactoring', '{det_name}')"
                    }
                )
                results["tasks_created"].append({
                    "task_id": task_id,
                    "detector": det_name,
                    "type": "patterns",
                    "description": det_info.get("description", "")
                })
                results["total_tasks"] += 1

        return results

    # DETECTOR MODE: Run specific detector
    if detector_name in detectors:
        # Mark detector task as completed
        store = improving_tool._get_unified_store()
        if store:
            session_id = improving_tool._get_or_create_session(store, session_instance)
            if session_id:
                _complete_detector_task(store, session_id, "refactoring", detector_name)

        return improving_tool.detector(
            detector_name=detector_name,
            instance_name=instance_name,
            params=params,
            session_instance=session_instance,
            create_tasks=create_tasks,
            link_to_thought=link_to_thought
        )
    elif detector_name in pattern_detectors:
        # Mark detector task as completed
        store = patterns_tool._get_unified_store()
        if store:
            session_id = patterns_tool._get_or_create_session(store, session_instance)
            if session_id:
                _complete_detector_task(store, session_id, "refactoring", detector_name)

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

    # QUEUE MODE: Create tasks for each detector when no specific detector given
    if detector_name is None:
        store = tool._get_unified_store()
        if not store:
            return {"success": False, "error": "Could not access unified store"}

        session_id = tool._get_or_create_session(store, session_instance)
        if not session_id:
            return {"success": False, "error": "Could not create session"}

        # Mark parent task as completed (created by recommender())
        _complete_recommender_task(store, session_id, "test_coverage")

        results = {
            "success": True,
            "recommender_type": "test_coverage",
            "mode": "queue",
            "tasks_created": [],
            "total_tasks": 0,
            "session_instance": session_instance,
            "_tip": "Run items(action='list', source_tool='recommender:test_coverage') to see queued tasks",
            "_resources": {
                "python://reter/tools": "Python Analysis Tools Reference",
                "recipe://refactoring/chapter-04": "Building Tests"
            }
        }

        for det_name, det_info in detectors.items():
            task_id = store.add_item(
                session_id=session_id,
                item_type="task",
                content=f"Run {det_name} detector: {det_info.get('description', '')}",
                category="research",
                priority="medium",
                status="pending",
                source_tool=f"recommender:test_coverage:{det_name}",
                metadata={
                    "recommender_type": "test_coverage",
                    "detector_name": det_name,
                    "instance_name": instance_name,
                    "command": f"recommender('test_coverage', '{det_name}')"
                }
            )
            results["tasks_created"].append({
                "task_id": task_id,
                "detector": det_name,
                "description": det_info.get("description", "")
            })
            results["total_tasks"] += 1

        return results

    # DETECTOR MODE: Run specific detector
    if detector_name in detectors:
        # Mark detector task as completed
        store = tool._get_unified_store()
        if store:
            session_id = tool._get_or_create_session(store, session_instance)
            if session_id:
                _complete_detector_task(store, session_id, "test_coverage", detector_name)

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

    # QUEUE MODE: Create tasks for each detector when no specific detector given
    if detector_name is None:
        store = tool._get_unified_store()
        if not store:
            return {"success": False, "error": "Could not access unified store"}

        session_id = tool._get_or_create_session(store, session_instance)
        if not session_id:
            return {"success": False, "error": "Could not create session"}

        # Mark parent task as completed (created by recommender())
        _complete_recommender_task(store, session_id, "documentation_maintenance")

        results = {
            "success": True,
            "recommender_type": "documentation_maintenance",
            "mode": "queue",
            "tasks_created": [],
            "total_tasks": 0,
            "session_instance": session_instance,
            "_tip": "Run items(action='list', source_tool='recommender:documentation_maintenance') to see queued tasks",
            "_resources": {
                "python://reter/tools": "Python Analysis Tools Reference",
            }
        }

        for det_name, det_info in detectors.items():
            task_id = store.add_item(
                session_id=session_id,
                item_type="task",
                content=f"Run {det_name} detector: {det_info.get('description', '')}",
                category="research",
                priority="medium",
                status="pending",
                source_tool=f"recommender:documentation_maintenance:{det_name}",
                metadata={
                    "recommender_type": "documentation_maintenance",
                    "detector_name": det_name,
                    "instance_name": instance_name,
                    "command": f"recommender('documentation_maintenance', '{det_name}')"
                }
            )
            results["tasks_created"].append({
                "task_id": task_id,
                "detector": det_name,
                "description": det_info.get("description", "")
            })
            results["total_tasks"] += 1

        return results

    # DETECTOR MODE: Run specific detector
    if detector_name in detectors:
        # Mark detector task as completed
        store = tool._get_unified_store()
        if store:
            session_id = tool._get_or_create_session(store, session_instance)
            if session_id:
                _complete_detector_task(store, session_id, "documentation_maintenance", detector_name)

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


def _run_redundancy_reduction_recommender(
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
    """Run the redundancy reduction recommender."""

    # QUEUE MODE: Create tasks for each detector when no specific detector given
    if detector_name is None:
        store = tool._get_unified_store()
        if not store:
            return {"success": False, "error": "Could not access unified store"}

        session_id = tool._get_or_create_session(store, session_instance)
        if not session_id:
            return {"success": False, "error": "Could not create session"}

        # Mark parent task as completed (created by recommender())
        _complete_recommender_task(store, session_id, "redundancy_reduction")

        results = {
            "success": True,
            "recommender_type": "redundancy_reduction",
            "mode": "queue",
            "tasks_created": [],
            "total_tasks": 0,
            "session_instance": session_instance,
            "_tip": "Run items(action='list', source_tool='recommender:redundancy_reduction') to see queued tasks",
            "_resources": {
                "python://reter/tools": "Python Analysis Tools Reference",
            }
        }

        for det_name, det_info in detectors.items():
            task_id = store.add_item(
                session_id=session_id,
                item_type="task",
                content=f"Run {det_name} detector: {det_info.get('description', '')}",
                category="research",
                priority="medium",
                status="pending",
                source_tool=f"recommender:redundancy_reduction:{det_name}",
                metadata={
                    "recommender_type": "redundancy_reduction",
                    "detector_name": det_name,
                    "instance_name": instance_name,
                    "command": f"recommender('redundancy_reduction', '{det_name}')"
                }
            )
            results["tasks_created"].append({
                "task_id": task_id,
                "detector": det_name,
                "description": det_info.get("description", "")
            })
            results["total_tasks"] += 1

        return results

    # DETECTOR MODE: Run specific detector
    if detector_name in detectors:
        # Mark detector task as completed
        store = tool._get_unified_store()
        if store:
            session_id = tool._get_or_create_session(store, session_instance)
            if session_id:
                _complete_detector_task(store, session_id, "redundancy_reduction", detector_name)

        result = tool.detector(
            detector_name=detector_name,
            instance_name=instance_name,
            params=params,
            session_instance=session_instance,
            create_tasks=create_tasks,
            link_to_thought=link_to_thought
        )
        result["recommender_type"] = "redundancy_reduction"
        return result
    else:
        return {
            "success": False,
            "error": f"Unknown detector: '{detector_name}'",
            "available_detectors": list(detectors.keys()),
            "hint": "Call recommender('redundancy_reduction') to see all available detectors"
        }


def _complete_recommender_task(store, session_id: str, recommender_type: str) -> bool:
    """Mark the parent recommender task as completed.

    When recommender("refactoring") is called, find and complete the task
    "Run refactoring recommender" that was created by recommender().
    """
    try:
        # Find task with source_tool="recommender:{type}" and status="pending"
        items = store.get_items(
            session_id=session_id,
            item_type="task",
            status="pending",
            source_tool=f"recommender:{recommender_type}",
            limit=1
        )
        if items:
            store.update_item(items[0]["item_id"], status="completed")
            return True
        return False
    except Exception:
        return False


def _complete_detector_task(store, session_id: str, recommender_type: str, detector_name: str) -> bool:
    """Mark the detector task as completed.

    When recommender("refactoring", "find_large_classes") is called, find and complete
    the task "Run find_large_classes detector" that was created by recommender("refactoring").
    """
    try:
        # Find task with source_tool="recommender:{type}:{detector}" and status="pending"
        items = store.get_items(
            session_id=session_id,
            item_type="task",
            status="pending",
            source_tool=f"recommender:{recommender_type}:{detector_name}",
            limit=1
        )
        if items:
            store.update_item(items[0]["item_id"], status="completed")
            return True
        return False
    except Exception:
        return False
