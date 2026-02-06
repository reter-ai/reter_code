"""
RETER Code Inspection Handler.

Handles code inspection and diagram generation operations.

::: This is-in-layer Handler-Layer.
::: This is-in-component Query-Handlers.
::: This depends-on reter_code.services.code_inspector.
"""

from typing import Any, Dict, List

from . import BaseHandler
from ..protocol import METHOD_CODE_INSPECTION, METHOD_DIAGRAM, METHOD_RECOMMENDER


class InspectionHandler(BaseHandler):
    """Handler for code inspection operations.

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is stateful.
    """

    def _register_methods(self) -> None:
        """Register inspection method handlers."""
        self._methods = {
            METHOD_CODE_INSPECTION: self._handle_code_inspection,
            METHOD_DIAGRAM: self._handle_diagram,
            METHOD_RECOMMENDER: self._handle_recommender,
        }

    def can_handle(self, method: str) -> bool:
        """Check if this handler can process the method."""
        return method in self._methods

    def _handle_code_inspection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code inspection action.

        Params:
            action: Inspection action (list_classes, describe_class, find_usages, etc.)
            target: Target entity name
            module: Module filter
            limit: Maximum results
            offset: Pagination offset
            format: Output format (json, markdown, mermaid)
            include_methods: Include methods in class descriptions
            include_attributes: Include attributes in class descriptions
            include_docstrings: Include docstrings
            summary_only: Return summary only

        Returns:
            Action-specific results
        """
        action = params.get("action", "")

        if not action:
            raise ValueError("Action is required")

        # Import code inspector
        from ...services.code_inspector import CodeInspector

        inspector = CodeInspector(self.reter)

        # Build kwargs from params
        kwargs = {
            k: v for k, v in params.items()
            if k not in ("action",) and v is not None
        }

        # Execute the action
        result = inspector.execute_action(action, **kwargs)

        return result

    def _handle_diagram(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate diagram.

        Params:
            diagram_type: Type of diagram (class_hierarchy, class_diagram, sequence, etc.)
            target: Target entity/module
            classes: List of class names for class diagrams
            format: Output format (mermaid, markdown, json)
            include_methods: Include methods in class diagrams
            include_attributes: Include attributes
            max_depth: Maximum depth for hierarchies
            show_external: Show external dependencies

        Returns:
            Dictionary with diagram content
        """
        diagram_type = params.get("diagram_type", "")

        if not diagram_type:
            raise ValueError("Diagram type is required")

        # Import diagram generator
        from ...services.diagram_generator import DiagramGenerator

        generator = DiagramGenerator(self.reter)

        # Build kwargs from params
        kwargs = {
            k: v for k, v in params.items()
            if k not in ("diagram_type",) and v is not None
        }

        # Generate the diagram
        result = generator.generate(diagram_type, **kwargs)

        return result

    def _handle_recommender(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run recommender/detector.

        Params:
            recommender_type: Type (refactoring, testing, etc.)
            detector_name: Specific detector to run
            categories: Filter by category
            severities: Filter by severity
            detector_type: "all", "improving", or "patterns"
            create_tasks: Auto-create tasks from findings

        Returns:
            Detector results and recommendations
        """
        recommender_type = params.get("recommender_type")
        detector_name = params.get("detector_name")

        # Import recommender
        from ...services.recommender import Recommender

        recommender = Recommender(self.reter)

        # Build kwargs from params
        kwargs = {
            k: v for k, v in params.items()
            if k not in ("recommender_type", "detector_name") and v is not None
        }

        # Run recommender
        if detector_name:
            result = recommender.run_detector(recommender_type, detector_name, **kwargs)
        elif recommender_type:
            result = recommender.run_all(recommender_type, **kwargs)
        else:
            result = recommender.list_available()

        return result


__all__ = ["InspectionHandler"]
