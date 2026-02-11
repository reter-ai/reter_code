"""
RETER Code Inspection Handler.

Handles code inspection, diagram generation, and recommender operations
by delegating to CADSL scripts in the tools directory.

::: This is-in-layer Handler-Layer.
::: This is-in-component Query-Handlers.
::: This depends-on reter_code.cadsl.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseHandler
from ..protocol import METHOD_CODE_INSPECTION, METHOD_DIAGRAM, METHOD_RECOMMENDER

# CADSL tools directory (relative to package root)
_CADSL_TOOLS_DIR = Path(__file__).parent.parent.parent / "cadsl" / "tools"

# Recommender categories that map to CADSL tool directories
_RECOMMENDER_CATEGORIES = [
    "smells", "refactoring", "testing", "security",
    "dependencies", "exceptions", "inheritance",
    "patterns", "rag", "good", "file_search",
]


class InspectionHandler(BaseHandler):
    """Handler for code inspection operations.

    Delegates to CADSL scripts in the tools directory.

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

    def _ensure_synced(self) -> None:
        """Ensure default instance is synced with file changes before query."""
        default_manager = None
        if hasattr(self.instance_manager, 'get_default_instance_manager'):
            default_manager = self.instance_manager.get_default_instance_manager()

        if default_manager and default_manager._dirty:
            rebuilt = default_manager.ensure_default_instance_synced(self.reter)
            if rebuilt is not None:
                self.context.reter = rebuilt

    def _execute_cadsl_file(self, cadsl_path: Path, script_params: Dict[str, Any],
                            timeout_ms: int = 300000) -> Dict[str, Any]:
        """Execute a CADSL file and return results.

        This replicates the core logic from QueryHandler._handle_execute_cadsl
        but operates on a resolved file path.
        """
        from ...cadsl.parser import parse_cadsl
        from ...cadsl.transformer import CADSLTransformer
        from ...cadsl.loader import build_pipeline_factory
        from ...dsl.core import Context as PipelineContext
        from ...dsl.catpy import Err, Ok

        self._ensure_synced()

        start_time = time.time()
        source_file = str(cadsl_path)

        try:
            with open(cadsl_path, 'r', encoding='utf-8') as f:
                cadsl_content = f.read()

            # Parse CADSL
            parse_result = parse_cadsl(cadsl_content)
            if not parse_result.success:
                return {
                    "success": False,
                    "error": f"Parse error: {parse_result.errors}",
                    "source_file": source_file,
                }

            # Transform to tool spec
            transformer = CADSLTransformer()
            tool_specs = transformer.transform(parse_result.tree)

            if not tool_specs:
                return {
                    "success": False,
                    "error": "No tool spec generated from CADSL",
                    "source_file": source_file,
                }

            tool_spec = tool_specs[0]

            # Build pipeline context
            rag_manager = self.rag_manager if hasattr(self, 'rag_manager') else None

            project_root = None
            default_manager = None
            if hasattr(self.instance_manager, 'get_default_instance_manager'):
                default_manager = self.instance_manager.get_default_instance_manager()
                if default_manager and default_manager.project_root:
                    project_root = str(default_manager.project_root)

            # Start with default param values from CADSL file
            pipeline_params = {
                "rag_manager": rag_manager,
                "timeout_ms": timeout_ms,
                "project_root": project_root,
                "view_push": self.push_view,
            }
            for param_spec in tool_spec.params:
                if param_spec.default is not None:
                    pipeline_params[param_spec.name] = param_spec.default

            # Override with user-provided params
            if script_params:
                pipeline_params.update(script_params)

            pipeline_ctx = PipelineContext(reter=self.reter, params=pipeline_params)

            # Build and execute pipeline
            pipeline_factory = build_pipeline_factory(tool_spec)
            pipeline = pipeline_factory(pipeline_ctx)
            result = pipeline.execute(pipeline_ctx)

            execution_time = (time.time() - start_time) * 1000

            if isinstance(result, Err):
                return {
                    "success": False,
                    "error": str(result.value),
                    "tool_name": tool_spec.name,
                    "tool_type": tool_spec.tool_type,
                    "source_file": source_file,
                    "execution_time_ms": execution_time,
                }

            if isinstance(result, Ok):
                result = result.value

            # Format result
            if isinstance(result, dict):
                results = result.get("results", [result])
                count = result.get("count", len(results) if isinstance(results, list) else 1)
            elif isinstance(result, list):
                results = result
                count = len(result)
            else:
                results = [result] if result is not None else []
                count = len(results)

            return {
                "success": True,
                "results": results,
                "count": count,
                "tool_name": tool_spec.name,
                "tool_type": tool_spec.tool_type,
                "source_file": source_file,
                "execution_time_ms": execution_time,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source_file": source_file,
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

    def _handle_code_inspection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code inspection action by running corresponding CADSL script.

        Maps action name to cadsl/tools/inspection/{action}.cadsl
        """
        action = params.get("action", "")
        if not action:
            raise ValueError("Action is required")

        cadsl_path = _CADSL_TOOLS_DIR / "inspection" / f"{action}.cadsl"
        if not cadsl_path.exists():
            available = [f.stem for f in (_CADSL_TOOLS_DIR / "inspection").glob("*.cadsl")]
            return {
                "success": False,
                "error": f"Unknown inspection action: '{action}'. Available: {sorted(available)}",
                "action": action,
            }

        # Forward all params except 'action' as CADSL params
        script_params = {k: v for k, v in params.items() if k != "action" and v is not None}

        return self._execute_cadsl_file(cadsl_path, script_params)

    def _handle_diagram(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate diagram by running corresponding CADSL script.

        Maps diagram_type to cadsl/tools/diagrams/{diagram_type}.cadsl
        """
        diagram_type = params.get("diagram_type", "")
        if not diagram_type:
            raise ValueError("Diagram type is required")

        cadsl_path = _CADSL_TOOLS_DIR / "diagrams" / f"{diagram_type}.cadsl"
        if not cadsl_path.exists():
            available = [f.stem for f in (_CADSL_TOOLS_DIR / "diagrams").glob("*.cadsl")]
            return {
                "success": False,
                "error": f"Unknown diagram type: '{diagram_type}'. Available: {sorted(available)}",
                "diagram_type": diagram_type,
            }

        # Forward all params except 'diagram_type' as CADSL params
        script_params = {k: v for k, v in params.items() if k != "diagram_type" and v is not None}

        return self._execute_cadsl_file(cadsl_path, script_params)

    def _handle_recommender(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run recommender/detector by executing corresponding CADSL script.

        - No args: list all available detectors across categories
        - recommender_type only: list detectors in that category
        - recommender_type + detector_name: run that specific detector
        - detector_name only: search all categories for it
        """
        recommender_type = params.get("recommender_type")
        detector_name = params.get("detector_name")

        # No detector specified — list available
        if not detector_name:
            return self._list_recommender_tools(recommender_type)

        # Find the CADSL file
        cadsl_path = self._find_detector_cadsl(recommender_type, detector_name)
        if cadsl_path is None:
            return {
                "success": False,
                "error": f"Detector '{detector_name}' not found"
                         + (f" in category '{recommender_type}'" if recommender_type else ""),
            }

        # Forward params except routing keys
        skip_keys = {"recommender_type", "detector_name", "session_instance"}
        script_params = {k: v for k, v in params.items() if k not in skip_keys and v is not None}

        return self._execute_cadsl_file(cadsl_path, script_params)

    def _find_detector_cadsl(self, category: Optional[str], detector_name: str) -> Optional[Path]:
        """Find the CADSL file for a detector, searching by category or all directories."""
        if category:
            # Direct lookup in specified category
            cadsl_path = _CADSL_TOOLS_DIR / category / f"{detector_name}.cadsl"
            if cadsl_path.exists():
                return cadsl_path
            return None

        # Search all categories
        for cat in _RECOMMENDER_CATEGORIES:
            cadsl_path = _CADSL_TOOLS_DIR / cat / f"{detector_name}.cadsl"
            if cadsl_path.exists():
                return cadsl_path
        return None

    def _list_recommender_tools(self, category: Optional[str] = None) -> Dict[str, Any]:
        """List available recommender/detector tools."""
        result = {}

        categories = [category] if category else _RECOMMENDER_CATEGORIES

        for cat in categories:
            cat_dir = _CADSL_TOOLS_DIR / cat
            if cat_dir.exists():
                tools = sorted(f.stem for f in cat_dir.glob("*.cadsl"))
                if tools:
                    result[cat] = tools

        if not result:
            return {
                "success": False,
                "error": f"No tools found"
                         + (f" in category '{category}'" if category else ""),
            }

        total = sum(len(v) for v in result.values())
        return {
            "success": True,
            "categories": result,
            "total_tools": total,
        }


__all__ = ["InspectionHandler"]
