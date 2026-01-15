"""
CADSL Tools Bridge - Provides Python-module-like interface to CADSL tools.

This module loads all .cadsl tool definitions and exposes them as callable
functions that match the interface expected by the registrars.

Usage:
    from codeine.cadsl.tools_bridge import inspection, smells, diagrams

    result = inspection.list_modules(ctx)
    result = smells.god_class(ctx)
    result = diagrams.class_diagram(ctx)
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import logging

from .loader import (
    load_tools_directory,
    build_pipeline_factory,
    LoadResult,
)
from .parser import parse_cadsl_file
from .transformer import CADSLTransformer


logger = logging.getLogger(__name__)


# ============================================================
# REGISTRY-COMPATIBLE TOOL SPEC
# ============================================================

@dataclass
class RegistryToolSpec:
    """
    Adapter class to make CADSL ToolSpec compatible with Registry.

    The Registry expects:
    - .name
    - .type (with .value attribute or string)
    - .meta (dict with category, severity, etc.)
    - .description
    """
    name: str
    type: str  # "query", "detector", "diagram"
    description: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    params: List[Any] = field(default_factory=list)
    executor: Optional[Callable] = None


# ============================================================
# CADSL TOOLS DIRECTORY
# ============================================================

CADSL_TOOLS_DIR = Path(__file__).parent / "tools"


# ============================================================
# TOOL MODULE WRAPPER
# ============================================================

class CADSLToolModule:
    """
    Wraps a directory of CADSL tools as a Python module-like object.

    This allows code like:
        inspection.list_modules(ctx)
        smells.god_class(ctx)

    Tools are loaded fresh from disk on each execution (no caching),
    so changes to .cadsl files take effect immediately.
    """

    def __init__(self, name: str, tools_path: Path):
        self.name = name
        self.tools_path = tools_path
        self._tool_files: Dict[str, Path] = {}  # tool_name -> .cadsl file path
        self._discovered = False

    def _discover_tools(self):
        """Discover available tools by scanning .cadsl files and register with Registry."""
        if self._discovered:
            return

        if not self.tools_path.exists():
            logger.warning(f"CADSL tools directory not found: {self.tools_path}")
            self._discovered = True
            return

        from ..dsl.registry import Registry

        # Parse each file to get metadata for Registry registration
        for cadsl_file in self.tools_path.glob("*.cadsl"):
            tool_name = cadsl_file.stem
            self._tool_files[tool_name] = cadsl_file

            # Parse to get metadata for Registry (lightweight - just for discovery)
            try:
                result = parse_cadsl_file(cadsl_file)
                if not result.success:
                    continue

                transformer = CADSLTransformer()
                specs = transformer.transform(result.tree)

                for spec in specs:
                    # Update tool_files with actual spec name (may differ from filename)
                    self._tool_files[spec.name] = cadsl_file

                    # Register with Registry for recommender discovery
                    registry_spec = RegistryToolSpec(
                        name=spec.name,
                        type=spec.tool_type,
                        description=spec.description or "",
                        meta=spec.metadata or {},
                        params=spec.params,
                        executor=self._make_executor(spec.name),
                    )
                    Registry.register(registry_spec)

            except Exception as e:
                logger.warning(f"Error discovering {cadsl_file}: {e}")

        self._discovered = True
        logger.debug(f"Discovered {len(self._tool_files)} tools in {self.name}")

    def _parse_and_execute(self, tool_name: str, ctx) -> Dict[str, Any]:
        """Parse a .cadsl file fresh from disk and execute it."""
        from codeine.dsl.core import Context

        cadsl_file = self._tool_files.get(tool_name)
        if not cadsl_file or not cadsl_file.exists():
            return {"success": False, "error": f"Tool file not found: {tool_name}"}

        try:
            # Parse fresh from disk
            result = parse_cadsl_file(cadsl_file)
            if not result.success:
                return {"success": False, "error": f"Parse error: {result.errors}"}

            # Transform to executable spec
            transformer = CADSLTransformer()
            specs = transformer.transform(result.tree)

            # Find the matching tool spec
            spec = None
            for s in specs:
                if s.name == tool_name:
                    spec = s
                    break

            if not spec:
                # Try first spec if name doesn't match (single-tool files)
                if specs:
                    spec = specs[0]
                else:
                    return {"success": False, "error": f"No tool spec found in {cadsl_file}"}

            # Build default params
            param_defaults = {}
            for param in spec.params:
                if param.default is not None:
                    param_defaults[param.name] = param.default

            # Merge with context params
            ctx_params_filtered = {k: v for k, v in ctx.params.items() if v is not None}
            merged_params = {**param_defaults, **ctx_params_filtered}

            ctx_with_defaults = Context(
                reter=ctx.reter,
                params=merged_params,
                instance_name=ctx.instance_name,
            )

            # Build and execute pipeline
            factory = build_pipeline_factory(spec)
            pipeline = factory(ctx_with_defaults)
            result = pipeline.execute(ctx_with_defaults)

            # Unwrap PipelineResult
            if hasattr(result, 'unwrap'):
                try:
                    return result.unwrap()
                except Exception as e:
                    return {"success": False, "error": str(e)}

            return result

        except Exception as e:
            logger.exception(f"Error executing CADSL tool {tool_name}")
            return {"success": False, "error": str(e)}

    def _make_executor(self, tool_name: str) -> Callable:
        """Create an executor function that parses fresh from disk on each call."""
        module = self  # Capture reference for closure

        def executor(ctx):
            """Execute the CADSL tool (parsed fresh from disk)."""
            from codeine.dsl.core import Context

            if not isinstance(ctx, Context):
                return {
                    "success": False,
                    "error": f"Expected Context, got {type(ctx).__name__}"
                }

            return module._parse_and_execute(tool_name, ctx)

        executor.__name__ = tool_name
        executor.__doc__ = f"CADSL tool: {tool_name} (loaded from {self.name}/{tool_name}.cadsl)"
        return executor

    def __getattr__(self, name: str) -> Callable:
        """Get a tool by name."""
        self._discover_tools()

        if name in self._tool_files:
            return self._make_executor(name)

        raise AttributeError(f"CADSL tool module '{self.name}' has no tool '{name}'")

    def __dir__(self) -> List[str]:
        """List available tools."""
        self._discover_tools()
        return list(self._tool_files.keys())

    @property
    def __all__(self) -> List[str]:
        """Get list of all tool names."""
        self._discover_tools()
        return list(self._tool_files.keys())

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name, returning None if not found."""
        self._discover_tools()
        if name in self._tool_files:
            return self._make_executor(name)
        return None

    def get_tool_spec(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool specification metadata (parses file fresh)."""
        self._discover_tools()
        cadsl_file = self._tool_files.get(name)
        if not cadsl_file:
            return None

        try:
            result = parse_cadsl_file(cadsl_file)
            if not result.success:
                return None

            transformer = CADSLTransformer()
            specs = transformer.transform(result.tree)

            for spec in specs:
                if spec.name == name:
                    return {
                        "name": spec.name,
                        "type": spec.tool_type,
                        "docstring": spec.description or "",
                        "params": spec.params,
                        "source_file": str(cadsl_file),
                        "metadata": spec.metadata,
                    }

            # Return first spec if name doesn't match
            if specs:
                spec = specs[0]
                return {
                    "name": spec.name,
                    "type": spec.tool_type,
                    "docstring": spec.description or "",
                    "params": spec.params,
                    "source_file": str(cadsl_file),
                    "metadata": spec.metadata,
                }
        except Exception as e:
            logger.warning(f"Error getting spec for {name}: {e}")

        return None

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        self._discover_tools()
        return list(self._tool_files.keys())

    def rescan(self) -> Dict[str, Any]:
        """
        Rescan the tools directory for new/removed .cadsl files.

        Returns:
            Dict with scan results
        """
        old_tools = set(self._tool_files.keys())
        self._discovered = False
        self._tool_files.clear()
        self._discover_tools()
        new_tools = set(self._tool_files.keys())

        return {
            "success": True,
            "module": self.name,
            "count": len(new_tools),
            "added": list(new_tools - old_tools),
            "removed": list(old_tools - new_tools),
        }


# ============================================================
# PRE-DEFINED TOOL MODULES
# ============================================================

# These match the structure of the old dsl.tools package
inspection = CADSLToolModule("inspection", CADSL_TOOLS_DIR / "inspection")
smells = CADSLToolModule("smells", CADSL_TOOLS_DIR / "smells")
refactoring = CADSLToolModule("refactoring", CADSL_TOOLS_DIR / "refactoring")
inheritance = CADSLToolModule("inheritance", CADSL_TOOLS_DIR / "inheritance")
exceptions = CADSLToolModule("exceptions", CADSL_TOOLS_DIR / "exceptions")
patterns = CADSLToolModule("patterns", CADSL_TOOLS_DIR / "patterns")
testing = CADSLToolModule("testing", CADSL_TOOLS_DIR / "testing")
dependencies = CADSLToolModule("dependencies", CADSL_TOOLS_DIR / "dependencies")
rag = CADSLToolModule("rag", CADSL_TOOLS_DIR / "rag")
diagrams = CADSLToolModule("diagrams", CADSL_TOOLS_DIR / "diagrams")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_all_tools() -> Dict[str, Callable]:
    """Get all tools from all modules."""
    all_tools = {}
    for module in [inspection, smells, refactoring, inheritance, exceptions,
                   patterns, testing, dependencies, rag, diagrams]:
        for name in module.list_tools():
            all_tools[name] = module.get_tool(name)
    return all_tools


def get_tool(name: str) -> Optional[Callable]:
    """Get a tool by name from any module."""
    for module in [inspection, smells, refactoring, inheritance, exceptions,
                   patterns, testing, dependencies, rag, diagrams]:
        tool = module.get_tool(name)
        if tool:
            return tool
    return None


def execute_tool(name: str, ctx) -> Dict[str, Any]:
    """Execute a tool by name (parses fresh from disk)."""
    tool = get_tool(name)
    if tool is None:
        return {"success": False, "error": f"Tool not found: {name}"}
    return tool(ctx)


def rescan_all() -> Dict[str, Any]:
    """
    Rescan all tool directories for new/removed .cadsl files.

    Note: This is only needed if you add/remove .cadsl files.
    Edits to existing files are picked up automatically on each execution.

    Returns:
        Dict with scan results for each module
    """
    results = {}
    total = 0

    for module in [inspection, smells, refactoring, inheritance, exceptions,
                   patterns, testing, dependencies, rag, diagrams]:
        result = module.rescan()
        results[module.name] = result
        total += result["count"]

    return {
        "success": True,
        "modules": results,
        "total_tools": total,
    }


__all__ = [
    # Tool modules
    "inspection",
    "smells",
    "refactoring",
    "inheritance",
    "exceptions",
    "patterns",
    "testing",
    "dependencies",
    "rag",
    "diagrams",
    # Helper functions
    "get_all_tools",
    "get_tool",
    "execute_tool",
    "rescan_all",
    # Classes
    "CADSLToolModule",
]
