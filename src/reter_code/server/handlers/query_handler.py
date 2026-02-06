"""
RETER Query Handler.

Handles REQL, DL, and CADSL query operations via ZeroMQ.

::: This is-in-layer Handler-Layer.
::: This is-in-component Query-Handlers.
::: This depends-on reter_code.reter_wrapper.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseHandler, HandlerContext
from ..protocol import (
    METHOD_REQL,
    METHOD_DL,
    METHOD_PATTERN,
    METHOD_NATURAL_LANGUAGE,
    METHOD_EXECUTE_CADSL,
    METHOD_GENERATE_CADSL,
    METHOD_SIMILAR_CADSL_TOOLS,
    QUERY_ERROR,
    ReterError,
)


class QueryHandler(BaseHandler):
    """Handler for query operations (REQL, DL, CADSL).

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is stateful.
    """

    def _register_methods(self) -> None:
        """Register query method handlers."""
        self._methods = {
            METHOD_REQL: self._handle_reql,
            METHOD_DL: self._handle_dl,
            METHOD_PATTERN: self._handle_pattern,
            METHOD_EXECUTE_CADSL: self._handle_execute_cadsl,
            METHOD_GENERATE_CADSL: self._handle_generate_cadsl,
            METHOD_SIMILAR_CADSL_TOOLS: self._handle_similar_cadsl_tools,
        }

    def can_handle(self, method: str) -> bool:
        """Check if this handler can process the method."""
        return method in self._methods

    def _ensure_synced(self) -> None:
        """Ensure default instance is synced with file changes before query.

        This checks the dirty flag from the file watcher and syncs if needed.
        Only syncs the "default" instance that auto-tracks project files.
        """
        # Get the DefaultInstanceManager which has the dirty flag and sync logic
        default_manager = None
        if hasattr(self.instance_manager, 'get_default_instance_manager'):
            default_manager = self.instance_manager.get_default_instance_manager()

        if default_manager and default_manager._dirty:
            # File changes detected - sync before query
            rebuilt = default_manager.ensure_default_instance_synced(self.reter)
            if rebuilt is not None:
                # Instance was rebuilt - update the context's reter reference
                self.context.reter = rebuilt

    def _handle_reql(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute REQL query.

        Params:
            query: REQL query string
            timeout_ms: Optional timeout in milliseconds

        Returns:
            Dictionary with columns, rows, and metadata
        """
        query = params.get("query", "")
        timeout_ms = params.get("timeout_ms")

        if not query:
            raise ValueError("Query string is required")

        # Sync files before query if changes detected (file watcher sets dirty flag)
        self._ensure_synced()

        # Execute query - returns PyArrow table
        result = self.reter.reql(query, timeout_ms=timeout_ms)

        # Convert PyArrow table to serializable format
        return self._serialize_table(result)

    def _handle_dl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DL (Description Logic) query.

        Params:
            query: DL query string

        Returns:
            Dictionary with instances matching the query
        """
        query = params.get("query", "")

        if not query:
            raise ValueError("Query string is required")

        # Sync files before query if changes detected
        self._ensure_synced()

        # Execute DL query
        result = self.reter.reasoner.dl(query)

        return {
            "instances": result,
            "count": len(result)
        }

    def _handle_pattern(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pattern query.

        Params:
            query: Pattern query string

        Returns:
            Dictionary with matching results
        """
        query = params.get("query", "")

        if not query:
            raise ValueError("Query string is required")

        # Sync files before query if changes detected
        self._ensure_synced()

        # Execute pattern query
        result = self.reter.reasoner.pattern(query)

        return self._serialize_table(result)

    def _handle_execute_cadsl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CADSL script.

        Params:
            script: CADSL script string or file path
            params: Optional parameters for the script
            timeout_ms: Optional timeout in milliseconds

        Returns:
            Dictionary with execution results
        """
        from ...cadsl.parser import parse_cadsl
        from ...cadsl.transformer import CADSLTransformer
        from ...cadsl.loader import build_pipeline_factory
        from ...dsl.core import Context as PipelineContext
        from ...dsl.catpy import Err, Ok

        # Sync files before query if changes detected
        self._ensure_synced()

        script = params.get("script", "")
        script_params = params.get("params", {})
        timeout_ms = params.get("timeout_ms", 300000)

        if not script:
            raise ValueError("Script is required")

        start_time = time.time()

        # Determine if script is a file path or inline CADSL
        cadsl_content = script
        source_file = None

        # Check if it looks like a file path (not inline CADSL)
        script_stripped = script.strip()

        # Inline CADSL detection: check for CADSL keywords or comments at start
        # Also check for newlines - file paths don't have newlines
        is_inline_cadsl = (
            script_stripped.startswith('query ') or
            script_stripped.startswith('detector ') or
            script_stripped.startswith('diagram ') or
            script_stripped.startswith('TOOL ') or
            script_stripped.startswith('//') or
            script_stripped.startswith('#') or  # CADSL comment
            '\n' in script_stripped  # Multi-line content is inline CADSL
        )

        # File path detection: must look like a path and not be inline CADSL
        # On Windows, check for drive letter or .cadsl extension
        # Avoid false positives from backslashes in REQL regex patterns
        looks_like_path = (
            script_stripped.endswith('.cadsl') or
            (len(script_stripped) > 2 and script_stripped[1] == ':') or  # Windows drive letter
            script_stripped.startswith('/')  # Unix absolute path
        )

        if not is_inline_cadsl and looks_like_path:
            path = Path(script_stripped)
            if path.exists() and path.is_file():
                source_file = str(path)
                with open(path, 'r', encoding='utf-8') as f:
                    cadsl_content = f.read()
            elif not path.exists():
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "error": f"CADSL file not found: {script}",
                    "source_file": None
                }

        try:
            # Parse CADSL
            parse_result = parse_cadsl(cadsl_content)
            if not parse_result.success:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "cadsl_script": cadsl_content[:500] + "..." if len(cadsl_content) > 500 else cadsl_content,
                    "source_file": source_file,
                    "error": f"Parse error: {parse_result.errors}"
                }

            # Transform to tool spec
            transformer = CADSLTransformer()
            tool_specs = transformer.transform(parse_result.tree)

            if not tool_specs:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "cadsl_script": cadsl_content[:500] + "..." if len(cadsl_content) > 500 else cadsl_content,
                    "source_file": source_file,
                    "error": "No tool spec generated from CADSL"
                }

            tool_spec = tool_specs[0]

            # Build pipeline context
            rag_manager = self.rag_manager if hasattr(self, 'rag_manager') else None

            # Get project root for content extraction
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

            # Check for Result monad errors (Err type)
            if isinstance(result, Err):
                error = result.value
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "tool_name": tool_spec.name,
                    "tool_type": tool_spec.tool_type,
                    "source_file": source_file,
                    "execution_time_ms": execution_time,
                    "error": str(error)
                }

            # Unwrap Ok result if needed
            if isinstance(result, Ok):
                result = result.value

            # Format result based on type
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
                "execution_time_ms": execution_time
            }

        except Exception as e:
            return {
                "success": False,
                "results": [],
                "count": 0,
                "source_file": source_file,
                "execution_time_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }

    def _handle_generate_cadsl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CADSL from natural language.

        Params:
            question: Natural language question
            max_retries: Maximum retry attempts
            timeout: Timeout in seconds

        Returns:
            Dictionary with generated CADSL query
        """
        question = params.get("question", "")
        max_retries = params.get("max_retries", 5)
        timeout = params.get("timeout", 300)

        if not question:
            raise ValueError("Question is required")

        # Use the agent SDK client for CADSL generation
        try:
            from ...services.agent_sdk_client import generate_cadsl_query
            import asyncio

            # Run the async function
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                generate_cadsl_query(
                    question=question,
                    reter_client=None,  # Use in-process RETER
                    max_retries=max_retries,
                    timeout=timeout
                )
            )
            return result

        except ImportError:
            return {
                "success": False,
                "error": "Agent SDK not available for CADSL generation"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _serialize_table(self, table) -> Dict[str, Any]:
        """Convert PyArrow table to serializable dictionary.

        Args:
            table: PyArrow Table or similar result

        Returns:
            Dictionary with columns, rows, and metadata
        """
        if table is None:
            return {"columns": [], "rows": [], "count": 0}

        try:
            # PyArrow table
            if hasattr(table, 'to_pydict'):
                py_dict = table.to_pydict()
                columns = list(py_dict.keys())
                num_rows = len(next(iter(py_dict.values()), []))

                # Convert to list of row dicts
                rows = []
                for i in range(num_rows):
                    row = {col: py_dict[col][i] for col in columns}
                    rows.append(row)

                return {
                    "columns": columns,
                    "rows": rows,
                    "count": num_rows
                }
            # Already a list/dict
            elif isinstance(table, list):
                return {
                    "columns": [],
                    "rows": table,
                    "count": len(table)
                }
            elif isinstance(table, dict):
                return table
            else:
                # Fallback - convert to string
                return {
                    "columns": [],
                    "rows": [str(table)],
                    "count": 1
                }
        except Exception as e:
            return {
                "columns": [],
                "rows": [],
                "count": 0,
                "error": str(e)
            }

    def _handle_similar_cadsl_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find CADSL tools similar to a question.

        Params:
            question: Natural language question
            max_results: Maximum number of similar tools to return (default: 5)

        Returns:
            Dictionary with similar_tools list
        """
        question = params.get("question", "")
        max_results = params.get("max_results", 5)

        if not question:
            raise ValueError("Question is required")

        try:
            from ...services.hybrid_query_engine import find_similar_cadsl_tools

            similar_tools = find_similar_cadsl_tools(question, max_results=max_results)

            return {
                "success": True,
                "similar_tools": [
                    {
                        "name": t.name,
                        "score": t.score,
                        "category": t.category,
                        "description": t.description,
                        "content": t.content,
                    }
                    for t in similar_tools
                ],
                "count": len(similar_tools)
            }

        except Exception as e:
            return {
                "success": False,
                "similar_tools": [],
                "count": 0,
                "error": str(e)
            }


__all__ = ["QueryHandler"]
