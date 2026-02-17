"""
RETER File Scan Handler.

Handles file scanning operations over RETER-tracked sources.

::: This is-in-layer Handler-Layer.
::: This is-in-component Query-Handlers.
::: This depends-on reter_code.dsl.core.
"""

from typing import Any, Dict, List

from . import BaseHandler
from ..protocol import METHOD_FILE_SCAN, FILE_SCAN_ERROR


class FileScanHandler(BaseHandler):
    """Handler for file scan operations.

    ::: This is-in-layer Service-Layer.
    ::: This is a handler.
    ::: This is stateful.
    """

    def _register_methods(self) -> None:
        """Register file scan method handlers."""
        self._methods = {
            METHOD_FILE_SCAN: self._handle_file_scan,
        }

    def can_handle(self, method: str) -> bool:
        """Check if this handler can process the method."""
        return method in self._methods

    def _handle_file_scan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scan RETER-tracked files with patterns.

        Params:
            glob: Glob pattern for file matching (e.g., "*.py", "src/**/*.ts")
            contains: Content pattern to search for
            exclude: Glob pattern to exclude files
            case_sensitive: Whether content search is case-sensitive
            include_matches: Include matching lines in results
            context_lines: Lines of context around matches
            limit: Maximum files to return

        Returns:
            Dictionary with matching files and statistics
        """
        glob_pattern = params.get("glob", "*")
        contains = params.get("contains")
        exclude = params.get("exclude")
        case_sensitive = params.get("case_sensitive", False)
        include_matches = params.get("include_matches", True)
        context_lines = params.get("context_lines", 0)
        limit = params.get("limit", 100)

        # Import FileScanSource from DSL
        from ...dsl.core import FileScanSource

        # Create and execute file scan
        source = FileScanSource(
            glob=glob_pattern,
            exclude=exclude,
            contains=contains,
            case_sensitive=case_sensitive,
            include_matches=include_matches,
            context_lines=context_lines,
            include_stats=True
        )

        # Execute the scan — returns Result[T, PipelineError]
        from ...dsl.core import Context
        from ...dsl.catpy import Ok, Err
        ctx = Context(reter=self.reter)
        result = source.execute(ctx)

        # Unwrap the Result monad
        if isinstance(result, Err):
            raise RuntimeError(f"File scan failed: {result.error}")

        data = result.value

        # execute returns Ok(list_of_dicts) when matches found,
        # or Ok({"files": [], "count": 0, "debug": ...}) when empty
        if isinstance(data, list):
            files = data[:limit]
            return {
                "success": True,
                "files": files,
                "count": len(files),
                "total_matches": sum(f.get("match_count", 0) for f in files),
            }
        else:
            # Dict format with files key
            files = data.get("files", [])[:limit]
            return {
                "success": True,
                "files": files,
                "count": len(files),
                "total_matches": data.get("total_matches", 0),
                "stats": data.get("stats", {}),
                "debug": data.get("debug"),
            }


__all__ = ["FileScanHandler"]
