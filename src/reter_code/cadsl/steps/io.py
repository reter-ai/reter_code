"""
CADSL I/O Steps.

Contains step classes for file I/O, content fetching, viewing, and Python execution:
- FetchContentStep: Fetch source code content from files using line numbers
- ViewStep: Push rendered content to RETER View (browser-based viewer)
- WriteFileStep: Write pipeline results to a file
- PythonStep: Execute inline Python code as a pipeline step
"""

from typing import Any, Dict, List, Optional


class FetchContentStep:
    """
    Fetches source code content from files using line numbers.

    Reads the actual source code between start_line and end_line from the file
    and adds it as a new field to each row. This is useful for RAG-based
    code analysis where you need the actual code body, not just metadata.

    Syntax: fetch_content { file: file_field, start_line: line, end_line: end_line, output: body }

    Parameters:
    - file: Field name containing the file path (default: "file")
    - start_line: Field name containing the start line number (default: "line")
    - end_line: Field name containing the end line number (optional)
    - output: Field name for the extracted content (default: "body")
    - max_lines: Maximum lines to extract (default: 50)

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(
        self,
        file_field: str = "file",
        start_line_field: str = "line",
        end_line_field: str = None,
        output_field: str = "body",
        max_lines: int = 50
    ):
        self.file_field = file_field
        self.start_line_field = start_line_field
        self.end_line_field = end_line_field
        self.output_field = output_field
        self.max_lines = max_lines

    def execute(self, data, ctx=None):
        """Execute content fetching for each row."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok([])

            # Get content extractor
            content_extractor = None
            project_root = None

            if ctx and hasattr(ctx, 'get'):
                content_extractor = ctx.get("content_extractor")
                project_root = ctx.get("project_root")

            # Also check params for project_root (passed from query handler)
            if project_root is None and ctx and hasattr(ctx, 'params'):
                project_root = ctx.params.get("project_root")

            # Priority order for finding project root:
            # 1. RETER_PROJECT_ROOT environment variable (set by MCP server)
            # 2. Context-provided content_extractor or project_root
            # 3. DefaultInstanceManager (if available)
            # 4. CWD as fallback
            import os
            from pathlib import Path

            if content_extractor is None:
                # First try environment variable - this is set by the MCP server
                env_root = os.environ.get("RETER_PROJECT_ROOT")
                if env_root:
                    project_root = Path(env_root)
                    logger.debug(f"FetchContentStep using RETER_PROJECT_ROOT: {project_root}")

            if content_extractor is None and project_root is None:
                # Try DefaultInstanceManager (may not be available in MCP mode)
                try:
                    from reter_code.services.default_instance_manager import DefaultInstanceManager
                    default_mgr = DefaultInstanceManager.get_instance()
                    if default_mgr:
                        rag_manager = default_mgr.get_rag_manager()
                        if rag_manager and hasattr(rag_manager, '_content_extractor') and rag_manager._content_extractor:
                            content_extractor = rag_manager._content_extractor
                            logger.debug(f"FetchContentStep using RAG manager's content extractor")
                        if content_extractor is None and default_mgr.project_root:
                            project_root = default_mgr.project_root
                except Exception as e:
                    logger.debug(f"DefaultInstanceManager not available: {e}")

            if content_extractor is None:
                # Create a new content extractor
                from reter_code.services.content_extractor import ContentExtractor

                # Use project_root we found, or CWD as fallback
                final_root = Path(project_root) if project_root else Path.cwd()
                logger.debug(f"FetchContentStep creating content extractor with root: {final_root}")
                content_extractor = ContentExtractor(
                    project_root=final_root,
                    max_body_lines=self.max_lines
                )

            result = []
            for row in data:
                new_row = dict(row)

                # Get field values (try with and without ? prefix)
                file_path = row.get(self.file_field) or row.get(f"?{self.file_field}")
                start_line = row.get(self.start_line_field) or row.get(f"?{self.start_line_field}")

                end_line = None
                if self.end_line_field:
                    end_line = row.get(self.end_line_field) or row.get(f"?{self.end_line_field}")

                # Extract content
                body = ""
                if file_path and start_line:
                    try:
                        start_line = int(start_line)
                        if end_line:
                            end_line = int(end_line)
                            # Limit end_line based on max_lines
                            if end_line - start_line + 1 > self.max_lines:
                                end_line = start_line + self.max_lines - 1

                        body = content_extractor.extract_entity_content(
                            file_path=str(file_path),
                            start_line=start_line,
                            end_line=end_line,
                            entity_type="code"
                        )
                        if body is None:
                            body = ""
                    except Exception as e:
                        logger.debug(f"Failed to extract content from {file_path}:{start_line}: {e}")
                        body = ""

                new_row[self.output_field] = body
                result.append(new_row)

            return pipeline_ok(result)

        except Exception as e:
            import traceback
            logger.error(f"Fetch content failed: {e}\n{traceback.format_exc()}")
            return pipeline_err("fetch_content", f"Fetch content failed: {e}", e)


class ViewStep:
    """
    Pushes rendered content to RETER View (browser-based viewer).

    This is a pass-through step: it pushes content to the viewer but always
    returns data unchanged. If view_push is not available (no ViewServer),
    it silently passes through.

    Syntax: view { skip: false, content: diagram, type: mermaid }

    Parameters:
    - skip: Whether to skip pushing (default: False)
    - content_key: Data dict key to push (default: auto-detect)
    - content_type: Content type hint: mermaid, markdown, html (default: auto-detect)

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    # Auto-detection mapping: data key -> content type
    _KEY_TYPE_MAP = {
        "diagram": "mermaid",
        "chart": "mermaid",
        "table": "markdown",
        "markdown": "markdown",
        "html": "html",
    }

    def __init__(self, skip: bool = False, content_key: str = None, content_type: str = None, description: str = None):
        self.skip = skip
        self.content_key = content_key
        self.content_type = content_type
        self.description = description

    def execute(self, data, ctx=None):
        """Execute view push (pass-through)."""
        from reter_code.dsl.core import pipeline_ok
        import logging

        logger = logging.getLogger(__name__)

        if self.skip:
            logger.debug("ViewStep: skipped (skip=True)")
            return pipeline_ok(data)

        # Get view_push callback from context
        view_push = None
        if ctx and hasattr(ctx, 'params'):
            view_push = ctx.params.get("view_push")

        if view_push is None:
            logger.debug("ViewStep: no view_push callback available, passing through")
            return pipeline_ok(data)

        try:
            # Determine content and type
            content = None
            content_type = self.content_type

            if isinstance(data, dict):
                if self.content_key:
                    content = data.get(self.content_key)
                    if content_type is None:
                        content_type = self._KEY_TYPE_MAP.get(self.content_key, "markdown")
                else:
                    # Auto-detect from dict keys
                    for key, ctype in self._KEY_TYPE_MAP.items():
                        if key in data and data[key]:
                            content = data[key]
                            if content_type is None:
                                content_type = ctype
                            break

            elif isinstance(data, str):
                content = data
                if content_type is None:
                    content_type = "markdown"

            if content and content_type:
                if content_type == "mermaid":
                    try:
                        from reter_code.mermaid.validator import validate_mermaid
                        vr = validate_mermaid(content)
                        if not vr.valid:
                            logger.warning("ViewStep: Mermaid validation failed, not pushing: %s",
                                           "; ".join(e.message for e in vr.errors))
                            return pipeline_ok(data)
                    except Exception:
                        pass
                elif content_type == "markdown":
                    try:
                        from reter_code.mermaid.markdown_validator import validate_markdown
                        vr = validate_markdown(content)
                        if not vr.valid:
                            logger.warning("ViewStep: Markdown validation failed, not pushing: %s",
                                           "; ".join(e.message for e in vr.errors))
                            return pipeline_ok(data)
                    except Exception:
                        pass
                view_push(content_type, content, title=self.description)
                logger.debug(f"ViewStep: pushed {content_type} content ({len(str(content))} chars)")
            else:
                logger.debug("ViewStep: no content to push")

        except Exception as e:
            logger.debug(f"ViewStep: push failed (non-fatal): {e}")

        return pipeline_ok(data)


class WriteFileStep:
    """
    Write pipeline results to a file. Pass-through: data flows to next step unchanged.

    Syntax: write_file { path: "output.csv", format: csv }
           write_file { }  -- auto-generates temp file in .reter_code/results/

    When called without a path, generates a random filename under
    <project_root>/.reter_code/results/ (default format: json).
    The output file path is added to the result as _output_file.

    Parameters:
    - path: Output file path (relative to project root). Empty = auto-generate.
    - format: csv, json, or parquet (default: json)
    - encoding: File encoding (default: utf-8)
    - separator: CSV separator (default: ,)
    - indent: JSON indent (default: 2)
    - overwrite: Whether to overwrite existing files (default: true)

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    RESULTS_DIR = ".reter_code/results"
    FORMAT_EXT = {"csv": ".csv", "json": ".json", "parquet": ".parquet"}

    def __init__(self, path: str = "", format: str = "json", encoding: str = "utf-8",
                 separator: str = ",", indent: int = 2, overwrite: bool = True):
        self.path = path
        self.format = format
        self.encoding = encoding
        self.separator = separator
        self.indent = indent
        self.overwrite = overwrite

    def _generate_path(self, project_root: str) -> "Path":
        """Generate a random filename in the results directory."""
        import uuid
        from datetime import datetime
        from pathlib import Path

        results_dir = Path(project_root) / self.RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        ext = self.FORMAT_EXT.get(self.format, ".json")
        return results_dir / f"{ts}_{short_id}{ext}"

    def execute(self, data, ctx=None):
        """Write data to file and pass through."""
        import pandas as pd
        from pathlib import Path
        from reter_code.dsl.core import pipeline_ok, pipeline_err, _get_project_root

        project_root = _get_project_root(ctx)
        auto_generated = not self.path

        if auto_generated:
            file_path = self._generate_path(project_root)
        else:
            file_path = Path(project_root) / self.path

        if file_path.exists() and not self.overwrite:
            return pipeline_err("write_file", f"File exists and overwrite=false: {file_path}")

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Normalize input
            rows = data.to_pylist() if hasattr(data, 'to_pylist') else data
            if isinstance(rows, dict):
                rows = [rows]

            df = pd.DataFrame(rows)

            if self.format == "csv":
                df.to_csv(file_path, index=False, encoding=self.encoding, sep=self.separator)
            elif self.format == "json":
                df.to_json(file_path, orient="records", indent=self.indent, force_ascii=False)
            elif self.format == "parquet":
                df.to_parquet(file_path, index=False)
            else:
                return pipeline_err("write_file", f"Unsupported format: {self.format}")

            if auto_generated:
                # Return data wrapped with the generated file path
                return pipeline_ok({"results": data, "_output_file": str(file_path)})
            return pipeline_ok(data)
        except Exception as e:
            return pipeline_err("write_file", f"Failed to write {self.path or str(file_path)}: {e}", e)


# ============================================================
# PYTHON STEP
# ============================================================

class PythonStep:
    """
    Executes inline Python code as a pipeline step.

    Available in the Python block:
    - rows: Input data from previous step
    - ctx: Execution context with params
    - result: Must be set to the output value

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, code: str):
        self.code = code
        try:
            self.compiled = compile(code, "<cadsl_python>", "exec")
        except SyntaxError as e:
            self.compiled = None
            self.error = str(e)

    def execute(self, data, ctx=None):
        """Execute the Python code."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        if self.compiled is None:
            return pipeline_err("python", f"Python syntax error: {self.error}")

        # Create execution namespace
        namespace = {
            "rows": data,
            "ctx": ctx,
            "result": None,
            # Common imports
            "defaultdict": __import__("collections").defaultdict,
            "Counter": __import__("collections").Counter,
            "re": __import__("re"),
            "json": __import__("json"),
            "math": __import__("math"),
        }

        try:
            exec(self.compiled, namespace)

            if namespace.get("result") is None:
                # If no result set, use rows
                return pipeline_ok(data)

            return pipeline_ok(namespace["result"])
        except Exception as e:
            return pipeline_err("python", f"Python execution error: {e}", e)
