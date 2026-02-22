"""
CADSL Integration Steps.

Contains step classes for external service integration in CADSL pipelines:
- RagEnrichStep: Per-row RAG enrichment with semantic search
- CreateTaskStep: Creates tasks from pipeline data (file-based or inline)
"""

from typing import Any, Dict, List, Optional


class RagEnrichStep:
    """
    Per-row RAG enrichment step - enriches each row with semantic search results.

    Syntax: rag_enrich { query: "template {field}", top_k: 3, threshold: 0.5, mode: "best" }

    Template placeholders like {field} are replaced with row values before search.

    Modes:
    - "best": Adds best match fields directly to row (similarity, similar_entity, similar_file)
    - "all": Adds array of all matches as rag_matches field

    Uses batching for performance optimization.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, query_template, top_k=1, threshold=None, mode="best",
                 batch_size=50, max_rows=1000, entity_types=None):
        self.query_template = query_template
        self.top_k = top_k
        self.threshold = threshold
        self.mode = mode
        self.batch_size = batch_size
        self.max_rows = max_rows
        self.entity_types = entity_types

    def execute(self, data, ctx=None):
        """Execute RAG enrichment with batching."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err
        import logging
        import re

        logger = logging.getLogger(__name__)

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok([])

            # Check row count limit
            if len(data) > self.max_rows:
                logger.warning(
                    f"RAG enrichment: {len(data)} rows exceeds max_rows ({self.max_rows}). "
                    f"Processing first {self.max_rows} rows only."
                )
                data = data[:self.max_rows]

            # Validate template fields against first row
            template_fields = re.findall(r'\{(\w+)\}', self.query_template)
            if data and template_fields:
                missing = [f for f in template_fields if f not in data[0]]
                if missing:
                    # Check for ?-prefixed versions (REQL output)
                    still_missing = []
                    for f in missing:
                        if f"?{f}" not in data[0]:
                            still_missing.append(f)
                    if still_missing:
                        return pipeline_err(
                            "rag_enrich",
                            f"Template field(s) not found in row: {still_missing}. "
                            f"Available fields: {list(data[0].keys())}"
                        )

            # Get RAG manager from context or default instance (same pattern as RAGSearchSource)
            rag_manager = None
            if ctx and hasattr(ctx, 'get'):
                rag_manager = ctx.get("rag_manager")
            if rag_manager is None:
                try:
                    from reter_code.services.default_instance_manager import DefaultInstanceManager
                    default_mgr = DefaultInstanceManager.get_instance()
                    if default_mgr:
                        rag_manager = default_mgr.get_rag_manager()
                except Exception as e:
                    logger.debug(f"Could not get RAG manager: {e}")

            if rag_manager is None:
                return pipeline_err(
                    "rag_enrich",
                    "RAG manager not available. Ensure project is initialized."
                )

            # Process in batches for performance
            result = []
            for batch_start in range(0, len(data), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(data))
                batch = data[batch_start:batch_end]

                # Prepare queries for batch
                queries = []
                for row in batch:
                    query = self._expand_template(row)
                    queries.append(query)

                # Execute batch search
                batch_results = self._batch_search(rag_manager, queries, ctx)

                # Enrich rows with results
                for i, row in enumerate(batch):
                    new_row = dict(row)
                    matches = batch_results[i] if i < len(batch_results) else []

                    # Ensure matches is always a list
                    if matches is None:
                        matches = []

                    # Filter by threshold (RAG results use 'score' field)
                    if self.threshold is not None and matches:
                        matches = [m for m in matches if m.get('score', m.get('similarity', 0)) >= self.threshold]

                    if self.mode == "best":
                        # Add best match fields directly
                        if matches:
                            best = matches[0]
                            # RAG results use 'score', fallback to 'similarity' for compatibility
                            score = best.get('score', best.get('similarity', 0))
                            new_row['similarity'] = score
                            new_row['rag_similarity'] = score  # alias for filter compatibility
                            new_row['similar_entity'] = best.get('name', best.get('entity', ''))
                            new_row['similar_file'] = best.get('file', '')
                            new_row['similar_line'] = best.get('line', 0)
                            new_row['similar_type'] = best.get('entity_type', '')
                        else:
                            new_row['similarity'] = 0
                            new_row['rag_similarity'] = 0
                            new_row['similar_entity'] = None
                            new_row['similar_file'] = None
                            new_row['similar_line'] = None
                            new_row['similar_type'] = None
                    else:  # mode == "all"
                        new_row['rag_matches'] = matches

                    result.append(new_row)

            return pipeline_ok(result)

        except Exception as e:
            import traceback
            logger.error(f"RAG enrichment failed: {e}\n{traceback.format_exc()}")
            return pipeline_err("rag_enrich", f"RAG enrichment failed: {e}", e)

    def _expand_template(self, row):
        """Expand template placeholders with row values."""
        import re

        def replacer(match):
            field = match.group(1)
            # Try exact field name first
            if field in row:
                return str(row[field])
            # Try ?-prefixed version (REQL output)
            if f"?{field}" in row:
                return str(row[f"?{field}"])
            return match.group(0)  # Keep original if not found

        return re.sub(r'\{(\w+)\}', replacer, self.query_template)

    def _batch_search(self, rag_manager, queries, ctx):
        """Execute batch RAG search. Returns list of result lists."""
        results = []

        for query in queries:
            try:
                # Use RAG manager's search method (returns (results, stats))
                if hasattr(rag_manager, 'search'):
                    search_results, stats = rag_manager.search(
                        query=query,
                        top_k=self.top_k,
                        entity_types=self.entity_types
                    )

                    # Check for errors
                    if stats.get("error"):
                        results.append([])
                        continue

                    # Convert RAGSearchResult objects to dicts
                    matches = []
                    for r in search_results:
                        if hasattr(r, 'to_dict'):
                            matches.append(r.to_dict())
                        elif isinstance(r, dict):
                            matches.append(r)
                        elif hasattr(r, '__dict__'):
                            matches.append(vars(r))
                        else:
                            matches.append({'entity': str(r), 'similarity': 0})
                    results.append(matches)
                else:
                    results.append([])
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"RAG search failed for query '{query[:50]}...': {e}")
                results.append([])

        return results


class CreateTaskStep:
    """
    Creates tasks in RETER session from pipeline data.

    Syntax: create_task { name: "Add annotation to {class_name}", category: "annotation", priority: medium }

    Template placeholders like {field} are replaced with row values.

    Parameters:
    - name: Task name template (required)
    - category: Task category (feature, bug, refactor, test, docs, annotation, research)
    - priority: critical, high, medium, low
    - description: Task description template
    - affects: Field name containing file path to mark as affected
    - batch_size: Number of tasks to create per batch
    - dry_run: If true, returns task data without creating tasks
    - filter_predicates: List of predicate names to filter out obvious FPs
      - skip_same_names: Skip if all methods have same name (interface pattern)
      - skip_trivial: Skip clusters with <3 lines per method
      - skip_boilerplate: Skip __init__, __str__, __repr__, etc.
      - skip_single_file: Skip if all in same file
    - metadata_template: Dict of metadata to store with task (supports {field} templates)
    - group_id: ID to group related tasks into a batch
    - source_tool: Tool name that created this task (for filtering/tracking)

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    # Built-in filter predicates
    FILTER_PREDICATES = {
        "skip_same_names",
        "skip_trivial",
        "skip_boilerplate",
        "skip_single_file",
    }

    # Boilerplate method names to skip
    BOILERPLATE_METHODS = {
        "__init__", "__str__", "__repr__", "__eq__", "__hash__",
        "__lt__", "__le__", "__gt__", "__ge__", "__ne__",
        "__len__", "__iter__", "__next__", "__getitem__", "__setitem__",
        "__delitem__", "__contains__", "__call__", "__enter__", "__exit__",
        "toString", "equals", "hashCode", "compareTo", "clone",
        "GetHashCode", "Equals", "ToString", "CompareTo",
    }

    def __init__(self, name_template, category="annotation", priority="medium",
                 description_template=None, prompt_template=None, affects_field=None,
                 batch_size=50, dry_run=False, filter_predicates=None, metadata_template=None,
                 group_id=None, source_tool=None):
        self.name_template = name_template
        self.category = category
        self.priority = priority
        self.description_template = description_template
        self.prompt_template = prompt_template
        self.affects_field = affects_field
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.filter_predicates = filter_predicates or []
        self.metadata_template = metadata_template or {}
        self.group_id = group_id
        self.source_tool = source_tool

    def execute(self, data, ctx=None):
        """Execute task creation. Returns Claude Code-compatible task data."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err
        import logging
        import re

        logger = logging.getLogger(__name__)

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok({"tasks_created": 0, "tasks": [], "filtered_out": 0})

            # Validate template fields against first row
            template_fields = re.findall(r'\{(\w+)\}', self.name_template)
            if self.description_template:
                template_fields.extend(re.findall(r'\{(\w+)\}', self.description_template))

            if data and template_fields:
                missing = [f for f in template_fields if f not in data[0]]
                if missing:
                    # Check for ?-prefixed versions (REQL output)
                    still_missing = []
                    for f in missing:
                        if f"?{f}" not in data[0]:
                            still_missing.append(f)
                    if still_missing:
                        return pipeline_err(
                            "create_task",
                            f"Template field(s) not found in row: {still_missing}. "
                            f"Available fields: {list(data[0].keys())}"
                        )

            tasks_created = []
            filtered_out = 0
            filter_reasons = {}

            for row in data:
                # Apply filter predicates
                filter_result = self._apply_filter_predicates(row)
                if filter_result:
                    filtered_out += 1
                    filter_reasons[filter_result] = filter_reasons.get(filter_result, 0) + 1
                    continue

                task_data = self._create_task_data(row)
                tasks_created.append(task_data)

            # Check for file-based output mode
            output_dir = None
            if ctx and hasattr(ctx, 'params') and ctx.params:
                output_dir = ctx.params.get("output_dir")

            if output_dir and not self.dry_run:
                from pathlib import Path
                from reter_code.dsl.core import _get_project_root
                project_root = _get_project_root(ctx)
                resolved_dir = str(Path(project_root) / ".reter_code" / output_dir)
                written_files = self._write_task_files(tasks_created, resolved_dir)
                return pipeline_ok({
                    "tasks_created": len(tasks_created),
                    "filtered_out": filtered_out,
                    "filter_reasons": filter_reasons,
                    "output_dir": resolved_dir,
                    "task_files": written_files,
                })
            else:
                return pipeline_ok({
                    "tasks_created": len(tasks_created),
                    "filtered_out": filtered_out,
                    "filter_reasons": filter_reasons,
                    "tasks": tasks_created,
                    "_claude_code_hint": "Create these as Claude Code tasks using TaskCreate for each item in 'tasks'"
                })

        except Exception as e:
            logger.error(f"Create task step failed: {e}", exc_info=True)
            return pipeline_err("create_task", str(e))

    def _apply_filter_predicates(self, row):
        """Apply filter predicates to a row. Returns reason string if filtered, None if passed."""
        for predicate in self.filter_predicates:
            if predicate == "skip_same_names":
                # Skip if all methods in cluster have same name (interface pattern)
                members = row.get("members", row.get("?members", []))
                if members and isinstance(members, list) and len(members) > 1:
                    names = set()
                    for m in members:
                        if isinstance(m, dict):
                            name = m.get("name", m.get("method_name", ""))
                        else:
                            name = str(m).split(".")[-1] if "." in str(m) else str(m)
                        if name:
                            names.add(name)
                    if len(names) == 1:
                        return "skip_same_names"

            elif predicate == "skip_trivial":
                # Skip if methods are too small (< 3 lines average)
                members = row.get("members", row.get("?members", []))
                if members and isinstance(members, list):
                    total_lines = 0
                    count = 0
                    for m in members:
                        if isinstance(m, dict):
                            lines = m.get("line_count", m.get("lines", 0))
                            if lines:
                                total_lines += lines
                                count += 1
                    if count > 0 and total_lines / count < 3:
                        return "skip_trivial"

            elif predicate == "skip_boilerplate":
                # Skip if all methods are boilerplate (__init__, toString, etc.)
                members = row.get("members", row.get("?members", []))
                if members and isinstance(members, list):
                    all_boilerplate = True
                    for m in members:
                        if isinstance(m, dict):
                            name = m.get("name", m.get("method_name", ""))
                        else:
                            name = str(m).split(".")[-1] if "." in str(m) else str(m)
                        if name and name not in self.BOILERPLATE_METHODS:
                            all_boilerplate = False
                            break
                    if all_boilerplate and members:
                        return "skip_boilerplate"

            elif predicate == "skip_single_file":
                # Skip if all members are in the same file
                members = row.get("members", row.get("?members", []))
                if members and isinstance(members, list) and len(members) > 1:
                    files = set()
                    for m in members:
                        if isinstance(m, dict):
                            f = m.get("file", m.get("source_file", ""))
                        else:
                            f = ""
                        if f:
                            files.add(f)
                    if len(files) == 1:
                        return "skip_single_file"

        return None

    @staticmethod
    def _sanitize_filename(name):
        """Convert task name to filesystem-safe string."""
        import re as _re
        # Strip markdown formatting
        s = _re.sub(r'[*_`#\[\]]', '', name)
        # Replace unsafe chars with underscore
        s = _re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s)
        # Collapse whitespace and dashes to single underscore
        s = _re.sub(r'[\s\-]+', '_', s)
        # Collapse runs of underscores
        s = _re.sub(r'_+', '_', s)
        # Strip leading/trailing underscores
        s = s.strip('_')
        # Truncate to 80 chars
        if len(s) > 80:
            s = s[:80].rstrip('_')
        return s

    def _render_task_file(self, task):
        """Render a task dict as a self-contained markdown file."""
        import json

        lines = []

        # Title
        lines.append(f"# {task['name']}")
        lines.append("")

        # Metadata bar
        meta_parts = [f"Category: {task.get('category', 'annotation')}",
                      f"Priority: {task.get('priority', 'medium')}"]
        if task.get('metadata', {}).get('source_tool'):
            meta_parts.append(f"Source: {task['metadata']['source_tool']}")
        elif self.source_tool:
            meta_parts.append(f"Source: {self.source_tool}")
        if task.get('affects'):
            meta_parts.append(f"Affects: {task['affects']}")
        lines.append(f"> {' | '.join(meta_parts)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Detection & Review
        lines.append("## Detection & Review")
        lines.append("")
        if task.get('description'):
            # Unescape literal \n and \" from CADSL templates
            desc = task['description'].replace('\\n', '\n').replace('\\"', '"')
            lines.append(desc)
            lines.append("")

        # Prompt section (separate if present)
        if task.get('prompt'):
            lines.append("## Task Instructions")
            lines.append("")
            prompt = task['prompt'].replace('\\n', '\n').replace('\\"', '"')
            lines.append(prompt)
            lines.append("")

        lines.append("---")
        lines.append("")

        # Lifecycle
        lines.append("## Lifecycle")
        lines.append("")
        lines.append("**If FALSE POSITIVE:** Delete this file.")
        lines.append("")
        lines.append("**If TRUE POSITIVE:** Rewrite this file with your implementation plan (what to change, files affected, risk).")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Metadata JSON block
        meta_block = {}
        if task.get('metadata'):
            meta_block['metadata'] = task['metadata']
        if task.get('source_row'):
            meta_block['source_row'] = task['source_row']
        if meta_block:
            lines.append("## Metadata")
            lines.append("```json")
            lines.append(json.dumps(meta_block, indent=2, default=str))
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def _write_task_files(self, tasks, output_dir):
        """Write each task as a self-contained .md file. Returns list of written paths."""
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        written = []
        for i, task in enumerate(tasks, 1):
            sanitized = self._sanitize_filename(task.get('name', f'task_{i}'))
            filename = f"{i:03d}_{sanitized}.md"
            filepath = out / filename
            content = self._render_task_file(task)
            filepath.write_text(content, encoding='utf-8')
            written.append(str(filepath))

        return written

    def _create_task_data(self, row):
        """Create task data from a row using templates."""
        import re

        def expand_template(template, row):
            """Expand {field} placeholders with row values."""
            if not template:
                return template

            result = template
            for match in re.finditer(r'\{(\w+)\}', template):
                field = match.group(1)
                # Try both with and without ? prefix (REQL output)
                value = row.get(field, row.get(f"?{field}", ""))
                result = result.replace(match.group(0), str(value) if value else "")
            return result

        def expand_metadata_value(value, row):
            """Expand metadata value - can be a template string or direct value."""
            if isinstance(value, str) and "{" in value:
                return expand_template(value, row)
            return value

        task_data = {
            "name": expand_template(self.name_template, row),
            "category": self.category,
            "priority": self.priority,
        }

        if self.description_template:
            task_data["description"] = expand_template(self.description_template, row)

        if self.prompt_template:
            task_data["prompt"] = expand_template(self.prompt_template, row)

        if self.affects_field:
            affects_value = row.get(self.affects_field, row.get(f"?{self.affects_field}"))
            if affects_value:
                task_data["affects"] = str(affects_value)

        # Build metadata from template
        if self.metadata_template:
            metadata = {}
            for key, value in self.metadata_template.items():
                expanded = expand_metadata_value(value, row)
                # Try to convert numeric strings back to numbers
                if isinstance(expanded, str):
                    try:
                        if "." in expanded:
                            expanded = float(expanded)
                        else:
                            expanded = int(expanded)
                    except (ValueError, TypeError):
                        pass
                metadata[key] = expanded
            task_data["metadata"] = metadata

        # Include original row data for reference
        task_data["source_row"] = dict(row)

        return task_data
