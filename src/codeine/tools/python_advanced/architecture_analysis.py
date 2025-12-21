"""
Architecture Analysis Tools

Provides tools for analyzing codebase architecture, structure, and organization.
"""

from typing import Dict, Any
from collections import defaultdict
import os
import time
from .base import AdvancedToolsBase


class ArchitectureAnalysisTools(AdvancedToolsBase):
    """Architecture and structure analysis tools."""

    def get_exception_hierarchy(self, instance_name: str) -> Dict[str, Any]:
        """
        Get exception class hierarchy (classes inheriting from Exception).

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, exception classes, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            query = f"""
                SELECT ?exception ?name ?parent
                WHERE {{
                    ?exception type {class_concept} .
                    ?exception name ?name .
                    ?exception inheritsFrom ?parent .
                    FILTER(REGEX(?parent, "Exception"))
                }}
                ORDER BY ?parent ?name
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            exceptions = [
                {
                    "qualified_name": row[0],
                    "name": row[1],
                    "parent": row[2]
                }
                for row in rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "exceptions": exceptions,
                "count": len(exceptions),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "exceptions": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def get_package_structure(self, instance_name: str) -> Dict[str, Any]:
        """
        Get package/module structure of the codebase.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, modules organized by path, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            module_concept = self._concept('Module')
            query = f"""
                SELECT ?module ?file
                WHERE {{
                    ?module type {module_concept} .
                    ?module inFile ?file
                }}
                ORDER BY ?file
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            modules = [
                {
                    "module": row[0],
                    "file": row[1]
                }
                for row in rows
            ]

            # Group by directory
            by_directory = defaultdict(list)
            for module in modules:
                directory = os.path.dirname(module["file"]) or "."
                by_directory[directory].append(module)

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "modules": modules,
                "by_directory": dict(by_directory),
                "module_count": len(modules),
                "directory_count": len(by_directory),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "modules": [],
                "by_directory": {},
                "module_count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def find_duplicate_names(self, instance_name: str) -> Dict[str, Any]:
        """
        Find entities with duplicate names across modules.

        Useful for identifying naming conflicts and potential confusion.

        Args:
            instance_name: RETER instance name

        Returns:
            dict with success, duplicates list, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            func_concept = self._concept('Function')

            # Find duplicate class names
            class_query = f"""
                SELECT ?name (COUNT(?class) AS ?count)
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name
                }}
                GROUP BY ?name
                HAVING (?count > 1)
                ORDER BY DESC(?count)
            """
            queries.append(class_query.strip())
            class_result = self.reter.reql(class_query)
            class_rows = self._query_to_list(class_result)

            # Find duplicate function names
            function_query = f"""
                SELECT ?name (COUNT(?func) AS ?count)
                WHERE {{
                    ?func type {func_concept} .
                    ?func name ?name
                }}
                GROUP BY ?name
                HAVING (?count > 1)
                ORDER BY DESC(?count)
            """
            queries.append(function_query.strip())
            func_result = self.reter.reql(function_query)
            func_rows = self._query_to_list(func_result)

            # BATCH: Get all class details in one query (fixes N+1 problem)
            # Use inFile (works for all languages) instead of inModule
            all_class_details_query = f"""
                SELECT ?name ?class ?file
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name .
                    ?class inFile ?file
                }}
            """
            queries.append(all_class_details_query.strip())
            all_class_details_result = self.reter.reql(all_class_details_query)
            all_class_details = self._query_to_list(all_class_details_result)

            # Build lookup: name -> list of (class, file)
            class_details_map = {}
            for name, class_id, file in all_class_details:
                if name not in class_details_map:
                    class_details_map[name] = []
                class_details_map[name].append((class_id, file))

            # Build duplicate_classes from pre-fetched data
            duplicate_classes = []
            for row in class_rows:
                name, count = row[0], row[1]
                details = class_details_map.get(name, [])
                duplicate_classes.append({
                    "name": name,
                    "type": "class",
                    "count": count,
                    "locations": [
                        {"class": d[0], "file": d[1]}
                        for d in details
                    ]
                })

            # BATCH: Get all function details in one query (fixes N+1 problem)
            # Use inFile (works for all languages) instead of inModule
            all_func_details_query = f"""
                SELECT ?name ?func ?file
                WHERE {{
                    ?func type {func_concept} .
                    ?func name ?name .
                    ?func inFile ?file
                }}
            """
            queries.append(all_func_details_query.strip())
            all_func_details_result = self.reter.reql(all_func_details_query)
            all_func_details = self._query_to_list(all_func_details_result)

            # Build lookup: name -> list of (func, file)
            func_details_map = {}
            for name, func_id, file in all_func_details:
                if name not in func_details_map:
                    func_details_map[name] = []
                func_details_map[name].append((func_id, file))

            # Build duplicate_functions from pre-fetched data
            duplicate_functions = []
            for row in func_rows:
                name, count = row[0], row[1]
                details = func_details_map.get(name, [])
                duplicate_functions.append({
                    "name": name,
                    "type": "function",
                    "count": count,
                    "locations": [
                        {"function": d[0], "file": d[1]}
                        for d in details
                    ]
                })

            all_duplicates = duplicate_classes + duplicate_functions

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "duplicates": all_duplicates,
                "duplicate_classes": len(duplicate_classes),
                "duplicate_functions": len(duplicate_functions),
                "count": len(all_duplicates),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "duplicates": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def get_complexity_metrics(self, instance_name: str) -> Dict[str, Any]:
        """
        Calculate complexity metrics for the codebase.

        Provides aggregated metrics including:
        - Method count per class (distribution)
        - Parameter count distribution
        - Inheritance depth statistics
        - Call graph fan-in/fan-out

        Args:
            instance_name: RETER instance name

        Returns:
            success: Whether analysis succeeded
            metrics: Dictionary of complexity metrics
            queries: List of REQL queries executed
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')
            func_concept = self._concept('Function')
            param_concept = self._concept('Parameter')

            # Method count per class distribution
            methods_query = f"""
                SELECT ?class ?class_name (COUNT(?method) AS ?method_count)
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?class_name .
                    ?method type {method_concept} .
                    ?method definedIn ?class
                }}
                GROUP BY ?class ?class_name
                ORDER BY DESC(?method_count)
            """
            queries.append(methods_query.strip())
            methods_result = self.reter.reql(methods_query)
            methods_rows = self._query_to_list(methods_result)

            class_sizes = [
                {"class": row[0], "class_name": row[1], "method_count": row[2]}
                for row in methods_rows
            ]

            # Calculate distribution stats
            if class_sizes:
                method_counts = [c["method_count"] for c in class_sizes]
                avg_methods = sum(method_counts) / len(method_counts)
                max_methods = max(method_counts)
                min_methods = min(method_counts)
            else:
                avg_methods = max_methods = min_methods = 0

            # Parameter count distribution
            params_query = f"""
                SELECT ?func ?func_name (COUNT(?param) AS ?param_count)
                WHERE {{
                    ?func type ?type .
                    ?func name ?func_name .
                    ?param type {param_concept} .
                    ?param ofFunction ?func .
                    FILTER(?type = {func_concept} || ?type = {method_concept})
                }}
                GROUP BY ?func ?func_name
                ORDER BY DESC(?param_count)
            """
            queries.append(params_query.strip())
            params_result = self.reter.reql(params_query)
            params_rows = self._query_to_list(params_result)

            param_distribution = [
                {"function": row[0], "function_name": row[1], "parameter_count": row[2]}
                for row in params_rows
            ]

            if param_distribution:
                param_counts = [p["parameter_count"] for p in param_distribution]
                avg_params = sum(param_counts) / len(param_counts)
                max_params = max(param_counts)
            else:
                avg_params = max_params = 0

            # Inheritance depth (classes with parents)
            inheritance_query = f"""
                SELECT ?child ?child_name (COUNT(?parent) AS ?parent_count)
                WHERE {{
                    ?child type {class_concept} .
                    ?child name ?child_name .
                    ?child inheritsFrom ?parent
                }}
                GROUP BY ?child ?child_name
                ORDER BY DESC(?parent_count)
            """
            queries.append(inheritance_query.strip())
            inheritance_result = self.reter.reql(inheritance_query)
            inheritance_rows = self._query_to_list(inheritance_result)

            inheritance_depth = [
                {"class": row[0], "class_name": row[1], "parent_count": row[2]}
                for row in inheritance_rows
            ]

            # Call graph metrics (fan-out: number of callees per function)
            fanout_query = """
                SELECT ?caller ?caller_name (COUNT(?callee) AS ?call_count)
                WHERE {
                    ?caller calls ?callee .
                    ?caller name ?caller_name
                }
                GROUP BY ?caller ?caller_name
                ORDER BY DESC(?call_count)
            """
            queries.append(fanout_query.strip())
            fanout_result = self.reter.reql(fanout_query)
            fanout_rows = self._query_to_list(fanout_result)

            fan_out = [
                {"function": row[0], "function_name": row[1], "calls_count": row[2]}
                for row in fanout_rows
            ]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "metrics": {
                    "class_complexity": {
                        "average_methods_per_class": round(avg_methods, 2),
                        "max_methods_in_class": max_methods,
                        "min_methods_in_class": min_methods,
                        "total_classes": len(class_sizes),
                        "top_10_largest": class_sizes[:10]
                    },
                    "parameter_complexity": {
                        "average_parameters": round(avg_params, 2),
                        "max_parameters": max_params,
                        "total_functions": len(param_distribution),
                        "top_10_longest": param_distribution[:10]
                    },
                    "inheritance_complexity": {
                        "total_classes_with_parents": len(inheritance_depth),
                        "max_parent_count": max(inheritance_depth, key=lambda x: x["parent_count"])["parent_count"] if inheritance_depth else 0,
                        "top_10_deepest": inheritance_depth[:10]
                    },
                    "call_complexity": {
                        "total_callers": len(fan_out),
                        "max_fan_out": max(fan_out, key=lambda x: x["calls_count"])["calls_count"] if fan_out else 0,
                        "top_10_highest_fanout": fan_out[:10]
                    }
                },
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "metrics": {},
                "queries": queries,
                "time_ms": time_ms
            }

    def get_architecture_overview(self, instance_name: str, output_format: str = "json") -> Dict[str, Any]:
        """
        Generate high-level architectural overview of the codebase.

        Provides:
        - Module count by directory
        - Class count by module
        - Exception hierarchy summary
        - Most connected classes (hub classes)
        - Architectural statistics

        Args:
            instance_name: RETER instance name
            output_format: Output format:
                - "json": Structured data (default)
                - "markdown": Human-readable markdown text
                - "mermaid": Mermaid block diagram

        Returns:
            success: Whether analysis succeeded
            overview: Architectural overview data (if output_format="json")
            markdown: Markdown text (if output_format="markdown")
            diagram: Mermaid block diagram (if output_format="mermaid")
            queries: List of REQL queries executed
        """
        start_time = time.time()
        queries = []
        try:
            module_concept = self._concept('Module')
            class_concept = self._concept('Class')

            # Get module count by directory
            module_query = f"""
                SELECT ?module ?name ?file
                WHERE {{
                    ?module type {module_concept} .
                    ?module name ?name .
                    ?module inFile ?file
                }}
            """
            queries.append(module_query.strip())
            module_result = self.reter.reql(module_query)
            module_rows = self._query_to_list(module_result)

            modules_by_dir = defaultdict(list)
            for row in module_rows:
                file_path = row[2]
                directory = os.path.dirname(file_path) or "."
                modules_by_dir[directory].append(row[1])

            # Get class count by file
            class_file_query = f"""
                SELECT ?file (COUNT(?class) AS ?class_count)
                WHERE {{
                    ?class type {class_concept} .
                    ?class inFile ?file
                }}
                GROUP BY ?file
                ORDER BY DESC(?class_count)
            """
            queries.append(class_file_query.strip())
            class_file_result = self.reter.reql(class_file_query)
            class_file_rows = self._query_to_list(class_file_result)

            classes_by_module = [
                {"file": row[0], "class_count": row[1]}
                for row in class_file_rows
            ]

            # Get exception hierarchy count
            exception_query = f"""
                SELECT ?exc ?name
                WHERE {{
                    ?exc type {class_concept} .
                    ?exc name ?name .
                    ?exc inheritsFrom ?parent .
                    FILTER(CONTAINS(?name, "Error") || CONTAINS(?name, "Exception"))
                }}
            """
            queries.append(exception_query.strip())
            exception_result = self.reter.reql(exception_query)
            exception_rows = self._query_to_list(exception_result)

            # Find hub classes (most inherited from)
            hub_query = f"""
                SELECT ?class ?name (COUNT(?child) AS ?child_count)
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name .
                    ?child inheritsFrom ?class
                }}
                GROUP BY ?class ?name
                ORDER BY DESC(?child_count)
            """
            queries.append(hub_query.strip())
            hub_result = self.reter.reql(hub_query)
            hub_rows = self._query_to_list(hub_result)

            hub_classes = [
                {"class": row[0], "class_name": row[1], "children_count": row[2]}
                for row in hub_rows[:10]
            ]

            # Calculate overall statistics
            total_modules = len(module_rows)
            total_classes = sum(c["class_count"] for c in classes_by_module)
            avg_classes_per_module = round(total_classes / total_modules, 2) if total_modules > 0 else 0

            # Build structured overview data
            overview_data = {
                "directory_structure": {
                    "directories": list(modules_by_dir.keys()),
                    "directory_count": len(modules_by_dir),
                    "modules_by_directory": {
                        dir: len(mods) for dir, mods in modules_by_dir.items()
                    }
                },
                "module_statistics": {
                    "total_modules": total_modules,
                    "total_classes": total_classes,
                    "average_classes_per_module": avg_classes_per_module,
                    "top_10_largest_modules": classes_by_module[:10]
                },
                "exception_hierarchy": {
                    "total_exceptions": len(exception_rows),
                    "custom_exceptions": exception_rows[:20]
                },
                "hub_classes": {
                    "description": "Classes that are most inherited from (base classes)",
                    "top_10": hub_classes
                }
            }

            time_ms = (time.time() - start_time) * 1000

            # Generate output based on output_format
            fmt = output_format.lower()
            if fmt == "mermaid":
                diagram = self._generate_architecture_block_diagram(
                    modules_by_dir,
                    classes_by_module,
                    hub_classes,
                    total_modules,
                    total_classes
                )
                return {
                    "success": True,
                    "format": "mermaid",
                    "diagram": diagram,
                    "statistics": {
                        "total_modules": total_modules,
                        "total_classes": total_classes,
                        "directory_count": len(modules_by_dir)
                    },
                    "queries": queries,
                    "time_ms": time_ms
                }
            elif fmt == "markdown":
                markdown = self._generate_architecture_markdown(
                    modules_by_dir,
                    classes_by_module,
                    hub_classes,
                    exception_rows,
                    total_modules,
                    total_classes
                )
                return {
                    "success": True,
                    "format": "markdown",
                    "markdown": markdown,
                    "statistics": {
                        "total_modules": total_modules,
                        "total_classes": total_classes,
                        "directory_count": len(modules_by_dir)
                    },
                    "queries": queries,
                    "time_ms": time_ms
                }
            else:
                return {
                    "success": True,
                    "format": "json",
                    "overview": overview_data,
                    "queries": queries,
                    "time_ms": time_ms
                }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "overview": {},
                "queries": queries,
                "time_ms": time_ms
            }

    def _generate_architecture_block_diagram(
        self,
        modules_by_dir: dict,
        classes_by_module: list,
        hub_classes: list,
        total_modules: int,
        total_classes: int
    ) -> str:
        """Generate a Mermaid block diagram representing the codebase architecture."""
        lines = ["block-beta"]
        lines.append("  columns 3")
        lines.append("")

        # Statistics block (spans full width)
        lines.append("  block:stats:3")
        lines.append("    columns 1")
        lines.append(f"    STATS[\"**Codebase Statistics**<br/>Modules: {total_modules}<br/>Classes: {total_classes}<br/>Directories: {len(modules_by_dir)}\"]")
        lines.append("  end")
        lines.append("")

        # Directory structure block
        lines.append("  block:dirs:1")
        lines.append("    columns 1")
        lines.append("    DIR_TITLE[\"**Directory Structure**\"]")

        sorted_dirs = sorted(modules_by_dir.items(), key=lambda x: len(x[1]), reverse=True)[:8]
        for i, (dir_name, modules) in enumerate(sorted_dirs):
            dir_id = f"dir{i}"
            display_name = dir_name if len(dir_name) < 30 else "..." + dir_name[-27:]
            lines.append(f"    {dir_id}[(\"{display_name}<br/>{len(modules)} modules\")]")

        lines.append("  end")
        lines.append("")

        # Top modules block
        lines.append("  block:modules:1")
        lines.append("    columns 1")
        lines.append("    MOD_TITLE[\"**Largest Modules**\"]")

        top_modules = classes_by_module[:8]
        for i, mod in enumerate(top_modules):
            mod_id = f"mod{i}"
            mod_name = mod["file"]
            if len(mod_name) > 25:
                mod_name = mod_name[:22] + "..."
            lines.append(f"    {mod_id}[\"{mod_name}<br/>{mod['class_count']} classes\"]")

        lines.append("  end")
        lines.append("")

        # Hub classes block
        lines.append("  block:hubs:1")
        lines.append("    columns 1")
        lines.append("    HUB_TITLE[\"**Base Classes**\"]")

        top_hubs = hub_classes[:8]
        for i, hub in enumerate(top_hubs):
            hub_id = f"hub{i}"
            hub_name = hub["class_name"]
            if len(hub_name) > 25:
                hub_name = hub_name[:22] + "..."
            lines.append(f"    {hub_id}((\"{hub_name}<br/>{hub['children_count']} children\"))")

        lines.append("  end")
        lines.append("")

        # Add styling
        lines.append("  style STATS fill:#e1f5ff,stroke:#01579b,stroke-width:2px")
        lines.append("  style DIR_TITLE fill:#fff3e0,stroke:#e65100,stroke-width:2px")
        lines.append("  style MOD_TITLE fill:#f3e5f5,stroke:#4a148c,stroke-width:2px")
        lines.append("  style HUB_TITLE fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px")

        return "\n".join(lines)

    def _generate_architecture_markdown(
        self,
        modules_by_dir: dict,
        classes_by_module: list,
        hub_classes: list,
        exception_rows: list,
        total_modules: int,
        total_classes: int
    ) -> str:
        """Generate a markdown text representation of the codebase architecture."""
        lines = ["# Codebase Architecture Overview", ""]

        # Statistics section
        lines.append("## Statistics")
        lines.append("")
        lines.append(f"- **Total Modules:** {total_modules}")
        lines.append(f"- **Total Classes:** {total_classes}")
        lines.append(f"- **Directories:** {len(modules_by_dir)}")
        avg_classes = round(total_classes / total_modules, 2) if total_modules > 0 else 0
        lines.append(f"- **Average Classes per Module:** {avg_classes}")
        lines.append("")

        # Directory structure section
        lines.append("## Directory Structure")
        lines.append("")
        sorted_dirs = sorted(modules_by_dir.items(), key=lambda x: len(x[1]), reverse=True)
        lines.append("| Directory | Module Count |")
        lines.append("|-----------|--------------|")
        for dir_name, modules in sorted_dirs[:15]:
            lines.append(f"| `{dir_name}` | {len(modules)} |")
        if len(sorted_dirs) > 15:
            lines.append(f"| ... | ({len(sorted_dirs) - 15} more) |")
        lines.append("")

        # Largest modules section
        lines.append("## Largest Modules (by class count)")
        lines.append("")
        lines.append("| Module | Classes |")
        lines.append("|--------|---------|")
        for mod in classes_by_module[:15]:
            lines.append(f"| `{mod['file']}` | {mod['class_count']} |")
        if len(classes_by_module) > 15:
            lines.append(f"| ... | ({len(classes_by_module) - 15} more) |")
        lines.append("")

        # Hub classes section
        lines.append("## Base Classes (Most Inherited)")
        lines.append("")
        lines.append("| Class | Children |")
        lines.append("|-------|----------|")
        for hub in hub_classes[:10]:
            lines.append(f"| `{hub['class_name']}` | {hub['children_count']} |")
        lines.append("")

        # Exception hierarchy
        if exception_rows:
            lines.append("## Custom Exceptions")
            lines.append("")
            lines.append(f"Found **{len(exception_rows)}** custom exception classes:")
            lines.append("")
            for exc in exception_rows[:15]:
                lines.append(f"- `{exc[1]}`")
            if len(exception_rows) > 15:
                lines.append(f"- ... and {len(exception_rows) - 15} more")
            lines.append("")

        return "\n".join(lines)
