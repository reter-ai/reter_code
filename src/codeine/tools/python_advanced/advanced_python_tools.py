"""
Advanced Code Analysis Tools

Implements advanced code analysis tools using REQL queries.
These tools provide deeper insights into code quality, dependencies, testing,
change impact, and documentation.

Supports multiple languages via the LanguageSupport module:
- "oo" (default): Language-independent queries (Python + JavaScript)
- "python" or "py": Python-specific queries
- "javascript" or "js": JavaScript-specific queries

Now extends AdvancedToolsBase and delegates to specialized tool classes
for methods that have been extracted.
"""

from typing import Dict, Any, List, Optional
import time

from .base import AdvancedToolsBase
from .code_quality import CodeQualityTools
from .dependency_analysis import DependencyAnalysisTools
from .pattern_detection import PatternDetectionTools
from .change_impact import ChangeImpactTools
from .type_analysis import TypeAnalysisTools
from .exception_analysis import ExceptionAnalysisTools
from .test_analysis import TestAnalysisTools
from .documentation_analysis import DocumentationAnalysisTools
from .architecture_analysis import ArchitectureAnalysisTools
from codeine.services.language_support import LanguageType


class AdvancedPythonTools(AdvancedToolsBase):
    """
    Advanced code analysis tools using REQL

    Supports Python, JavaScript, and language-independent queries.

    Provides tools for:
    - Code quality metrics
    - Dependency analysis
    - Pattern detection
    - API analysis
    - Test coverage
    - Change impact analysis
    - Documentation analysis
    - Architecture metrics

    Methods that have been extracted to specialized classes now delegate
    to those classes for the actual implementation.
    """

    def __init__(self, reter_wrapper, language: LanguageType = "oo"):
        """
        Initialize with ReterWrapper instance

        Args:
            reter_wrapper: ReterWrapper instance with loaded code
            language: Programming language to analyze ("oo", "python", "javascript")
        """
        super().__init__(reter_wrapper, language)

        # Initialize specialized tool classes for delegation (passing language)
        self._code_quality = CodeQualityTools(reter_wrapper, language)
        self._dependency = DependencyAnalysisTools(reter_wrapper, language)
        self._pattern = PatternDetectionTools(reter_wrapper, language)
        self._change_impact = ChangeImpactTools(reter_wrapper, language)
        self._type_analysis = TypeAnalysisTools(reter_wrapper, language)
        self._exception_analysis = ExceptionAnalysisTools(reter_wrapper, language)
        self._test_analysis = TestAnalysisTools(reter_wrapper, language)
        self._documentation_analysis = DocumentationAnalysisTools(reter_wrapper, language)
        self._architecture_analysis = ArchitectureAnalysisTools(reter_wrapper, language)

    # =========================================================================
    # 1. CODE QUALITY METRICS (delegated to CodeQualityTools)
    # =========================================================================

    def find_large_classes(
        self,
        instance_name: str,
        threshold: int = 20,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find classes with too many methods (God classes).

        Delegates to CodeQualityTools.
        """
        return self._code_quality.find_large_classes(instance_name, threshold)

    def find_long_parameter_lists(
        self,
        instance_name: str,
        threshold: int = 5,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find functions/methods with too many parameters.

        Delegates to CodeQualityTools.
        """
        return self._code_quality.find_long_parameter_lists(instance_name, threshold)

    def find_magic_numbers(
        self,
        instance_name: str,
        exclude_common: bool = True,
        min_occurrences: int = 1,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find magic numbers (numeric literals) in code.

        Magic numbers are numeric literals embedded directly in code that should
        typically be extracted to named constants for better maintainability.

        Delegates to CodeQualityTools.
        """
        return self._code_quality.find_magic_numbers(
            instance_name, exclude_common, min_occurrences, limit, offset
        )

    # =========================================================================
    # 2. DEPENDENCY ANALYSIS (delegated to DependencyAnalysisTools)
    # =========================================================================

    def get_import_graph(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get complete module import dependency graph.

        Delegates to DependencyAnalysisTools.
        """
        return self._dependency.get_import_graph(instance_name)

    def find_circular_imports(self, instance_name: str) -> Dict[str, Any]:
        """
        Find circular import dependencies.

        Delegates to DependencyAnalysisTools.
        """
        return self._dependency.find_circular_imports(instance_name)

    def get_external_dependencies(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get external package dependencies.

        Delegates to DependencyAnalysisTools.
        """
        return self._dependency.get_external_dependencies(instance_name)

    # =========================================================================
    # 3. PATTERN DETECTION (delegated to PatternDetectionTools)
    # =========================================================================

    def find_decorators_usage(
        self,
        instance_name: str,
        decorator_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find all uses of decorators, optionally filtered by name.

        Delegates to PatternDetectionTools.
        """
        return self._pattern.find_decorators_usage(instance_name, decorator_name)

    def get_magic_methods(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find all magic methods (__init__, __str__, etc.).

        Delegates to PatternDetectionTools.
        """
        return self._pattern.get_magic_methods(instance_name)

    def get_interface_implementations(
        self,
        instance_name: str,
        interface_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find classes that implement abstract base classes/interfaces.

        Delegates to PatternDetectionTools.
        """
        return self._pattern.get_interface_implementations(instance_name, interface_name)

    # =========================================================================
    # 4. API ANALYSIS (delegated to PatternDetectionTools)
    # =========================================================================

    def get_public_api(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get all public classes and functions (not starting with _).

        Delegates to PatternDetectionTools.
        """
        return self._pattern.get_public_api(instance_name)

    # =========================================================================
    # 5. TYPE ANALYSIS (delegated to TypeAnalysisTools)
    # =========================================================================

    def get_type_hints(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Extract all type hints from parameters and return types.

        Delegates to TypeAnalysisTools.
        """
        return self._type_analysis.get_type_hints(instance_name)

    def find_untyped_functions(self, instance_name: str) -> Dict[str, Any]:
        """
        Find functions/methods without type hints.

        Delegates to TypeAnalysisTools.
        """
        return self._type_analysis.find_untyped_functions(instance_name)

    # =========================================================================
    # 6. TEST COVERAGE ANALYSIS (delegated to TestAnalysisTools)
    # =========================================================================

    def find_test_files(self, instance_name: str) -> Dict[str, Any]:
        """Find test files based on naming conventions. Delegates to TestAnalysisTools."""
        return self._test_analysis.find_test_files(instance_name)

    def find_test_fixtures(self, instance_name: str) -> Dict[str, Any]:
        """Find pytest fixtures. Delegates to TestAnalysisTools."""
        return self._test_analysis.find_test_fixtures(instance_name)

    # =========================================================================
    # 7. CHANGE IMPACT ANALYSIS (delegated to ChangeImpactTools)
    # =========================================================================

    def predict_change_impact(
        self,
        instance_name: str,
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Predict impact of changing a function/method/class.

        Delegates to ChangeImpactTools.
        """
        return self._change_impact.predict_change_impact(instance_name, entity_name)

    def find_callers_recursive(
        self,
        instance_name: str,
        target_name: str
    ) -> Dict[str, Any]:
        """
        Find all callers of a function/method (recursive, transitive).

        Delegates to ChangeImpactTools.
        """
        return self._change_impact.find_callers_recursive(instance_name, target_name)

    def find_callees_recursive(
        self,
        instance_name: str,
        source_name: str
    ) -> Dict[str, Any]:
        """
        Find all functions/methods called by a function (recursive, transitive).

        Delegates to ChangeImpactTools.
        """
        return self._change_impact.find_callees_recursive(instance_name, source_name)

    # =========================================================================
    # 8. DOCUMENTATION ANALYSIS (delegated to DocumentationAnalysisTools)
    # =========================================================================

    def find_undocumented_code(self, instance_name: str) -> Dict[str, Any]:
        """Find undocumented classes and functions. Delegates to DocumentationAnalysisTools."""
        return self._documentation_analysis.find_undocumented_code(instance_name)

    def get_api_documentation(self, instance_name: str) -> Dict[str, Any]:
        """Extract API documentation. Delegates to DocumentationAnalysisTools."""
        return self._documentation_analysis.get_api_documentation(instance_name)

    # =========================================================================
    # 8. ARCHITECTURE METRICS (delegated to ArchitectureAnalysisTools)
    # =========================================================================

    def get_exception_hierarchy(self, instance_name: str) -> Dict[str, Any]:
        """Get exception class hierarchy. Delegates to ArchitectureAnalysisTools."""
        return self._architecture_analysis.get_exception_hierarchy(instance_name)

    def get_package_structure(self, instance_name: str) -> Dict[str, Any]:
        """Get package/module structure. Delegates to ArchitectureAnalysisTools."""
        return self._architecture_analysis.get_package_structure(instance_name)

    # =====================================================================
    # NEW AGGREGATION TOOLS (Phase 2)
    # =====================================================================

    def find_duplicate_names(self, instance_name: str) -> Dict[str, Any]:
        """
        Find entities with duplicate names across modules.

        Useful for identifying naming conflicts and potential confusion.

        Args:
            instance_name: RETER instance name

        Returns:
            success: Whether query succeeded
            duplicates: List of duplicate names with their occurrences
            count: Number of duplicate names found
            queries: List of REQL queries executed
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
                "count": len(all_duplicates),
                "duplicate_classes": len(duplicate_classes),
                "duplicate_functions": len(duplicate_functions),
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

    def find_unused_code(
        self,
        instance_name: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find potentially unused code in the codebase.

        Identifies:
        - Classes with no subclasses and no instantiation/usage
        - Functions/methods with no callers
        - Potentially dead code

        Args:
            instance_name: RETER instance name
            limit: Maximum number of results to return (default: 50)
            offset: Number of results to skip (default: 0)

        Returns:
            success: Whether query succeeded
            unused_classes: Classes that appear unused (paginated)
            unused_functions: Functions that appear unused (paginated)
            total_unused: Total count of ALL potentially unused entities (before pagination)
            count_returned: Number of items returned in this page
            limit: Limit used for pagination
            offset: Offset used for pagination
            has_more: Whether there are more results available
            queries: List of REQL queries executed
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')

            # Find classes with no subclasses using two-query approach (REQL doesn't support NOT EXISTS)
            # Query 1: Get all classes
            # Use inFile (works for all languages) instead of inModule
            all_classes_query = f"""
                SELECT ?class ?name ?file
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name .
                    ?class inFile ?file
                }}
            """
            queries.append(all_classes_query.strip())
            all_classes_result = self.reter.reql(all_classes_query)
            all_classes_rows = self._query_to_list(all_classes_result)

            # Query 2: Get all classes that have subclasses
            has_subclass_query = """
                SELECT DISTINCT ?parent
                WHERE {
                    ?child inheritsFrom ?parent
                }
            """
            queries.append(has_subclass_query.strip())
            has_subclass_result = self.reter.reql(has_subclass_query)
            has_subclass_rows = self._query_to_list(has_subclass_result)

            # Compute set difference: all_classes - has_subclasses
            parent_classes = {row[0] for row in has_subclass_rows}
            classes_no_children = [
                {"class": row[0], "name": row[1], "file": row[2], "reason": "no_subclasses"}
                for row in all_classes_rows
                if row[0] not in parent_classes
            ]

            # Find functions with no callers using two-query approach
            # Query 1: Get all functions/methods (filter magic methods in Python)
            # Use inFile (works for all languages) instead of inModule
            all_funcs_query = f"""
                SELECT ?func ?name ?file
                WHERE {{
                    ?func type ?type .
                    ?func name ?name .
                    ?func inFile ?file .
                    FILTER(?type = {func_concept} || ?type = {method_concept})
                }}
            """
            queries.append(all_funcs_query.strip())
            all_funcs_result = self.reter.reql(all_funcs_query)
            all_funcs_rows = self._query_to_list(all_funcs_result)

            # Query 2: Get all functions that are called
            has_caller_query = """
                SELECT DISTINCT ?callee
                WHERE {
                    ?caller calls ?callee
                }
            """
            queries.append(has_caller_query.strip())
            has_caller_result = self.reter.reql(has_caller_query)
            has_caller_rows = self._query_to_list(has_caller_result)

            # Compute set difference: all_funcs - has_callers (excluding magic methods)
            called_funcs = {row[0] for row in has_caller_rows}
            functions_no_callers = [
                {"function": row[0], "name": row[1], "file": row[2], "reason": "no_callers"}
                for row in all_funcs_rows
                if row[0] not in called_funcs and not (row[1].startswith("__") and row[1].endswith("__"))
            ]

            # Calculate totals before pagination
            total_classes = len(classes_no_children)
            total_functions = len(functions_no_callers)
            total_unused = total_classes + total_functions

            # Combine results for pagination
            all_unused = []
            all_unused.extend([{"type": "class", **item} for item in classes_no_children])
            all_unused.extend([{"type": "function", **item} for item in functions_no_callers])

            # Apply pagination
            paginated_results = all_unused[offset:offset + limit]

            # Split back into classes and functions
            paginated_classes = [item for item in paginated_results if item["type"] == "class"]
            paginated_functions = [item for item in paginated_results if item["type"] == "function"]

            # Remove the "type" field added for pagination
            for item in paginated_classes:
                del item["type"]
            for item in paginated_functions:
                del item["type"]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "unused_classes": paginated_classes,
                "unused_functions": paginated_functions,
                "total_unused": total_unused,
                "total_unused_classes": total_classes,
                "total_unused_functions": total_functions,
                "count_returned": len(paginated_results),
                "count_returned_classes": len(paginated_classes),
                "count_returned_functions": len(paginated_functions),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_unused,
                "next_offset": offset + limit if (offset + limit) < total_unused else None,
                "warning": "These are POTENTIAL unused code - verify before removing (may be entry points, external API, etc.)",
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "unused_classes": [],
                "unused_functions": [],
                "total_unused": 0,
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
        """
        Generate a Mermaid block diagram representing the codebase architecture.

        Creates a visual layout showing:
        - Directory structure with module counts
        - Top modules by class count
        - Hub classes (most inherited base classes)
        - Overall statistics

        Args:
            modules_by_dir: Dictionary mapping directories to module lists
            classes_by_module: List of modules with class counts
            hub_classes: List of most inherited classes
            total_modules: Total number of modules
            total_classes: Total number of classes

        Returns:
            Mermaid block diagram as string
        """
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

        # Show top directories by module count
        sorted_dirs = sorted(modules_by_dir.items(), key=lambda x: len(x[1]), reverse=True)[:8]
        for i, (dir_name, modules) in enumerate(sorted_dirs):
            # Sanitize directory name for use as block ID
            dir_id = f"dir{i}"
            # Truncate long directory names
            display_name = dir_name if len(dir_name) < 30 else "..." + dir_name[-27:]
            lines.append(f"    {dir_id}[(\"{display_name}<br/>{len(modules)} modules\")]")

        lines.append("  end")
        lines.append("")

        # Top modules block
        lines.append("  block:modules:1")
        lines.append("    columns 1")
        lines.append("    MOD_TITLE[\"**Largest Modules**\"]")

        # Show top 8 modules by class count
        top_modules = classes_by_module[:8]
        for i, mod in enumerate(top_modules):
            mod_id = f"mod{i}"
            mod_name = mod["file"]
            # Truncate long module names
            if len(mod_name) > 25:
                mod_name = mod_name[:22] + "..."
            lines.append(f"    {mod_id}[\"{mod_name}<br/>{mod['class_count']} classes\"]")

        lines.append("  end")
        lines.append("")

        # Hub classes block (base classes)
        lines.append("  block:hubs:1")
        lines.append("    columns 1")
        lines.append("    HUB_TITLE[\"**Base Classes**\"]")

        # Show top 8 hub classes
        top_hubs = hub_classes[:8]
        for i, hub in enumerate(top_hubs):
            hub_id = f"hub{i}"
            hub_name = hub["class_name"]
            # Truncate long class names
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
        """
        Generate a markdown text representation of the codebase architecture.

        Args:
            modules_by_dir: Dictionary mapping directories to module lists
            classes_by_module: List of modules with class counts
            hub_classes: List of most inherited classes
            exception_rows: List of exception classes
            total_modules: Total number of modules
            total_classes: Total number of classes

        Returns:
            Markdown formatted architecture overview
        """
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
        lines.append("| Module | Class Count |")
        lines.append("|--------|-------------|")
        for mod in classes_by_module[:10]:
            lines.append(f"| `{mod['module_name']}` | {mod['class_count']} |")
        lines.append("")

        # Hub classes section
        lines.append("## Hub Classes (most inherited)")
        lines.append("")
        lines.append("These are base classes with the most subclasses:")
        lines.append("")
        lines.append("| Class | Subclass Count |")
        lines.append("|-------|----------------|")
        for hub in hub_classes[:10]:
            lines.append(f"| `{hub['class_name']}` | {hub['children_count']} |")
        lines.append("")

        # Exception hierarchy section
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

            from collections import defaultdict
            import os

            modules_by_dir = defaultdict(list)
            for row in module_rows:
                file_path = row[2]
                directory = os.path.dirname(file_path) or "."
                modules_by_dir[directory].append(row[1])

            # Get class count by file
            # Use inFile (works for all languages) instead of inModule
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
                for row in hub_rows[:10]  # Top 10
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
                    "custom_exceptions": exception_rows[:20]  # Sample
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

    # =========================================================================
    # 10. CODE SMELL DETECTION (Based on Fowler's Refactoring)
    # =========================================================================

    def detect_long_functions(
        self,
        instance_name: str,
        threshold: int = 20,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Find functions/methods exceeding line count threshold.

        Thresholds (Fowler recommendations):
        - 20 lines: Good target (extract anything longer)
        - 50 lines: Code smell
        - 100+ lines: Severe problem

        Args:
            instance_name: RETER instance name
            threshold: Maximum acceptable line count (default: 20)
            limit: Maximum results to return (default: 100)

        Returns:
            dict with success, long_functions list with severity, count, summary, queries
        """
        start_time = time.time()
        queries = []
        try:
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')

            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?func ?name ?qualified_name ?file ?start ?end ?type
                WHERE {{
                    ?func type ?type .
                    ?func name ?name .
                    ?func qualifiedName ?qualified_name .
                    ?func inFile ?file .
                    ?func hasLineStart ?start .
                    ?func hasLineEnd ?end .
                    FILTER(?type = {func_concept} || ?type = {method_concept})
                }}
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Calculate lines and filter by threshold
            long_functions = []
            for row in rows:
                start_line = int(row[4]) if row[4] else 0
                end_line = int(row[5]) if row[5] else 0
                lines = end_line - start_line + 1

                if lines > threshold:
                    # Calculate severity
                    if lines > 100:
                        severity = "CRITICAL"
                    elif lines > 50:
                        severity = "HIGH"
                    elif lines > 20:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    long_functions.append({
                        "qualified_name": row[2],
                        "name": row[1],
                        "file": row[3],
                        "lines": lines,
                        "start": start_line,
                        "end": end_line,
                        "type": row[6],
                        "severity": severity,
                        "suggested_refactoring": "Extract Function (106)",
                        "description": f"This {'method' if row[6] == 'py:Method' else 'function'} is {lines} lines long. Consider extracting logical blocks into smaller functions."
                    })

            # Sort by line count descending
            long_functions.sort(key=lambda x: x["lines"], reverse=True)

            # Apply limit
            long_functions = long_functions[:limit]

            # Calculate summary
            critical = sum(1 for f in long_functions if f["severity"] == "CRITICAL")
            high = sum(1 for f in long_functions if f["severity"] == "HIGH")
            medium = sum(1 for f in long_functions if f["severity"] == "MEDIUM")

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "long_functions": long_functions,
                "count": len(long_functions),
                "threshold": threshold,
                "summary": {
                    "critical": critical,
                    "high": high,
                    "medium": medium
                },
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "long_functions": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def detect_data_classes(
        self,
        instance_name: str,
        property_ratio: float = 0.8,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Detect data classes (classes with only getters/setters and no business logic).

        Detection approaches:
        1. Classes with @dataclass decorator (obvious)
        2. Classes where >80% of methods are @property or get_*/set_* pattern

        Args:
            instance_name: RETER instance name
            property_ratio: Minimum ratio of property methods (default: 0.8)
            limit: Maximum results to return (default: 100)

        Returns:
            dict with success, data_classes list, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Approach 1: Find @dataclass classes
            # Use inFile (works for all languages) instead of inModule
            dataclass_query = f"""
                SELECT ?class ?name ?file
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name .
                    ?class inFile ?file .
                    ?class hasDecorator "dataclass"
                }}
            """
            queries.append(dataclass_query.strip())
            dataclass_result = self.reter.reql(dataclass_query)
            dataclass_rows = self._query_to_list(dataclass_result)

            data_classes = []

            # Add @dataclass classes
            for row in dataclass_rows:
                data_classes.append({
                    "qualified_name": row[0],
                    "name": row[1],
                    "file": row[2],
                    "detection_method": "@dataclass decorator",
                    "severity": "MEDIUM",
                    "suggested_refactoring": "Move behavior from clients into this class (Move Function 198), or keep as immutable record",
                    "description": "This class uses @dataclass decorator, indicating it's primarily for data storage."
                })

            # Approach 2: Find classes with high property ratio
            # Get all classes with method counts
            # Use inFile (works for all languages) instead of inModule
            class_methods_query = f"""
                SELECT ?class ?name ?file (COUNT(?method) AS ?total_methods)
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?name .
                    ?class inFile ?file .
                    ?method type {method_concept} .
                    ?method definedIn ?class
                }}
                GROUP BY ?class ?name ?file
                HAVING (?total_methods > 0)
            """
            queries.append(class_methods_query.strip())
            class_methods_result = self.reter.reql(class_methods_query)
            class_methods_rows = self._query_to_list(class_methods_result)

            # BATCH: Get all property counts in one query (fixes N+1 problem)
            property_counts_query = """
                SELECT ?class (COUNT(?method) AS ?property_count)
                WHERE {
                    ?method definedIn ?class .
                    ?method hasDecorator "property"
                }
                GROUP BY ?class
            """
            queries.append(property_counts_query.strip())
            property_counts_result = self.reter.reql(property_counts_query)
            property_counts_rows = self._query_to_list(property_counts_result)

            # Build lookup: class_id -> property_count
            property_counts_map = {}
            for class_id_p, prop_count in property_counts_rows:
                property_counts_map[class_id_p] = int(prop_count) if prop_count else 0

            # Build set of @dataclass qualified names for fast lookup
            dataclass_qnames = {dc["qualified_name"] for dc in data_classes}

            # Process each class using the pre-fetched property counts
            for row in class_methods_rows:
                class_id = row[0]
                class_name = row[1]
                file = row[2]
                total_methods = int(row[3])

                # Skip if already detected as @dataclass
                if class_id in dataclass_qnames:
                    continue

                # Get property count from pre-fetched map
                property_count = property_counts_map.get(class_id, 0)

                # Check if property ratio exceeds threshold
                if total_methods > 0:
                    ratio = property_count / total_methods
                    if ratio >= property_ratio:
                        data_classes.append({
                            "qualified_name": class_id,
                            "name": class_name,
                            "file": file,
                            "total_methods": total_methods,
                            "property_methods": property_count,
                            "property_ratio": round(ratio, 2),
                            "detection_method": "high property ratio",
                            "severity": "MEDIUM",
                            "suggested_refactoring": "Encapsulate behavior (Combine Functions into Class 144)",
                            "description": f"This class has {property_count}/{total_methods} property methods ({int(ratio*100)}%), suggesting it's primarily a data holder."
                        })

            # Apply limit
            data_classes = data_classes[:limit]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "data_classes": data_classes,
                "count": len(data_classes),
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "data_classes": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def detect_feature_envy(
        self,
        instance_name: str,
        external_ratio: float = 1.5,
        min_external_calls: int = 3,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect methods with feature envy (methods that call other classes more than their own).

        A method has feature envy when:
        external_calls > internal_calls * ratio_threshold (e.g., 1.5x)

        Args:
            instance_name: RETER instance name
            external_ratio: Threshold for external/internal call ratio (default: 1.5)
            min_external_calls: Minimum external calls to flag (default: 3)
            limit: Maximum results to return (default: 20)
            offset: Skip this many results (default: 0)

        Returns:
            dict with success, envious_methods list, count, pagination info, sample queries
        """
        start_time = time.time()
        sample_queries = []
        try:
            # OPTIMIZED: Use batched queries instead of per-method queries
            # This reduces O(N*2) queries to O(3) queries
            method_concept = self._concept('Method')

            # Query 1: Get all methods with their classes
            methods_query = f"""
                SELECT ?method ?method_name ?own_class ?class_name
                WHERE {{
                    ?method type {method_concept} .
                    ?method name ?method_name .
                    ?method definedIn ?own_class .
                    ?own_class name ?class_name
                }}
            """
            sample_queries.append(methods_query.strip())
            methods_result = self.reter.reql(methods_query)
            methods_rows = self._query_to_list(methods_result)

            # Build method lookup: method_id -> (method_name, own_class, class_name)
            method_info = {}
            for row in methods_rows:
                method_id = row[0]
                method_info[method_id] = {
                    "method_name": row[1],
                    "own_class": row[2],
                    "class_name": row[3]
                }

            # Query 2: Get ALL external calls grouped by method (single batched query)
            external_query = f"""
                SELECT ?method (COUNT(?target) AS ?external_count)
                WHERE {{
                    ?method type {method_concept} .
                    ?method definedIn ?own_class .
                    ?method calls ?target .
                    ?target definedIn ?other_class .
                    FILTER(?other_class != ?own_class)
                }}
                GROUP BY ?method
            """
            sample_queries.append(external_query.strip())
            external_result = self.reter.reql(external_query)
            external_rows = self._query_to_list(external_result)

            # Build external calls lookup: method_id -> count
            external_calls_map = {}
            for row in external_rows:
                method_id = row[0]
                count = int(row[1]) if row[1] else 0
                external_calls_map[method_id] = count

            # Query 3: Get ALL internal calls grouped by method (single batched query)
            internal_query = f"""
                SELECT ?method (COUNT(?target) AS ?internal_count)
                WHERE {{
                    ?method type {method_concept} .
                    ?method definedIn ?own_class .
                    ?method calls ?target .
                    ?target definedIn ?own_class
                }}
                GROUP BY ?method
            """
            sample_queries.append(internal_query.strip())
            internal_result = self.reter.reql(internal_query)
            internal_rows = self._query_to_list(internal_result)

            # Build internal calls lookup: method_id -> count
            internal_calls_map = {}
            for row in internal_rows:
                method_id = row[0]
                count = int(row[1]) if row[1] else 0
                internal_calls_map[method_id] = count

            # Process all methods using the precomputed maps (in-memory, fast)
            envious_methods = []
            methods_checked = len(method_info)

            for method_id, info in method_info.items():
                external_calls = external_calls_map.get(method_id, 0)
                internal_calls = internal_calls_map.get(method_id, 0)

                # Check if method has feature envy
                if external_calls >= min_external_calls:
                    ratio = external_calls / max(internal_calls, 1)
                    if ratio >= external_ratio:
                        # Calculate severity
                        if ratio > 5:
                            severity = "HIGH"
                        elif ratio > 3:
                            severity = "MEDIUM"
                        else:
                            severity = "LOW"

                        envious_methods.append({
                            "qualified_name": method_id,
                            "method": f"{info['class_name']}.{info['method_name']}",
                            "class": info['class_name'],
                            "external_calls": external_calls,
                            "internal_calls": internal_calls,
                            "ratio": round(ratio, 2),
                            "severity": severity,
                            "suggested_refactoring": "Move Function (198) - Move method to the envied class",
                            "description": f"This method makes {external_calls} external calls but only {internal_calls} internal calls (ratio: {ratio:.1f}x)"
                        })

            # Sort by ratio descending
            envious_methods.sort(key=lambda x: x["ratio"], reverse=True)

            # Apply pagination
            total_count = len(envious_methods)
            paginated = envious_methods[offset:offset + limit]
            has_more = offset + limit < total_count

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "envious_methods": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "external_ratio_threshold": external_ratio,
                "methods_checked": methods_checked,
                "sample_queries": sample_queries,
                "time_ms": time_ms,
                "note": "Optimized: Uses 3 batched queries instead of per-method queries."
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "envious_methods": [],
                "count": 0,
                "total_count": 0,
                "sample_queries": sample_queries,
                "time_ms": time_ms
            }

    def detect_refused_bequest(
        self,
        instance_name: str,
        usage_threshold: float = 0.5,
        min_inherited_methods: int = 3,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Detect subclasses that don't use most of their inherited methods (refused bequest).

        Detection logic:
        used_inherited_methods < total_inherited_methods * threshold (e.g., 0.5)

        Args:
            instance_name: RETER instance name
            usage_threshold: Minimum usage ratio (default: 0.5 = 50%)
            min_inherited_methods: Only flag if parent has this many methods (default: 3)
            limit: Maximum results to return (default: 100)

        Returns:
            dict with success, refused_bequest_cases list, count, queries
        """
        start_time = time.time()
        queries = []
        try:
            # Get all inheritance relationships
            inheritance_query = """
                SELECT ?subclass ?subclass_name ?parent ?parent_name
                WHERE {
                    ?subclass inheritsFrom ?parent .
                    ?subclass name ?subclass_name .
                    ?parent name ?parent_name
                }
            """
            queries.append(inheritance_query.strip())
            inheritance_result = self.reter.reql(inheritance_query)
            inheritance_rows = self._query_to_list(inheritance_result)

            method_concept = self._concept('Method')
            # BATCH: Get method counts per class (fixes N+1 problem)
            method_counts_query = f"""
                SELECT ?class (COUNT(?method) AS ?method_count)
                WHERE {{
                    ?method definedIn ?class .
                    ?method type {method_concept}
                }}
                GROUP BY ?class
            """
            queries.append(method_counts_query.strip())
            method_counts_result = self.reter.reql(method_counts_query)
            method_counts_rows = self._query_to_list(method_counts_result)

            # Build lookup: class_id -> method_count
            method_counts_map = {}
            for class_id_m, count in method_counts_rows:
                method_counts_map[class_id_m] = int(count) if count else 0

            # BATCH: Get all calls from subclass methods to parent methods
            calls_query = """
                SELECT ?subclass ?parent (COUNT(DISTINCT ?used_method) AS ?used_count)
                WHERE {
                    ?caller definedIn ?subclass .
                    ?caller calls ?used_method .
                    ?used_method definedIn ?parent .
                    ?subclass inheritsFrom ?parent
                }
                GROUP BY ?subclass ?parent
            """
            queries.append(calls_query.strip())
            calls_result = self.reter.reql(calls_query)
            calls_rows = self._query_to_list(calls_result)

            # Build lookup: (subclass_id, parent_id) -> used_count
            used_counts_map = {}
            for subclass_id_c, parent_id_c, used_count in calls_rows:
                used_counts_map[(subclass_id_c, parent_id_c)] = int(used_count) if used_count else 0

            refused_cases = []

            for row in inheritance_rows:
                subclass_id = row[0]
                subclass_name = row[1]
                parent_id = row[2]
                parent_name = row[3]

                # Get inherited count from pre-fetched map
                inherited_count = method_counts_map.get(parent_id, 0)

                # Skip if parent doesn't have enough methods
                if inherited_count < min_inherited_methods:
                    continue

                # Get used count from pre-fetched map
                used_count = used_counts_map.get((subclass_id, parent_id), 0)

                # Check if usage is below threshold
                usage_ratio = used_count / inherited_count if inherited_count > 0 else 0

                if usage_ratio < usage_threshold:
                    # Calculate severity
                    if usage_ratio < 0.25:
                        severity = "HIGH"
                    elif usage_ratio < 0.5:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    refused_cases.append({
                        "subclass": subclass_name,
                        "subclass_qualified": subclass_id,
                        "parent": parent_name,
                        "parent_qualified": parent_id,
                        "inherited_methods": inherited_count,
                        "used_methods": used_count,
                        "usage_ratio": round(usage_ratio, 2),
                        "severity": severity,
                        "suggested_refactoring": "Replace Subclass with Delegate (381) or Replace Superclass with Delegate (399)",
                        "description": f"{subclass_name} uses only {used_count}/{inherited_count} methods from {parent_name} ({int(usage_ratio*100)}% usage)"
                    })

            # Sort by usage ratio ascending
            refused_cases.sort(key=lambda x: x["usage_ratio"])

            # Apply limit
            refused_cases = refused_cases[:limit]

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "refused_bequest_cases": refused_cases,
                "count": len(refused_cases),
                "usage_threshold": usage_threshold,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "refused_bequest_cases": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def detect_code_smells(
        self,
        instance_name: str,
        categories: list = None,
        severity_threshold: str = "MEDIUM",
        limit: int = 20,
        offset: int = 0,
        quick_mode: bool = False,
        limit_per_detector: int = None
    ) -> Dict[str, Any]:
        """
        Unified interface to detect all code smells in the analyzed codebase.

        Runs multiple smell detectors and provides a comprehensive report with
        refactoring suggestions.

        Args:
            instance_name: RETER instance name
            categories: Smell categories to check (default: ["all"])
                       Options: "all", "quick", "long_function", "data_class", "feature_envy",
                               "refused_bequest", "long_parameter_list", "large_class"
            severity_threshold: Minimum severity to report (default: "MEDIUM")
                              Options: "LOW", "MEDIUM", "HIGH", "CRITICAL"
            limit: Maximum results to return (default: 20, reduced from 50 for performance)
            offset: Number of results to skip (default: 0)
            quick_mode: If True, only run fast detectors (default: False)
            limit_per_detector: Max results per detector (default: auto-calculated from limit)

        Returns:
            dict with success, summary (before pagination), smells (paginated),
            refactoring priorities, and pagination metadata

        Performance Notes:
            - Use quick_mode=True for 3-5x faster results (runs only 3 fast detectors)
            - Reduce limit for faster queries (limit=10 is ~2x faster than limit=50)
            - Specify specific categories instead of "all" to skip unneeded detectors
        """
        start_time = time.time()

        if categories is None:
            categories = ["all"]

        # Severity level mapping
        severity_levels = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        min_severity = severity_levels.get(severity_threshold.upper(), 1)

        try:
            all_smells = []

            # Run detectors based on categories
            run_all = "all" in categories

            # Quick mode: only run fast detectors
            if quick_mode or "quick" in categories:
                run_all = False
                categories = ["long_function", "large_class", "long_parameter_list"]

            # Calculate limit per detector to avoid over-fetching
            if limit_per_detector is None:
                num_active_detectors = 6 if run_all else len(categories)
                # Give each detector a fraction of the total limit, with a minimum of 10
                limit_per_detector = max(10, (limit * 2) // num_active_detectors)

            # Track if we have enough high-priority results for early termination
            def should_continue():
                """Check if we should continue running detectors"""
                if not run_all:
                    return True  # Always run all requested categories
                critical_count = sum(1 for s in all_smells if s["severity"] == "CRITICAL")
                high_count = sum(1 for s in all_smells if s["severity"] == "HIGH")
                # Stop if we have enough critical/high priority results
                return (critical_count + high_count) < (limit * 2)

            # 1. Long Functions (Fast - simple line count query)
            if (run_all or "long_function" in categories) and should_continue():
                result = self.detect_long_functions(instance_name, threshold=20, limit=limit_per_detector)
                if result["success"]:
                    for func in result["long_functions"]:
                        if severity_levels.get(func["severity"], 0) >= min_severity:
                            all_smells.append({
                                "type": "long_function",
                                "location": f"{func['module']}:{func['name']}:{func['start']}-{func['end']}",
                                "severity": func["severity"],
                                "metrics": {"lines": func["lines"], "threshold": 20},
                                "suggested_refactoring": func["suggested_refactoring"],
                                "description": func["description"]
                            })

            # 2. Data Classes (Medium - requires method analysis)
            if (run_all or "data_class" in categories) and should_continue():
                result = self.detect_data_classes(instance_name, property_ratio=0.8, limit=limit_per_detector)
                if result["success"]:
                    for cls in result["data_classes"]:
                        if severity_levels.get(cls["severity"], 0) >= min_severity:
                            all_smells.append({
                                "type": "data_class",
                                "location": f"{cls['module']}:{cls['name']}",
                                "severity": cls["severity"],
                                "metrics": {"detection_method": cls["detection_method"]},
                                "suggested_refactoring": cls["suggested_refactoring"],
                                "description": cls["description"]
                            })

            # 3. Feature Envy (Slow - requires call graph analysis)
            if (run_all or "feature_envy" in categories) and should_continue():
                result = self.detect_feature_envy(instance_name, external_ratio=1.5, min_external_calls=3, limit=limit_per_detector)
                if result["success"]:
                    for method in result["envious_methods"]:
                        if severity_levels.get(method["severity"], 0) >= min_severity:
                            all_smells.append({
                                "type": "feature_envy",
                                "location": f"{method['class']}.{method['method']}",
                                "severity": method["severity"],
                                "metrics": {
                                    "external_calls": method["external_calls"],
                                    "internal_calls": method["internal_calls"],
                                    "ratio": method["ratio"]
                                },
                                "suggested_refactoring": method["suggested_refactoring"],
                                "description": method["description"]
                            })

            # 4. Refused Bequest (Slow - requires inheritance analysis)
            if (run_all or "refused_bequest" in categories) and should_continue():
                result = self.detect_refused_bequest(instance_name, usage_threshold=0.5, min_inherited_methods=3, limit=limit_per_detector)
                if result["success"]:
                    for case in result["refused_bequest_cases"]:
                        if severity_levels.get(case["severity"], 0) >= min_severity:
                            all_smells.append({
                                "type": "refused_bequest",
                                "location": f"{case['subclass']} -> {case['parent']}",
                                "severity": case["severity"],
                                "metrics": {
                                    "inherited_methods": case["inherited_methods"],
                                    "used_methods": case["used_methods"],
                                    "usage_ratio": case["usage_ratio"]
                                },
                                "suggested_refactoring": case["suggested_refactoring"],
                                "description": case["description"]
                            })

            # 5. Long Parameter Lists (Fast - simple parameter count query)
            if (run_all or "long_parameter_list" in categories) and should_continue():
                result = self.find_long_parameter_lists(instance_name, threshold=5)
                if result["success"]:
                    # Apply limit_per_detector to results
                    for func in result["functions"][:limit_per_detector]:
                        param_count = func["parameter_count"]
                        # Calculate severity
                        if param_count > 10:
                            severity = "CRITICAL"
                        elif param_count > 7:
                            severity = "HIGH"
                        elif param_count > 5:
                            severity = "MEDIUM"
                        else:
                            severity = "LOW"

                        if severity_levels.get(severity, 0) >= min_severity:
                            all_smells.append({
                                "type": "long_parameter_list",
                                "location": f"{func['name']}",
                                "severity": severity,
                                "metrics": {"parameters": param_count, "threshold": 5},
                                "suggested_refactoring": "Introduce Parameter Object (140)",
                                "description": f"This {'method' if func['type'] == 'py:Method' else 'function'} has {param_count} parameters"
                            })

            # 6. Large Classes (Fast - simple method count query)
            if (run_all or "large_class" in categories) and should_continue():
                result = self.find_large_classes(instance_name, threshold=20)
                if result["success"]:
                    # Apply limit_per_detector to results
                    for cls in result["classes"][:limit_per_detector]:
                        method_count = cls["method_count"]
                        # Calculate severity
                        if method_count > 50:
                            severity = "CRITICAL"
                        elif method_count > 30:
                            severity = "HIGH"
                        elif method_count >= 20:
                            severity = "MEDIUM"
                        else:
                            severity = "LOW"

                        if severity_levels.get(severity, 0) >= min_severity:
                            all_smells.append({
                                "type": "large_class",
                                "location": f"{cls['module']}:{cls['name']}",
                                "severity": severity,
                                "metrics": {"methods": method_count, "threshold": 20},
                                "suggested_refactoring": "Extract Class (182)",
                                "description": f"This class has {method_count} methods, indicating it may have too many responsibilities"
                            })

            # Sort by severity (CRITICAL > HIGH > MEDIUM > LOW) before pagination
            severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            all_smells.sort(key=lambda s: severity_order.get(s["severity"], 999))

            # Calculate summary statistics (before pagination)
            total_smells = len(all_smells)
            by_severity = {
                "CRITICAL": sum(1 for s in all_smells if s["severity"] == "CRITICAL"),
                "HIGH": sum(1 for s in all_smells if s["severity"] == "HIGH"),
                "MEDIUM": sum(1 for s in all_smells if s["severity"] == "MEDIUM"),
                "LOW": sum(1 for s in all_smells if s["severity"] == "LOW")
            }
            by_type = {}
            for smell in all_smells:
                smell_type = smell["type"]
                by_type[smell_type] = by_type.get(smell_type, 0) + 1

            # Generate refactoring priorities (from all smells, not paginated)
            priorities = []
            critical_smells = [s for s in all_smells if s["severity"] == "CRITICAL"]
            high_smells = [s for s in all_smells if s["severity"] == "HIGH"]

            for i, smell in enumerate(critical_smells[:5], 1):
                priorities.append(f"{i}. CRITICAL: {smell['type'].replace('_', ' ').title()} at {smell['location']}")

            priority_offset = len(priorities)
            for i, smell in enumerate(high_smells[:5], priority_offset + 1):
                priorities.append(f"{i}. HIGH: {smell['type'].replace('_', ' ').title()} at {smell['location']}")

            # Apply pagination to smells
            paginated_smells = all_smells[offset:offset + limit]
            count_returned = len(paginated_smells)

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "summary": {
                    "total_smells": total_smells,
                    "by_severity": by_severity,
                    "by_type": by_type
                },
                "smells": paginated_smells,
                "refactoring_priorities": priorities[:10],  # Top 10 priorities
                "total_count": total_smells,
                "count_returned": count_returned,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_smells,
                "next_offset": offset + limit if (offset + limit) < total_smells else None,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "summary": {},
                "smells": [],
                "refactoring_priorities": [],
                "time_ms": time_ms
            }

    # =========================================================================
    # REFACTORING OPPORTUNITY DETECTION (Chapter 6 - Fowler)
    # =========================================================================

    def find_data_clumps(
        self,
        instance_name: str,
        min_params: int = 3,
        min_functions: int = 2
    ) -> Dict[str, Any]:
        """
        Detect parameter groups that appear together in multiple functions.

        Data clumps suggest the need for Introduce Parameter Object refactoring.

        Args:
            instance_name: RETER instance name
            min_params: Minimum parameters in a clump (default: 3)
            min_functions: Minimum functions sharing parameters (default: 2)

        Returns:
            dict with success, data_clumps list, count, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_data_clumps(instance_name, min_params, min_functions)

    def find_function_groups(
        self,
        instance_name: str,
        min_shared_params: int = 2,
        min_functions: int = 3
    ) -> Dict[str, Any]:
        """
        Identify groups of functions operating on shared data.

        Function groups suggest the need for Combine Functions into Class refactoring.

        Args:
            instance_name: RETER instance name
            min_shared_params: Minimum shared parameters (default: 2)
            min_functions: Minimum functions in group (default: 3)

        Returns:
            dict with success, function_groups list, count, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_function_groups(instance_name, min_shared_params, min_functions)

    def find_extract_function_opportunities(
        self,
        instance_name: str,
        min_lines: int = 20,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Enhanced long function detection for Extract Function refactoring.

        Identifies functions that are candidates for Extract Function based on:
        - Line count exceeding threshold
        - Severity levels for prioritization

        Args:
            instance_name: RETER instance name
            min_lines: Minimum line count to flag (default: 20)
            limit: Maximum results to return (default: 100)
            offset: Skip this many results (default: 0)

        Returns:
            dict with success, opportunities list, count, pagination info, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_extract_function_opportunities(instance_name, min_lines, limit, offset)

    def find_inline_function_candidates(
        self,
        instance_name: str,
        max_lines: int = 5,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Detect trivial functions that are called from only one location.

        Inline Function (Fowler, Chapter 6):
        When a function body is as clear as its name, inline the function.

        Detection:
        - Function has ≤ max_lines (default: 5)
        - Function is called from exactly 1 location
        - Not a public API function (avoid breaking external callers)

        Args:
            instance_name: RETER instance name
            max_lines: Maximum lines to be considered trivial (default: 5)
            limit: Maximum results to return (default: 100)

        Returns:
            dict with success, candidates list, count, max_lines, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_inline_function_candidates(instance_name, max_lines, limit)

    def find_duplicate_parameter_lists(
        self,
        instance_name: str,
        min_params: int = 2,
        min_functions: int = 2,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Find functions with identical parameter signatures.

        Change Function Declaration (Fowler, Chapter 6):
        Functions with identical signatures may indicate need for interface extraction.

        Detection:
        - Functions have identical parameter lists (names and order)
        - At least min_params parameters
        - At least min_functions functions share the signature

        Args:
            instance_name: RETER instance name
            min_params: Minimum parameters required (default: 2)
            min_functions: Minimum functions sharing signature (default: 2)
            limit: Maximum results to return (default: 100)

        Returns:
            dict with success, duplicates list, count, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_duplicate_parameter_lists(instance_name, min_params, min_functions, limit)

    def find_unused_parameters(
        self,
        instance_name: str,
        include_self: bool = False
    ) -> Dict[str, Any]:
        """
        Find parameters that are declared but never used in function bodies.

        Based on Martin Fowler's "Remove Dead Code" refactoring pattern.
        A parameter is unused if it has hasParameter relationship with a function
        but does NOT have usesParameter relationship.

        Args:
            instance_name: RETER instance name
            include_self: Whether to include unused 'self' parameters (default: False)

        Returns:
            dict with success, unused_parameters list, count, functions_affected, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_unused_parameters(instance_name, include_self)

    def find_shotgun_surgery(
        self,
        instance_name: str,
        min_callers: int = 5,
        min_modules: int = 3
    ) -> Dict[str, Any]:
        """
        Detect functions/classes with high fan-in from many modules.

        Shotgun Surgery (Fowler, Chapter 3) occurs when a single change
        requires modifications in many different places. This detector finds
        code that is called from many different locations.

        Args:
            instance_name: RETER instance name
            min_callers: Minimum number of callers to flag (default: 5)
            min_modules: Minimum number of calling modules (default: 3)

        Returns:
            dict with success, shotgun_surgery_cases list, count, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_shotgun_surgery(instance_name, min_callers, min_modules)

    def find_middle_man(
        self,
        instance_name: str,
        max_lines: int = 10,
        min_delegation_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect classes/methods that just delegate to other classes.

        Middle Man (Fowler, Chapter 3) occurs when a class or method does
        little more than delegate to another class.

        Args:
            instance_name: RETER instance name
            max_lines: Maximum lines for a method to be considered (default: 10)
            min_delegation_ratio: Minimum ratio of delegating calls (default: 0.5)

        Returns:
            dict with success, middle_man_cases list, count, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_middle_man(instance_name, max_lines, min_delegation_ratio)

    def find_extract_class_opportunities(
        self,
        instance_name: str,
        min_methods: int = 10,
        min_cohesion_gap: float = 0.3
    ) -> Dict[str, Any]:
        """
        Detect classes that should be split into multiple classes.

        Extract Class (Fowler, Chapter 7) is needed when a class is doing
        the work of two or more classes.

        Args:
            instance_name: RETER instance name
            min_methods: Minimum methods to consider (default: 10)
            min_cohesion_gap: Minimum cohesion difference to split (default: 0.3)

        Returns:
            dict with success, extract_class_opportunities list, count, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_extract_class_opportunities(instance_name, min_methods, min_cohesion_gap)

    def analyze_refactoring_opportunities(
        self,
        instance_name: str,
        severity_threshold: str = "MEDIUM",
        limit_per_type: int = 5,
        limit: int = 20,
        offset: int = 0,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Unified analysis of ALL refactoring opportunities across detectors.

        Runs all refactoring detectors and provides prioritized dashboard.

        Args:
            instance_name: RETER instance name
            severity_threshold: Minimum severity to include (LOW, MEDIUM, HIGH, CRITICAL)
            limit_per_type: Maximum results per refactoring type (default: 5, reduced from 10)
            limit: Maximum number of results to return (default: 20, reduced from 50)
            offset: Number of results to skip (default: 0)
            quick_mode: If True, run only fast detectors (default: False)

        Returns:
            dict with success, summary, opportunities list (paginated), total_count, pagination metadata, time_ms

        Performance Notes:
            - Use quick_mode=True for 2-3x faster results (runs only 2 fast detectors)
            - Reduce limit for faster queries (limit=10 is ~2x faster than limit=50)
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.analyze_refactoring_opportunities(
            instance_name, severity_threshold, limit_per_type, limit, offset, quick_mode
        )

    def find_inline_class_opportunities(
        self,
        instance_name: str,
        max_methods: int = 3,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect small, trivial classes that should be inlined into their clients.

        Inline Class (Fowler, Chapter 7) is the inverse of Extract Class.
        Use when a class is no longer pulling its weight and shouldn't be around anymore.

        Detection signals:
        - Class has <= max_methods methods (default: 3)
        - Class is used by only 1 or 2 client classes
        - Class is a data class (only getters/setters, no business logic)
        - Class has no subclasses (can't inline if subclassed)

        Args:
            instance_name: RETER instance name
            max_methods: Maximum method count for small class (default: 3)
            limit: Maximum results to return (pagination, default: 50)
            offset: Starting offset for pagination (default: 0)

        Returns:
            dict with success, inline_class_opportunities list, count, pagination metadata, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_inline_class_opportunities(instance_name, max_methods, limit, offset)

    def find_primitive_obsession(
        self,
        instance_name: str,
        min_usages: int = 5,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect primitive parameters (str, int, float) that should be value objects.

        Replace Primitive with Object (Fowler, Chapter 7) is used when simple
        primitives aren't so simple anymore and need behavior, validation, etc.

        Detection signals:
        - Same primitive parameter name appears in many functions
        - High fan-out (>= min_usages functions)
        - Parameter name suggests domain concept (email, phone, money, etc.)

        This detects "Primitive Obsession" code smell - when primitives are used
        instead of small objects for simple tasks (phone numbers, ZIP codes, money, etc.).

        Args:
            instance_name: RETER instance name
            min_usages: Minimum function count to flag (default: 5)
            limit: Maximum results to return (pagination, default: 50)
            offset: Starting offset for pagination (default: 0)

        Returns:
            dict with success, primitive_obsession_cases list, count, pagination metadata, queries, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_primitive_obsession(instance_name, min_usages, limit, offset)

    def find_encapsulate_collection_opportunities(
        self,
        instance_name: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect methods that return mutable collections without encapsulation.

        Fowler Chapter 7: Encapsulate Collection (heuristic approach with type hints)

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return (pagination, default: 50)
            offset: Starting offset for pagination (default: 0)

        Returns:
            dict with success, encapsulate_collection_opportunities list, count, pagination metadata, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_encapsulate_collection_opportunities(instance_name, limit, offset)

    def find_encapsulate_field_opportunities(
        self,
        instance_name: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find public attributes that should be private with getters/setters.

        Fowler Chapter 7: Encapsulate Field

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return (pagination, default: 50)
            offset: Starting offset for pagination (default: 0)

        Returns:
            dict with success, encapsulate_field_opportunities list, count, pagination metadata, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_encapsulate_field_opportunities(instance_name, limit, offset)

    def find_attribute_data_clumps(
        self,
        instance_name: str,
        min_attrs: int = 3,
        min_classes: int = 2,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect groups of attributes that appear together in multiple classes.

        Suggests Introduce Value Object refactoring when multiple classes
        share the same set of attributes.

        Args:
            instance_name: RETER instance name
            min_attrs: Minimum attributes in a clump (default: 3)
            min_classes: Minimum classes sharing attributes (default: 2)
            limit: Maximum results to return (pagination, default: 100)
            offset: Starting offset for pagination (default: 0)

        Returns:
            dict with success, attribute_data_clumps list, count, pagination metadata, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_attribute_data_clumps(instance_name, min_attrs, min_classes, limit, offset)

    def find_hide_delegate_opportunities(
        self,
        instance_name: str,
        min_client_calls: int = 3,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect classes that should add delegating methods (inverse of Middle Man).

        Fowler Chapter 7: Hide Delegate (pragmatic heuristic approach)

        Args:
            instance_name: RETER instance name
            min_client_calls: Minimum client calls to flag (default: 3)
            limit: Maximum results to return (pagination, default: 50)
            offset: Starting offset for pagination (default: 0)

        Returns:
            dict with success, hide_delegate_opportunities list, count, pagination metadata, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_hide_delegate_opportunities(instance_name, min_client_calls, limit, offset)

    def find_encapsulate_record_opportunities(
        self,
        instance_name: str,
        min_accesses: int = 5,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find dict/record usage that should be encapsulated in a class.

        Fowler Chapter 7: Encapsulate Record

        Args:
            instance_name: RETER instance name
            min_accesses: Minimum number of dict accesses to flag (default: 5)
            limit: Maximum results to return (pagination, default: 100)
            offset: Starting offset for pagination (default: 0)

        Returns:
            dict with success, encapsulate_record_opportunities list, count, pagination metadata, time_ms
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_encapsulate_record_opportunities(instance_name, min_accesses, limit, offset)

    def find_split_loop_opportunities(
        self,
        instance_name: str,
        min_operations: int = 2,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find loops doing multiple things (Fowler Chapter 8: Split Loop)."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_split_loop_opportunities(instance_name, min_operations, limit, offset)

    def find_pipeline_conversion_opportunities(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find loops replaceable with pipelines (Fowler Chapter 8)."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_pipeline_conversion_opportunities(instance_name, limit, offset)

    def find_move_function_opportunities(
        self,
        instance_name: str,
        coupling_threshold: float = 0.5,
        min_external_refs: int = 5,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find functions that should move to a different class (Fowler Chapter 8: Move Function)."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_move_function_opportunities(
            instance_name, coupling_threshold, min_external_refs, limit, offset
        )

    def find_move_field_opportunities(
        self,
        instance_name: str,
        access_ratio_threshold: float = 0.6,
        min_external_accesses: int = 3,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find fields that should move to a different class (Fowler Chapter 8: Move Field)."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_move_field_opportunities(
            instance_name, access_ratio_threshold, min_external_accesses, limit, offset
        )

    # =========================================================================
    # CHAPTER 9: ORGANIZING DATA REFACTORINGS
    # =========================================================================

    def find_split_variable_opportunities(
        self,
        instance_name: str,
        min_assignments: int = 2,
        include_loop_vars: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find variables assigned multiple times for different purposes (Fowler Chapter 9: Split Variable).

        Variables should have a single responsibility. If a variable is assigned multiple times
        (not as a loop variable or accumulator), it likely has multiple responsibilities and
        should be split into separate variables.

        Args:
            instance_name: RETER instance name
            min_assignments: Minimum number of assignments to flag (default: 2)
            include_loop_vars: Whether to include loop variables (default: False)
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with split variable opportunities including function names, variable names,
            assignment counts, locations, severity, and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_split_variable_opportunities(
            instance_name, min_assignments, include_loop_vars, limit, offset
        )

    # =========================================================================
    # CHAPTER 3: CODE SMELLS (Additional Detectors)
    # =========================================================================

    def find_message_chains(
        self,
        instance_name: str,
        min_chain_length: int = 3,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find long method call chains (Fowler Chapter 3: Message Chains).

        Message chains like a.b().c().d() create coupling and violate the Law of Demeter.

        Args:
            instance_name: RETER instance name
            min_chain_length: Minimum chain depth to flag (default: 3)
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with message chains including chain length, functions involved, severity,
            and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_message_chains(instance_name, min_chain_length, limit, offset)

    def find_global_data(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find module-level mutable assignments (Fowler Chapter 3: Global Data).

        Global mutable data creates hidden coupling and makes testing difficult.

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with global data items including module, variable name, severity,
            and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_global_data(instance_name, limit, offset)

    def find_speculative_generality(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find abstract classes with only 1 subclass (Fowler Chapter 3: Speculative Generality).

        Abstractions created "just in case" without current need indicate over-engineering.

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with speculative generality cases including parent/child classes, severity,
            and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_speculative_generality(instance_name, limit, offset)

    def find_parallel_inheritance_hierarchies(
        self,
        instance_name: str,
        min_similarity: float = 0.6,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find class hierarchies that mirror each other (Fowler Chapter 3).

        Parallel hierarchies like OrderProcessor/OrderValidator grow together
        and should be consolidated.

        Args:
            instance_name: RETER instance name
            min_similarity: Name similarity threshold (0-1, default: 0.6)
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with parallel hierarchy pairs including similarity scores, matching subclasses,
            severity, and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_parallel_inheritance_hierarchies(
            instance_name, min_similarity, limit, offset
        )

    def find_mutable_data_across_functions(
        self,
        instance_name: str,
        min_functions: int = 3,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find variables assigned in multiple different functions (Fowler Chapter 3: Mutable Data).

        Shared mutable state across functions suggests poor encapsulation.

        Args:
            instance_name: RETER instance name
            min_functions: Minimum functions assigning the variable (default: 3)
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with mutable data cases including variable name, affected functions, severity,
            and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_mutable_data_across_functions(
            instance_name, min_functions, limit, offset
        )

    def find_alternative_classes_with_different_interfaces(
        self,
        instance_name: str,
        min_method_similarity: float = 0.5,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find classes with similar responsibilities but different interfaces (Fowler Chapter 3).

        Classes doing similar things with different method names create confusion.

        Args:
            instance_name: RETER instance name
            min_method_similarity: Method count similarity threshold (0-1, default: 0.5)
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with alternative class pairs including method counts, overlaps, severity,
            and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_alternative_classes_with_different_interfaces(
            instance_name, min_method_similarity, limit, offset
        )

    def find_flag_arguments(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find boolean parameters (flag arguments) that control function behavior.

        Fowler Chapter 11: Remove Flag Argument

        Flag arguments are boolean parameters that control function behavior,
        often leading to if/else branches. They should be replaced with separate functions.

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with flag arguments including parameter names, type hints, functions,
            detection reasons, severity, and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_flag_arguments(instance_name, limit, offset)

    def find_setting_methods(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find setter methods that could be removed for immutability.

        Fowler Chapter 11: Remove Setting Method

        Setter methods allow mutation of object state after construction.
        Removing them and setting values only in constructor improves immutability.

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with setting methods including method names, classes, parameter counts,
            naming styles, severity, and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_setting_methods(instance_name, limit, offset)

    def find_trivial_commands(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find trivial command objects that should be simple functions.

        Fowler Chapter 11: Replace Command with Function

        Command objects with only one method (like execute/run/handle) are often
        over-engineered and should just be functions.

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with trivial commands including class names, command methods,
            method counts, pattern confidence, severity, and refactoring suggestions.
        """
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_trivial_commands(instance_name, limit, offset)

    # =========================================================================
    # CHAPTER 12: Dealing with Inheritance Refactoring Tools
    # =========================================================================

    def find_pull_up_method_candidates(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find duplicate methods in sibling classes - Fowler Chapter 12."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_pull_up_method_candidates(instance_name, limit, offset)

    def find_push_down_method_candidates(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find superclass methods only used by some subclasses - Fowler Chapter 12."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_push_down_method_candidates(instance_name, limit, offset)

    def find_remove_subclass_candidates(
        self,
        instance_name: str,
        max_methods: int = 2,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find trivial subclasses that should be removed - Fowler Chapter 12."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_remove_subclass_candidates(instance_name, max_methods, limit, offset)

    def find_extract_superclass_candidates(
        self,
        instance_name: str,
        min_shared_methods: int = 2,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find classes with similar methods that should share superclass - Fowler Chapter 12."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_extract_superclass_candidates(instance_name, min_shared_methods, limit, offset)

    def find_collapse_hierarchy_candidates(
        self,
        instance_name: str,
        max_additional_methods: int = 2,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find nearly identical parent-child pairs that should be merged - Fowler Chapter 12."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_collapse_hierarchy_candidates(instance_name, max_additional_methods, limit, offset)

    def find_replace_with_delegate_candidates(
        self,
        instance_name: str,
        max_coupling_ratio: float = 0.3,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find inheritance with low coupling (refused bequest) - Fowler Chapter 12."""
        from .refactoring_detector import RefactoringOpportunityDetector
        detector = RefactoringOpportunityDetector(self.reter)
        return detector.find_replace_with_delegate_candidates(instance_name, max_coupling_ratio, limit, offset)

    # =========================================================================
    # EXCEPTION HANDLING DETECTORS (delegated to ExceptionAnalysisTools)
    # =========================================================================

    def detect_silent_exception_swallowing(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find except blocks that silently swallow exceptions. Delegates to ExceptionAnalysisTools."""
        return self._exception_analysis.detect_silent_exception_swallowing(instance_name, limit, offset)

    def detect_too_general_exceptions(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find except blocks that catch overly broad exceptions. Delegates to ExceptionAnalysisTools."""
        return self._exception_analysis.detect_too_general_exceptions(instance_name, limit, offset)

    def detect_general_exception_raising(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find raise statements with overly general exceptions. Delegates to ExceptionAnalysisTools."""
        return self._exception_analysis.detect_general_exception_raising(instance_name, limit, offset)

    def detect_error_codes_over_exceptions(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find functions returning error codes instead of exceptions. Delegates to ExceptionAnalysisTools."""
        return self._exception_analysis.detect_error_codes_over_exceptions(instance_name, limit, offset)

    def detect_finally_without_context_manager(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Find try/finally blocks that should use context managers. Delegates to ExceptionAnalysisTools."""
        return self._exception_analysis.detect_finally_without_context_manager(instance_name, limit, offset)

    def analyze_exception_handling(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Comprehensive exception handling analysis. Delegates to ExceptionAnalysisTools."""
        return self._exception_analysis.analyze_exception_handling(instance_name, limit, offset)
