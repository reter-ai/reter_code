"""
Function Analysis Tools

Provides tools for analyzing functions for refactoring opportunities:
- Function groups (candidates for Combine Functions into Class)
- Extract Function opportunities (long functions)
- Inline Function candidates (trivial functions)
- Duplicate parameter signatures

Based on Martin Fowler's "Refactoring" Chapter 6.
"""

from typing import Dict, Any, List, Set
from collections import defaultdict
import time
from .base import AdvancedToolsBase


class FunctionAnalysisTools(AdvancedToolsBase):
    """
    Function analysis tools for refactoring opportunities.

    Detects:
    - Function groups that should become classes
    - Long functions needing extraction
    - Trivial functions to inline
    - Duplicate parameter signatures
    """

    def find_function_groups(
        self,
        instance_name: str,
        min_shared_params: int = 2,
        min_functions: int = 3
    ) -> Dict[str, Any]:
        """
        Identify groups of functions operating on shared data.

        Function groups are sets of functions that:
        1. Share common parameters (operate on same data)
        2. May call each other (collaboration)
        3. Are candidates for Combine Functions into Class

        Example:
            calculateBaseCharge(reading), calculateTaxableCharge(reading),
            and printInvoice(reading) all work with the same 'reading' data
            -> Suggest ReadingProcessor class

        Args:
            instance_name: RETER instance name
            min_shared_params: Minimum shared parameters (default: 2)
            min_functions: Minimum functions in group (default: 3)

        Returns:
            dict with:
                success: bool
                function_groups: List of function groups
                count: Number of groups found
                queries: REQL queries executed
                time_ms: Execution time
        """
        start_time = time.time()
        queries = []

        try:
            param_concept = self._concept('Parameter')
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')
            class_concept = self._concept('Class')

            # Query 1: Get all function-parameter relationships
            # Use inFile (works for all languages) instead of inModule
            param_query = f"""
                SELECT ?func ?func_name ?file ?param_name
                WHERE {{
                    ?func type ?type .
                    ?func name ?func_name .
                    ?func inFile ?file .
                    ?param type {param_concept} .
                    ?param ofFunction ?func .
                    ?param name ?param_name .
                    FILTER(?type = {func_concept} || ?type = {method_concept})
                }}
                ORDER BY ?func
            """
            queries.append(param_query.strip())

            result = self.reter.reql(param_query)
            rows = self._query_to_list(result)

            # Build function info and parameter mappings
            func_info: Dict[str, Dict[str, Any]] = {}
            param_to_funcs: Dict[str, Set[str]] = defaultdict(set)

            for func_id, func_name, file, param_name in rows:
                if func_id not in func_info:
                    func_info[func_id] = {
                        "name": func_name,
                        "file": file,
                        "params": set()
                    }
                func_info[func_id]["params"].add(param_name)
                param_to_funcs[param_name].add(func_id)

            # Query 2: Get call relationships
            call_query = f"""
                SELECT ?caller ?callee
                WHERE {{
                    ?caller type ?type1 .
                    ?caller calls ?callee .
                    ?callee type ?type2 .
                    FILTER(?type1 = "{func_concept}" || ?type1 = "{method_concept}")
                    FILTER(?type2 = "{func_concept}" || ?type2 = "{method_concept}")
                }}
            """
            queries.append(call_query.strip())

            result = self.reter.reql(call_query)
            call_rows = self._query_to_list(result)

            # Build call graph
            call_graph: Dict[str, Set[str]] = defaultdict(set)
            for caller, callee in call_rows:
                call_graph[caller].add(callee)
                # Add reverse edge for bidirectional grouping
                call_graph[callee].add(caller)

            # BATCH: Pre-fetch all function-to-class mappings (fixes N+1 problem)
            class_query = f"""
            SELECT ?func ?class
            WHERE {{
                ?func definedIn ?class .
                ?class type {class_concept}
            }}
            """
            queries.append(class_query.strip())
            class_result = self.reter.reql(class_query)
            class_rows = self._query_to_list(class_result)

            # Build lookup: func_id -> class_id
            func_to_class: Dict[str, str] = {}
            for func_id, class_id in class_rows:
                func_to_class[func_id] = class_id

            # Find function groups: functions sharing parameters
            # Group by shared parameter patterns
            param_pattern_to_funcs: Dict[frozenset, List[str]] = defaultdict(list)

            for func_id, info in func_info.items():
                # Create a pattern from sorted parameter names
                if len(info["params"]) > 0:
                    pattern = frozenset(info["params"])
                    param_pattern_to_funcs[pattern].append(func_id)

            # Find groups with sufficient shared parameters
            function_groups = []
            processed_funcs = set()

            for pattern, func_ids in param_pattern_to_funcs.items():
                # Skip if we've already processed these functions
                if any(fid in processed_funcs for fid in func_ids):
                    continue

                if len(func_ids) >= min_functions and len(pattern) >= min_shared_params:
                    # Get call relationships within group
                    call_relationships = []
                    for func_id in func_ids:
                        for callee in call_graph.get(func_id, []):
                            if callee in func_ids:
                                caller_name = func_info[func_id]["name"]
                                callee_name = func_info[callee]["name"]
                                call_relationships.append(f"{caller_name} -> {callee_name}")

                    # Calculate severity
                    func_count = len(func_ids)
                    param_count = len(pattern)
                    has_calls = len(call_relationships) > 0

                    if func_count >= 5 or (param_count >= 3 and has_calls):
                        severity = "HIGH"
                    elif func_count >= 3 or param_count >= 2:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    # Get function details
                    functions = [
                        {
                            "qualified_name": func_id,
                            "name": func_info[func_id]["name"],
                            "module": func_info[func_id]["module"]
                        }
                        for func_id in func_ids
                    ]

                    # Suggest class name
                    func_names = [f["name"] for f in functions]
                    suggested_class = self._suggest_class_name(func_names, list(pattern))

                    # Check if functions are already in the same class (false positive)
                    # Uses pre-fetched func_to_class mapping (avoids N+1 query)
                    classes_for_funcs = [func_to_class.get(fid) for fid in func_ids]
                    classes_in_group = set(c for c in classes_for_funcs if c is not None)
                    # All functions must be methods (have a class) and all in the same class
                    if len(classes_in_group) == 1 and all(c is not None for c in classes_for_funcs):
                        continue  # Skip - all functions are already in the same class

                    function_groups.append({
                        "functions": functions,
                        "function_count": func_count,
                        "shared_parameters": sorted(pattern),
                        "parameter_count": param_count,
                        "call_relationships": call_relationships,
                        "has_collaboration": has_calls,
                        "severity": severity,
                        "refactoring": "Combine Functions into Class",
                        "suggestion": f"Create {suggested_class} class with methods: {', '.join([f['name'] for f in functions[:3]])}{'...' if len(functions) > 3 else ''}",
                        "estimated_effort": "moderate" if func_count <= 5 else "high"
                    })

                    # Mark as processed
                    processed_funcs.update(func_ids)

            # Sort by severity and function count
            severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            function_groups.sort(
                key=lambda x: (severity_order[x["severity"]], -x["function_count"])
            )

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "function_groups": function_groups,
                "count": len(function_groups),
                "min_shared_params": min_shared_params,
                "min_functions": min_functions,
                "queries": queries,
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "function_groups": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def _suggest_class_name(
        self,
        func_names: List[str],
        params: List[str]
    ) -> str:
        """
        Suggest a class name based on function names and shared parameters.

        Args:
            func_names: List of function names in the group
            params: List of shared parameter names

        Returns:
            Suggested class name
        """
        # Strategy 1: Look for common prefix in function names
        if len(func_names) >= 2:
            prefix = self._common_prefix(func_names)
            if len(prefix) > 2:
                # Remove common action verbs
                prefix_clean = prefix.rstrip('_')
                if prefix_clean:
                    return prefix_clean.capitalize() + "Processor"

        # Strategy 2: Use dominant parameter name
        domain_terms = {'customer', 'order', 'product', 'invoice', 'user', 'reading', 'account'}
        for param in params:
            param_lower = param.lower()
            if param_lower in domain_terms:
                return param.capitalize() + "Processor"

        # Strategy 3: Use first parameter
        if params:
            return params[0].capitalize() + "Handler"

        # Fallback
        return "ServiceClass"

    def _already_in_same_class(self, function_ids: List[str]) -> bool:
        """
        Check if all functions are already methods of the same class.

        This helps eliminate false positives where functions are already
        properly encapsulated in a class (no refactoring needed).

        Args:
            function_ids: List of function/method qualified names

        Returns:
            True if all functions are in the same class, False otherwise
        """
        try:
            if not function_ids:
                return False

            # Single query to get all function-to-class mappings
            class_concept = self._concept('Class')
            class_query = f"""
            SELECT ?func ?class
            WHERE {{
                ?func definedIn ?class .
                ?class type {class_concept} .
                ?func qualifiedName ?funcName .
            }}
            """
            result = self.reter.reql(class_query)
            all_mappings = self._query_to_list(result)

            # Build a dict of func_id -> class
            func_to_class = {row[0]: row[1] for row in all_mappings}

            # Check all function_ids
            classes = set()
            for func_id in function_ids:
                if func_id in func_to_class:
                    classes.add(func_to_class[func_id])
                else:
                    # Function not in a class (standalone function)
                    return False

            # All in same class = already combined
            return len(classes) == 1

        except (KeyError, IndexError, TypeError, RuntimeError):
            # KeyError: Missing function ID
            # IndexError: Empty query results
            # TypeError: Invalid data types
            # RuntimeError: REQL query errors
            return False

    def _common_prefix(self, strings: List[str]) -> str:
        """Find common prefix of strings."""
        if not strings:
            return ""

        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix

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
        - High complexity (if available)
        - Presence of comments indicating sections

        Args:
            instance_name: RETER instance name
            min_lines: Minimum line count to flag (default: 20)
            limit: Maximum results to return (default: 100)
            offset: Skip this many results (default: 0)

        Returns:
            dict with:
                success: bool
                opportunities: List of extraction opportunities
                count: Number of opportunities returned
                total_count: Total opportunities found
                has_more: Whether more results are available
                pagination: Pagination info
                queries: REQL queries executed
                time_ms: Execution time
        """
        start_time = time.time()
        queries = []

        try:
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')
            # Query functions with line counts
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?func ?name ?file ?line_count
                WHERE {{
                    ?func type ?type .
                    ?func name ?name .
                    ?func inFile ?file .
                    ?func lineCount ?line_count .
                    FILTER(?type = {func_concept} || ?type = {method_concept})
                    FILTER(?line_count >= {min_lines})
                }}
                ORDER BY DESC(?line_count)
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            opportunities = []

            for func_id, name, file, line_count in rows:
                # Convert line_count to integer (comes from REQL as string)
                line_count = int(line_count)

                # Calculate severity based on line count
                if line_count >= 100:
                    severity = "CRITICAL"
                    effort = "high"
                elif line_count >= 50:
                    severity = "HIGH"
                    effort = "moderate"
                elif line_count >= 30:
                    severity = "MEDIUM"
                    effort = "moderate"
                else:
                    severity = "LOW"
                    effort = "simple"

                opportunities.append({
                    "function": func_id,
                    "name": name,
                    "file": file,
                    "line_count": int(line_count) if line_count else 0,
                    "severity": severity,
                    "refactoring": "Extract Function",
                    "suggestion": f"Function '{name}' has {line_count} lines. Consider extracting logical sections into separate functions.",
                    "estimated_effort": effort,
                    "type": "Extract Function"
                })

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = offset + limit < total_count

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "opportunities": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "min_lines": min_lines,
                "queries": queries,
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "opportunities": [],
                "count": 0,
                "total_count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

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
        - Function has <= max_lines (default: 5)
        - Function is called from exactly 1 location
        - Not a public API function (avoid breaking external callers)

        Args:
            instance_name: RETER instance name
            max_lines: Maximum lines to be considered trivial (default: 5)
            limit: Maximum results to return (default: 100)

        Returns:
            success: Whether analysis succeeded
            candidates: List of inline function candidates
            count: Number of candidates found
            max_lines: Threshold used
            queries: REQL queries executed
            time_ms: Execution time
        """
        start_time = time.time()
        queries = []

        try:
            func_concept = self._concept('Function')
            # Query 1: Find all functions with line count <= max_lines
            # Use inFile (works for all languages) instead of inModule
            query1 = f"""
            SELECT ?func ?name ?file ?line_count ?line
            WHERE {{
                ?func type {func_concept} .
                ?func name ?name .
                ?func inFile ?file .
                ?func atLine ?line .
                ?func lineCount ?line_count .
                FILTER(?line_count <= {max_lines})
            }}
            """
            queries.append(("trivial_functions", query1))
            result1 = self.reter.reql(query1)
            trivial_functions = self._query_to_list(result1)

            # Query 2: Find all call relationships with caller names
            query2 = """
            SELECT ?caller ?callee ?caller_name
            WHERE {
                ?caller calls ?callee .
                ?caller name ?caller_name .
            }
            """
            queries.append(("call_graph", query2))
            result2 = self.reter.reql(query2)
            call_graph = self._query_to_list(result2)

            # Count callers for each function and build caller->name mapping
            caller_count = {}
            callee_to_caller_name = {}  # Maps callee -> caller_name for single-caller functions
            for caller, callee, caller_name in call_graph:
                if callee not in caller_count:
                    caller_count[callee] = 0
                    callee_to_caller_name[callee] = caller_name
                caller_count[callee] += 1

            # Check if functions are public API (not starting with _)
            public_functions = set()
            for func_id, name, file, line_count, line in trivial_functions:
                if not name.startswith('_'):
                    public_functions.add(func_id)

            # Find candidates: trivial functions called exactly once, not public API
            candidates = []
            for func_id, name, file, line_count, line in trivial_functions[:limit]:
                count = caller_count.get(func_id, 0)

                if count == 1 and func_id not in public_functions:
                    # Get caller name from pre-built mapping (no extra query needed)
                    caller_name = callee_to_caller_name.get(func_id, "unknown")

                    severity = "LOW" if line_count <= 3 else "MEDIUM"
                    effort = "simple"

                    candidates.append({
                        "function": func_id,
                        "name": name,
                        "file": file,
                        "line": int(line) if line else 0,
                        "line_count": int(line_count) if line_count else 0,
                        "caller": caller_name,
                        "severity": severity,
                        "refactoring": "Inline Function",
                        "suggestion": f"Function '{name}' ({line_count} lines) is only called once. Consider inlining into '{caller_name}'.",
                        "estimated_effort": effort
                    })

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "candidates": candidates,
                "count": len(candidates),
                "max_lines": max_lines,
                "queries": queries,
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "candidates": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def _suggest_interface_name(self, function_names: list) -> str:
        """
        Suggest an interface name based on function names.

        Args:
            function_names: List of function names with common signatures

        Returns:
            Suggested interface name
        """
        if not function_names:
            return "CommonInterface"

        # Find common prefix
        prefix = self._common_prefix(function_names)
        if prefix and len(prefix) >= 3:
            # Remove trailing underscore
            prefix = prefix.rstrip('_')
            return f"{prefix}Interface"

        # Try to find common suffix
        reversed_names = [name[::-1] for name in function_names]
        suffix = self._common_prefix(reversed_names)[::-1]
        if suffix and len(suffix) >= 3:
            suffix = suffix.lstrip('_')
            return f"{suffix}Interface"

        # Fallback: use first function name + Interface
        first_name = function_names[0].replace('_', ' ').title().replace(' ', '')
        return f"{first_name}Interface"

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
            success: Whether analysis succeeded
            duplicates: List of duplicate parameter lists
            count: Number of duplicates found
            min_params: Minimum parameters threshold
            min_functions: Minimum functions threshold
            queries: REQL queries executed
            time_ms: Execution time
        """
        start_time = time.time()
        queries = []

        try:
            func_concept = self._concept('Function')
            param_concept = self._concept('Parameter')
            # Query 1: Get all function parameters
            query1 = f"""
            SELECT ?func ?func_name ?param ?param_name ?param_index
            WHERE {{
                ?func type {func_concept} .
                ?func name ?func_name .
                ?param type {param_concept} .
                ?param ofFunction ?func .
                ?param name ?param_name .
                ?param atIndex ?param_index .
            }}
            ORDER BY ?func ?param_index
            """
            queries.append(("function_parameters", query1))
            result1 = self.reter.reql(query1)
            params_data = self._query_to_list(result1)

            # Build parameter signatures for each function
            function_signatures = {}
            for func_id, func_name, param_id, param_name, param_index in params_data:
                if func_id not in function_signatures:
                    function_signatures[func_id] = {
                        'name': func_name,
                        'params': []
                    }
                function_signatures[func_id]['params'].append((int(param_index) if param_index else 0, param_name))

            # Sort parameters by index and create signature tuples
            signature_to_functions = {}
            for func_id, data in function_signatures.items():
                params = sorted(data['params'], key=lambda x: x[0])
                param_names = tuple(p[1] for p in params)

                # Skip if less than min_params
                if len(param_names) < min_params:
                    continue

                if param_names not in signature_to_functions:
                    signature_to_functions[param_names] = []
                signature_to_functions[param_names].append((func_id, data['name']))

            # Find duplicates
            duplicates = []
            for signature, functions in signature_to_functions.items():
                if len(functions) >= min_functions:
                    func_names = [f[1] for f in functions]

                    # Calculate severity
                    func_count = len(functions)
                    param_count = len(signature)
                    if func_count >= 5 or param_count >= 5:
                        severity = "HIGH"
                        effort = "moderate"
                    elif func_count >= 3:
                        severity = "MEDIUM"
                        effort = "moderate"
                    else:
                        severity = "LOW"
                        effort = "simple"

                    suggested_interface = self._suggest_interface_name(func_names)

                    duplicates.append({
                        "signature": list(signature),
                        "parameter_count": param_count,
                        "functions": [{"id": f[0], "name": f[1]} for f in functions],
                        "function_count": func_count,
                        "severity": severity,
                        "refactoring": "Change Function Declaration / Extract Interface",
                        "suggestion": f"{func_count} functions share identical parameter signature ({', '.join(signature)}). Consider extracting interface '{suggested_interface}'.",
                        "suggested_interface": suggested_interface,
                        "estimated_effort": effort
                    })

            # Sort by severity and function count
            severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            duplicates.sort(key=lambda x: (severity_order.get(x["severity"], 0), x["function_count"]), reverse=True)

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "duplicates": duplicates[:limit],
                "count": len(duplicates),
                "min_params": min_params,
                "min_functions": min_functions,
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
