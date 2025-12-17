"""
Refactoring Opportunity Detector

Detects opportunities for Martin Fowler's fundamental refactorings from Chapter 6.
Implements automated detection for:
- Extract Function
- Introduce Parameter Object (Data Clumps)
- Combine Functions into Class (Function Groups)

Created as part of Phase 1 implementation of Chapter 6 refactoring patterns.
"""

from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import time
from .base import AdvancedToolsBase


class RefactoringOpportunityDetector(AdvancedToolsBase):
    """
    Detects refactoring opportunities based on Martin Fowler's patterns.

    This service provides detection for:
    - Data clumps (parameters appearing together → Introduce Parameter Object)
    - Function groups (functions sharing data → Combine Functions into Class)
    - Extract function opportunities (long functions, duplicated patterns)

    Responsibilities:
    - Query RETER for code patterns
    - Analyze parameter and call relationships
    - Identify refactoring opportunities
    - Provide actionable recommendations with severity levels
    """

    def __init__(self, reter_wrapper, language="oo"):
        """
        Initialize refactoring opportunity detector.

        Args:
            reter_wrapper: ReterWrapper instance with loaded Python code
            language: Language to analyze ("oo", "python", "javascript")
        """
        super().__init__(reter_wrapper, language)

    def _query_to_list(self, result) -> List[tuple]:
        """
        Convert PyArrow Table to list of tuples.

        Args:
            result: PyArrow Table from REQL query

        Returns:
            List of tuples, one per row
        """
        if result.num_rows == 0:
            return []

        columns = [result.column(name).to_pylist() for name in result.column_names]
        return list(zip(*columns))

    def _query_to_list_padded(self, result, expected_columns: int) -> List[tuple]:
        """
        Convert PyArrow Table to list of tuples, padding with None for missing OPTIONAL columns.

        REQL OPTIONAL clauses may not return columns when they don't match.
        This method ensures consistent tuple length for unpacking.

        Args:
            result: PyArrow Table from REQL query
            expected_columns: Number of columns expected in SELECT clause

        Returns:
            List of tuples, each padded to expected_columns length with None
        """
        if result.num_rows == 0:
            return []

        columns = [result.column(name).to_pylist() for name in result.column_names]
        rows = list(zip(*columns))

        # Pad rows if fewer columns than expected (OPTIONAL columns missing)
        actual_columns = len(result.column_names)
        if actual_columns < expected_columns:
            padding = tuple([None] * (expected_columns - actual_columns))
            rows = [row + padding for row in rows]

        return rows

    # =========================================================================
    # DATA CLUMP DETECTION (Introduce Parameter Object)
    # =========================================================================

    def find_data_clumps(
        self,
        instance_name: str,
        min_params: int = 3,
        min_functions: int = 2
    ) -> Dict[str, Any]:
        """
        Detect parameter groups that appear together in multiple functions.

        Data clumps are groups of parameters that travel together across
        multiple function signatures. This suggests the need for an
        Introduce Parameter Object refactoring.

        Example:
            If calculatePrice(customer, product, quantity) and
            applyDiscount(customer, product, quantity) both have the
            same three parameters, suggest OrderDetails parameter object.

        Args:
            instance_name: RETER instance name
            min_params: Minimum parameters in a clump (default: 3)
            min_functions: Minimum functions sharing parameters (default: 2)

        Returns:
            dict with:
                success: bool
                data_clumps: List of parameter clumps with affected functions
                count: Number of clumps found
                queries: REQL queries executed
                time_ms: Execution time
        """
        start_time = time.time()
        queries = []

        try:
            param_concept = self._concept('Parameter')
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')

            # Query all function-parameter relationships
            # Use inFile (works for all languages) instead of inModule
            query = f"""
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
                ORDER BY ?func ?param_name
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Prefetch call relationships for wrapper pattern detection (batch optimization)
            calls_query = """
                SELECT ?callerName ?calleeName
                WHERE {
                    ?caller calls ?callee .
                    ?caller qualifiedName ?callerName .
                    ?callee qualifiedName ?calleeName .
                }
            """
            queries.append(calls_query.strip())
            calls_result = self.reter.reql(calls_query)
            calls_rows = self._query_to_list(calls_result)

            # Build call graph lookup: caller -> set of callees
            call_graph: Dict[str, set] = defaultdict(set)
            for caller, callee in calls_rows:
                call_graph[caller].add(callee)

            # Prefetch line counts for wrapper pattern detection
            line_count_query = """
                SELECT ?funcName ?lineCount
                WHERE {
                    ?func qualifiedName ?funcName .
                    ?func lineCount ?lineCount .
                }
            """
            queries.append(line_count_query.strip())
            line_result = self.reter.reql(line_count_query)
            line_rows = self._query_to_list(line_result)

            # Build line count lookup
            line_counts: Dict[str, int] = {}
            for func_name, line_count in line_rows:
                try:
                    line_counts[func_name] = int(line_count) if line_count else 999
                except (ValueError, TypeError):
                    line_counts[func_name] = 999

            # Build function -> parameters mapping
            func_params: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
                "params": set(),
                "name": "",
                "module": ""
            })

            for func_id, func_name, module, param_name in rows:
                func_params[func_id]["params"].add(param_name)
                func_params[func_id]["name"] = func_name
                func_params[func_id]["module"] = module

            # Find parameter sets appearing in multiple functions
            param_set_to_functions: Dict[frozenset, List[Dict[str, str]]] = defaultdict(list)

            for func_id, func_data in func_params.items():
                params = func_data["params"]

                # Only consider functions with enough parameters
                if len(params) >= min_params:
                    # Generate all subsets of size >= min_params
                    param_list = sorted(params)

                    # For now, use the full parameter set
                    # TODO: Could enhance to find all n-sized subsets
                    param_frozenset = frozenset(params)

                    param_set_to_functions[param_frozenset].append({
                        "qualified_name": func_id,
                        "name": func_data["name"],
                        "module": func_data["module"]
                    })

            # Filter to clumps appearing in multiple functions
            data_clumps = []

            for param_set, functions in param_set_to_functions.items():
                if len(functions) >= min_functions and len(param_set) >= min_params:
                    # Check if this is a wrapper pattern (false positive) using prefetched data
                    if len(functions) == 2:
                        func1_id = functions[0]["qualified_name"]
                        func2_id = functions[1]["qualified_name"]
                        # Check call relationships
                        is_wrapper = False
                        if func2_id in call_graph.get(func1_id, set()):
                            # func1 calls func2, check if func1 is short
                            if line_counts.get(func1_id, 999) <= 10:
                                is_wrapper = True
                        elif func1_id in call_graph.get(func2_id, set()):
                            # func2 calls func1, check if func2 is short
                            if line_counts.get(func2_id, 999) <= 10:
                                is_wrapper = True
                        if is_wrapper:
                            continue  # Skip wrapper patterns
                    # Calculate severity based on occurrences and parameter count
                    occurrence_count = len(functions)
                    param_count = len(param_set)

                    if occurrence_count >= 5 or param_count >= 5:
                        severity = "HIGH"
                    elif occurrence_count >= 3 or param_count >= 4:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    # Generate suggested parameter object name
                    # Use common naming patterns: combine function name prefixes
                    func_names = [f["name"] for f in functions]
                    suggested_name = self._suggest_parameter_object_name(
                        sorted(param_set),
                        func_names
                    )

                    data_clumps.append({
                        "parameters": sorted(param_set),
                        "parameter_count": param_count,
                        "functions": functions,
                        "occurrence_count": occurrence_count,
                        "severity": severity,
                        "refactoring": "Introduce Parameter Object",
                        "suggestion": f"Create {suggested_name} class with fields: {', '.join(sorted(param_set))}",
                        "estimated_effort": "moderate"
                    })

            # Sort by severity and occurrence count
            severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            data_clumps.sort(
                key=lambda x: (severity_order[x["severity"]], -x["occurrence_count"])
            )

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "data_clumps": data_clumps,
                "count": len(data_clumps),
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
                "data_clumps": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

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

        Attribute data clumps suggest the need for an Introduce Value Object
        refactoring. When multiple classes have the same set of attributes,
        those attributes should likely be extracted into a separate class.

        Example:
            If both Order and Invoice classes have customer_name, customer_address,
            and customer_phone attributes, suggest creating a Customer value object.

        Args:
            instance_name: RETER instance name
            min_attrs: Minimum attributes in a clump (default: 3)
            min_classes: Minimum classes sharing attributes (default: 2)
            limit: Maximum results to return (pagination)
            offset: Number of results to skip (pagination)

        Returns:
            dict with success, attribute_data_clumps list, pagination, time_ms
        """
        start_time = time.time()
        queries = []

        try:
            class_concept = self._concept('Class')
            attr_concept = self._concept('Attribute')

            # Query all class-attribute relationships
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?class ?className ?file ?attrName
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?className .
                    ?class inFile ?file .
                    ?attr type {attr_concept} .
                    ?attr definedIn ?class .
                    ?attr name ?attrName
                }}
                ORDER BY ?class ?attrName
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Build class -> attributes mapping
            class_attrs: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
                "attrs": set(),
                "name": "",
                "file": ""
            })

            for class_id, class_name, file, attr_name in rows:
                class_attrs[class_id]["attrs"].add(attr_name)
                class_attrs[class_id]["name"] = class_name
                class_attrs[class_id]["file"] = file

            # Efficient algorithm: O(N² * M) instead of O(N * 2^M)
            # Compare pairs of classes and find shared attribute intersections
            class_list = [
                (class_id, class_data)
                for class_id, class_data in class_attrs.items()
                if len(class_data["attrs"]) >= min_attrs
            ]

            # Find shared attribute sets via pairwise intersection
            attr_set_to_classes: Dict[frozenset, set] = defaultdict(set)

            for i, (class_id1, data1) in enumerate(class_list):
                for j in range(i + 1, len(class_list)):
                    class_id2, data2 = class_list[j]

                    # Find intersection of attributes
                    shared = data1["attrs"] & data2["attrs"]

                    if len(shared) >= min_attrs:
                        shared_frozen = frozenset(shared)
                        attr_set_to_classes[shared_frozen].add(class_id1)
                        attr_set_to_classes[shared_frozen].add(class_id2)

            # Build class lookup for results
            class_lookup = {cid: data for cid, data in class_list}

            # Filter to clumps appearing in enough classes
            attribute_clumps = []

            for attr_set, class_ids in attr_set_to_classes.items():
                if len(class_ids) >= min_classes and len(attr_set) >= min_attrs:
                    classes = [
                        {
                            "qualified_name": cid,
                            "name": class_lookup[cid]["name"],
                            "module": class_lookup[cid]["module"]
                        }
                        for cid in class_ids
                    ]
                    # Calculate severity based on occurrences and attribute count
                    occurrence_count = len(classes)
                    attr_count = len(attr_set)

                    if occurrence_count >= 4 or attr_count >= 5:
                        severity = "HIGH"
                    elif occurrence_count >= 3 or attr_count >= 4:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    # Suggest value object name
                    class_names = [c["name"] for c in classes]
                    suggested_name = self._suggest_value_object_name(
                        list(attr_set),
                        class_names
                    )

                    attribute_clumps.append({
                        "attributes": sorted(attr_set),
                        "attribute_count": attr_count,
                        "classes": classes,
                        "occurrence_count": occurrence_count,
                        "severity": severity,
                        "refactoring": "Introduce Value Object (Fowler Chapter 7)",
                        "suggested_name": suggested_name,
                        "suggestion": (
                            f"Create {suggested_name} value object with attributes: {', '.join(sorted(attr_set))}\n"
                            f"Replace these {attr_count} attributes in {occurrence_count} classes:\n" +
                            "\n".join(f"  - {c['name']}" for c in classes)
                        ),
                        "estimated_effort": "moderate" if occurrence_count <= 3 else "high"
                    })

            # Sort by severity and occurrence count
            severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            attribute_clumps.sort(
                key=lambda x: (severity_order[x["severity"]], -x["occurrence_count"], -x["attribute_count"])
            )

            # Apply pagination
            total_count = len(attribute_clumps)
            paginated = attribute_clumps[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "attribute_data_clumps": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "min_attrs": min_attrs,
                "min_classes": min_classes,
                "queries": queries,
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "attribute_data_clumps": [],
                "count": 0,
                "total_count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def _suggest_parameter_object_name(
        self,
        params: List[str],
        func_names: List[str]
    ) -> str:
        """
        Suggest a name for parameter object based on parameters and functions.

        Args:
            params: List of parameter names
            func_names: List of function names using these parameters

        Returns:
            Suggested class name for parameter object
        """
        # Strategy 1: If params include obvious domain term (customer, order, product)
        domain_terms = {'customer', 'order', 'product', 'invoice', 'user', 'account', 'payment'}
        for param in params:
            param_lower = param.lower()
            if param_lower in domain_terms:
                return param.capitalize() + "Details"

        # Strategy 2: Look for common prefix in function names
        if len(func_names) >= 2:
            prefix = self._common_prefix(func_names)
            if len(prefix) > 2:
                return prefix.capitalize() + "Parameters"

        # Strategy 3: Use first parameter name as basis
        if params:
            return params[0].capitalize() + "Group"

        # Fallback
        return "ParameterObject"

    def _suggest_value_object_name(
        self,
        attrs: List[str],
        class_names: List[str]
    ) -> str:
        """
        Suggest a name for value object based on attributes and classes.

        Args:
            attrs: List of attribute names
            class_names: List of class names sharing these attributes

        Returns:
            Suggested class name for value object
        """
        # Strategy 1: If attributes include obvious domain term (customer, address, phone)
        domain_terms = {
            'customer': 'Customer',
            'address': 'Address',
            'location': 'Location',
            'phone': 'Contact',
            'email': 'Contact',
            'user': 'User',
            'account': 'Account',
            'payment': 'Payment',
            'order': 'Order',
            'product': 'Product',
            'price': 'Price',
            'amount': 'Amount',
            'date': 'DateInfo',
            'time': 'TimeInfo',
            'name': 'Name',
            'id': 'Identifier'
        }

        for attr in attrs:
            attr_lower = attr.lower().replace('_', '')
            for term, name in domain_terms.items():
                if term in attr_lower:
                    return name

        # Strategy 2: Look for common prefix in attribute names
        if len(attrs) >= 2:
            prefix = self._common_prefix(attrs)
            if len(prefix) > 2:
                return prefix.capitalize()

        # Strategy 3: Look for common suffix in attribute names
        common_suffix = self._common_suffix(attrs)
        if len(common_suffix) > 2:
            return common_suffix.capitalize() + "Info"

        # Strategy 4: Use first attribute name as basis
        if attrs:
            base = attrs[0].replace('_', ' ').title().replace(' ', '')
            return base + "Data"

        # Fallback
        return "ValueObject"

    def _common_suffix(self, strings: List[str]) -> str:
        """Find common suffix of strings."""
        if not strings:
            return ""

        # Reverse strings to find common prefix of reversed
        reversed_strings = [s[::-1] for s in strings]
        reversed_suffix = self._common_prefix(reversed_strings)
        return reversed_suffix[::-1]

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

    def _is_wrapper_pattern(self, func1: Dict[str, str], func2: Dict[str, str]) -> bool:
        """
        Detect if two functions are in a wrapper/delegate pattern.

        Wrapper pattern characteristics:
        - One function calls the other
        - The calling function is short (typically < 10 lines)
        - Minimal additional logic beyond delegation

        This helps eliminate false positives where parameter passing is intentional
        architectural design (Facade/Adapter patterns).

        Args:
            func1: First function dict with qualified_name, name, module
            func2: Second function dict with qualified_name, name, module

        Returns:
            True if functions form a wrapper pattern, False otherwise
        """
        try:
            func1_id = func1["qualified_name"]
            func2_id = func2["qualified_name"]

            # Query 1: Check if func1 calls func2 or vice versa
            calls_query = f"""
            SELECT ?callerName ?calleeName
            WHERE {{
                ?caller calls ?callee .
                ?caller qualifiedName ?callerName .
                ?callee qualifiedName ?calleeName .
                FILTER(
                    (?callerName = "{func1_id}" && ?calleeName = "{func2_id}") ||
                    (?callerName = "{func2_id}" && ?calleeName = "{func1_id}")
                )
            }}
            """
            calls_result = self.reter.reql(calls_query)
            calls_list = self._query_to_list(calls_result)

            if not calls_list:
                return False  # No call relationship

            # Determine which is the wrapper (caller) - first column is callerName
            caller_id = calls_list[0][0]

            # Query 2: Check line count of the wrapper
            line_count_query = f"""
            SELECT ?line_count
            WHERE {{
                ?func qualifiedName "{caller_id}" .
                ?func lineCount ?line_count .
            }}
            """
            line_result = self.reter.reql(line_count_query)
            line_list = self._query_to_list(line_result)

            if not line_list:
                return False

            line_count = int(line_list[0][0]) if line_list[0][0] else 999

            # Wrapper pattern: short function (< 10 lines) that calls another
            if line_count <= 10:
                return True

            return False

        except Exception as e:
            # If there's any error in pattern detection, don't skip
            return False

    # =========================================================================
    # FUNCTION GROUP DETECTION (Combine Functions into Class)
    # =========================================================================

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
            → Suggest ReadingProcessor class

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

        except Exception as e:
            # If there's any error, don't skip (conservative approach)
            return False

    # =========================================================================
    # EXTRACT FUNCTION OPPORTUNITIES
    # =========================================================================

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
    ):
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

            # Query 1: Find all functions with line count ≤ max_lines
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
    ):
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

    # =========================================================================
    # SHOTGUN SURGERY DETECTION
    # =========================================================================

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
        code that is called from many different locations, suggesting that
        related behavior should be consolidated.

        Detection:
        - Functions/classes called by >= min_callers callers
        - Callers spread across >= min_modules different modules
        - Suggests: Extract Class or Move Method to consolidate

        Args:
            instance_name: RETER instance name
            min_callers: Minimum number of callers to flag (default: 5)
            min_modules: Minimum number of calling modules (default: 3)

        Returns:
            success: bool
            shotgun_surgery_cases: List of high fan-in entities
            count: Number of cases found
            queries: REQL queries executed
            time_ms: Execution time
        """
        start_time = time.time()
        queries = []

        try:
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')

            # Query: Get all call relationships
            # Use inFile (works for all languages) instead of inModule
            query = f"""
            SELECT ?caller ?callerName ?callerFile ?callee ?calleeName ?calleeFile
            WHERE {{
                ?caller type ?callerType .
                ?caller name ?callerName .
                ?caller inFile ?callerFile .
                ?caller calls ?callee .
                ?callee type ?calleeType .
                ?callee name ?calleeName .
                ?callee inFile ?calleeFile .
                FILTER(?callerType = "{func_concept}" || ?callerType = "{method_concept}")
                FILTER(?calleeType = "{func_concept}" || ?calleeType = "{method_concept}")
            }}
            """
            queries.append(query)

            result = self.reter.reql(query)
            calls_list = self._query_to_list(result)

            # Build fan-in data: callee -> list of (caller, caller_file)
            fan_in = defaultdict(list)
            callee_info = {}  # callee_id -> (name, file)

            for caller_id, caller_name, caller_file, callee_id, callee_name, callee_file in calls_list:
                fan_in[callee_id].append((caller_id, caller_file))
                callee_info[callee_id] = (callee_name, callee_file)

            # Find high fan-in cases
            shotgun_cases = []

            for callee_id, callers in fan_in.items():
                caller_count = len(callers)
                unique_files = len(set(file for _, file in callers))

                # Check thresholds
                if caller_count >= min_callers and unique_files >= min_modules:
                    callee_name, callee_file = callee_info[callee_id]

                    # Calculate severity
                    if caller_count >= 10 and unique_files >= 5:
                        severity = "CRITICAL"
                    elif caller_count >= 7 and unique_files >= 4:
                        severity = "HIGH"
                    else:
                        severity = "MEDIUM"

                    shotgun_cases.append({
                        "entity": callee_id,
                        "entity_name": callee_name,
                        "file": callee_file,
                        "caller_count": caller_count,
                        "file_count": unique_files,
                        "severity": severity,
                        "smell": "Shotgun Surgery",
                        "refactoring": "Extract Class or Consolidate Behavior",
                        "suggestion": f"Function '{callee_name}' is called from {caller_count} places across {unique_files} files. Consider consolidating related behavior into a class or module.",
                        "estimated_effort": "medium" if severity != "CRITICAL" else "high"
                    })

            # Sort by severity and caller count
            severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            shotgun_cases.sort(key=lambda x: (severity_order[x["severity"]], -x["caller_count"]))

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "shotgun_surgery_cases": shotgun_cases,
                "count": len(shotgun_cases),
                "queries": queries,
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "shotgun_surgery_cases": [],
                "count": 0,
                "time_ms": time_ms
            }

    # =========================================================================
    # MIDDLE MAN DETECTION
    # =========================================================================

    def find_middle_man(
        self,
        instance_name: str,
        max_lines: int = 10,
        min_delegation_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect classes/methods that just delegate to other classes.

        Middle Man (Fowler, Chapter 3) occurs when a class or method does
        little more than delegate to another class. This adds unnecessary
        indirection without adding value.

        Detection:
        - Methods with <= max_lines lines
        - Methods where delegation calls >= min_delegation_ratio of total calls
        - Suggests: Remove Middle Man or Inline Method

        Args:
            instance_name: RETER instance name
            max_lines: Maximum lines for a method to be considered (default: 10)
            min_delegation_ratio: Minimum ratio of delegating calls (default: 0.5)

        Returns:
            success: bool
            middle_man_cases: List of middle man methods/classes
            count: Number of cases found
            queries: REQL queries executed
            time_ms: Execution time
        """
        start_time = time.time()
        queries = []

        try:
            method_concept = self._concept('Method')

            # Query 1: Get all methods with line counts
            # Use inFile (works for all languages) instead of inModule
            query1 = f"""
            SELECT ?method ?methodName ?className ?file ?lines
            WHERE {{
                ?method type {method_concept} .
                ?method name ?methodName .
                ?method inClass ?class .
                ?class name ?className .
                ?method inFile ?file .
                ?method lineCount ?lines .
                FILTER(?lines <= {max_lines})
            }}
            """
            queries.append(query1)

            result1 = self.reter.reql(query1)
            methods_list = self._query_to_list(result1)

            # Query 2: Get all call relationships for these methods
            query2 = f"""
            SELECT ?caller ?callee ?calleeName
            WHERE {{
                ?caller type {method_concept} .
                ?caller calls ?callee .
                ?callee name ?calleeName .
            }}
            """
            queries.append(query2)

            result2 = self.reter.reql(query2)
            calls_list = self._query_to_list(result2)

            # Build calls map: method -> list of callees
            method_calls = defaultdict(list)
            for caller_id, callee_id, callee_name in calls_list:
                method_calls[caller_id].append((callee_id, callee_name))

            # Analyze each short method
            middle_man_cases = []

            for method_id, method_name, class_name, file, lines in methods_list:
                calls = method_calls.get(method_id, [])
                total_calls = len(calls)

                if total_calls == 0:
                    continue

                # Check if most calls are to the same external class
                # (delegation pattern)
                callee_counts = defaultdict(int)
                for callee_id, callee_name in calls:
                    # Extract class from qualified name if it's a method
                    if "." in callee_name:
                        external_class = callee_name.split(".")[0]
                    else:
                        external_class = callee_name
                    callee_counts[external_class] += 1

                # Find most common delegation target
                if callee_counts:
                    max_delegations = max(callee_counts.values())
                    delegation_ratio = max_delegations / total_calls

                    if delegation_ratio >= min_delegation_ratio:
                        # This is a middle man
                        severity = "HIGH" if delegation_ratio >= 0.8 else "MEDIUM"

                        middle_man_cases.append({
                            "method": method_id,
                            "method_name": method_name,
                            "class_name": class_name,
                            "file": file,
                            "lines": int(lines),
                            "total_calls": total_calls,
                            "delegation_ratio": round(delegation_ratio, 2),
                            "severity": severity,
                            "smell": "Middle Man",
                            "refactoring": "Remove Middle Man or Inline Method",
                            "suggestion": f"Method '{method_name}' in class '{class_name}' primarily delegates to other classes ({int(delegation_ratio * 100)}% of calls). Consider removing this indirection.",
                            "estimated_effort": "simple"
                        })

            # Sort by delegation ratio
            middle_man_cases.sort(key=lambda x: -x["delegation_ratio"])

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "middle_man_cases": middle_man_cases,
                "count": len(middle_man_cases),
                "queries": queries,
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "middle_man_cases": [],
                "count": 0,
                "time_ms": time_ms
            }

    # =========================================================================
    # EXTRACT CLASS DETECTION
    # =========================================================================

    def find_extract_class_opportunities(
        self,
        instance_name: str,
        min_methods: int = 10,
        min_cohesion_gap: float = 0.3
    ) -> Dict[str, Any]:
        """
        Detect classes that should be split into multiple classes.

        Extract Class (Fowler, Chapter 7) is needed when a class is doing
        the work of two or more classes. This detector identifies:
        - Large classes with many methods
        - Low cohesion (methods using different attributes)
        - Distinct method groups suggesting separate responsibilities

        Detection:
        - Classes with >= min_methods methods
        - Identifies method groups based on call patterns
        - Suggests: Extract Class for each distinct group

        Args:
            instance_name: RETER instance name
            min_methods: Minimum methods to consider (default: 10)
            min_cohesion_gap: Minimum cohesion difference to split (default: 0.3)

        Returns:
            success: bool
            extract_class_opportunities: List of classes to split
            count: Number of opportunities found
            queries: REQL queries executed
            time_ms: Execution time
        """
        start_time = time.time()
        queries = []

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query 1: Get classes with method counts
            # Use inFile (works for all languages) instead of inModule
            query1 = f"""
            SELECT ?class ?className ?file (COUNT(?method) AS ?methodCount)
            WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?class inFile ?file .
                ?method type {method_concept} .
                ?method inClass ?class .
            }}
            GROUP BY ?class ?className ?file
            HAVING (COUNT(?method) >= {min_methods})
            ORDER BY DESC(?methodCount)
            """
            queries.append(query1)

            result1 = self.reter.reql(query1)
            classes_list = self._query_to_list(result1)

            # Query 2: Get method calls within classes
            query2 = f"""
            SELECT ?caller ?callee
            WHERE {{
                ?caller type {method_concept} .
                ?caller inClass ?class .
                ?caller calls ?callee .
                ?callee type {method_concept} .
                ?callee inClass ?class .
            }}
            """
            queries.append(query2)

            result2 = self.reter.reql(query2)
            calls_list = self._query_to_list(result2)

            # Build call graph within each class
            internal_calls = defaultdict(set)
            for caller, callee in calls_list:
                internal_calls[caller].add(callee)

            # Analyze each large class
            opportunities = []

            for class_id, class_name, file, method_count in classes_list:
                method_count = int(method_count)

                # Calculate severity based on size
                if method_count >= 20:
                    severity = "CRITICAL"
                elif method_count >= 15:
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"

                opportunities.append({
                    "class": class_id,
                    "class_name": class_name,
                    "file": file,
                    "method_count": method_count,
                    "severity": severity,
                    "smell": "Large Class / God Class",
                    "refactoring": "Extract Class",
                    "suggestion": f"Class '{class_name}' has {method_count} methods. Consider extracting related methods into separate classes with focused responsibilities.",
                    "estimated_effort": "high" if method_count >= 20 else "medium",
                    "recommendation": "Identify groups of methods that work on related data and extract them into cohesive classes."
                })

            # Sort by severity and method count
            severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            opportunities.sort(key=lambda x: (severity_order[x["severity"]], -x["method_count"]))

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "extract_class_opportunities": opportunities,
                "count": len(opportunities),
                "queries": queries,
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "extract_class_opportunities": [],
                "count": 0,
                "time_ms": time_ms
            }

    def analyze_refactoring_opportunities(
        self,
        instance_name: str,
        severity_threshold: str = "MEDIUM",
        limit_per_type: int = 5,
        limit: int = 20,
        offset: int = 0,
        quick_mode: bool = False
    ):
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
            success: Whether analysis succeeded
            summary: Summary statistics by refactoring type and severity (before pagination)
            opportunities: All opportunities sorted by priority (paginated)
            total_count: Total opportunities found (before pagination)
            count_returned: Number of items returned in this page
            limit: Limit used for pagination
            offset: Offset used for pagination
            has_more: Whether there are more results available
            next_offset: Offset for next page (if applicable)
            time_ms: Total execution time

        Performance Notes:
            - Use quick_mode=True for 2-3x faster results (runs only 2 fast detectors)
            - Reduce limit for faster queries (limit=10 is ~2x faster than limit=50)
            - Lower limit_per_type reduces per-detector overhead
        """
        start_time = time.time()
        all_opportunities = []
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

        try:
            # Early termination helper
            def should_continue():
                """Check if we should continue running detectors"""
                if quick_mode:
                    return False  # Quick mode: stop after fast detectors
                critical_high_count = sum(
                    1 for opp in all_opportunities
                    if severity_order.get(opp.get("severity", "LOW"), 1) >= 3
                )
                # Stop if we have enough high-priority results
                return critical_high_count < (limit * 2)

            # 1. Extract Function (Fast - simple line count query)
            extract_result = self.find_extract_function_opportunities(instance_name, min_lines=20)
            if extract_result["success"]:
                for opp in extract_result["opportunities"][:limit_per_type]:
                    all_opportunities.append({
                        **opp,
                        "type": "Extract Function"
                    })

            # 2. Inline Function (Fast - simple call count query)
            inline_result = self.find_inline_function_candidates(instance_name, max_lines=5, limit=limit_per_type)
            if inline_result["success"]:
                for candidate in inline_result["candidates"][:limit_per_type]:
                    all_opportunities.append({
                        **candidate,
                        "type": "Inline Function"
                    })

            # Stop here if quick_mode or we have enough results
            if not should_continue():
                # Skip remaining slow detectors
                pass
            else:
                # 3. Data Clumps (Slow - requires parameter analysis)
                clumps_result = self.find_data_clumps(instance_name, min_params=3, min_functions=2)
                if clumps_result["success"]:
                    for clump in clumps_result["data_clumps"][:limit_per_type]:
                        all_opportunities.append({
                            "name": f"Data Clump: {', '.join(clump['parameters'])}",
                            "type": "Introduce Parameter Object",
                            "severity": clump["severity"],
                            "suggestion": clump["suggestion"],
                            "estimated_effort": clump["estimated_effort"],
                            "affected_functions": clump["occurrence_count"],
                            "details": clump
                        })

                # 4. Function Groups (Slow - requires shared parameter analysis)
                if should_continue():
                    groups_result = self.find_function_groups(instance_name, min_shared_params=2, min_functions=3)
                    if groups_result["success"]:
                        for group in groups_result["function_groups"][:limit_per_type]:
                            # Extract suggested class name from functions
                            func_names = [f["name"] for f in group["functions"][:3]]
                            group_name = self._suggest_class_name(func_names, group["shared_parameters"])
                            all_opportunities.append({
                                "name": f"Function Group: {group_name}",
                                "type": "Combine Functions into Class",
                                "severity": group["severity"],
                                "suggestion": group["suggestion"],
                                "estimated_effort": group["estimated_effort"],
                                "affected_functions": group["function_count"],
                                "details": group
                            })

                # 5. Duplicate Parameter Lists (Medium - signature comparison)
                if should_continue():
                    duplicate_result = self.find_duplicate_parameter_lists(instance_name, min_params=2, min_functions=2, limit=limit_per_type)
                    if duplicate_result["success"]:
                        for dup in duplicate_result["duplicates"][:limit_per_type]:
                            all_opportunities.append({
                                "name": f"Duplicate Signature: {dup['suggested_interface']}",
                                "type": "Extract Interface",
                                "severity": dup["severity"],
                                "suggestion": dup["suggestion"],
                                "estimated_effort": dup["estimated_effort"],
                                "affected_functions": dup["function_count"],
                                "details": dup
                            })

            # Filter by severity threshold
            severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            threshold_value = severity_order.get(severity_threshold.upper(), 2)
            filtered_opportunities = [
                opp for opp in all_opportunities
                if severity_order.get(opp.get("severity", "LOW"), 1) >= threshold_value
            ]

            # Sort by severity and effort
            effort_order = {"simple": 1, "moderate": 2, "high": 3}
            filtered_opportunities.sort(
                key=lambda x: (
                    -severity_order.get(x.get("severity", "LOW"), 1),
                    effort_order.get(x.get("estimated_effort", "moderate"), 2)
                )
            )

            # Generate summary (before pagination)
            summary = {
                "by_type": {},
                "by_severity": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
            }

            for opp in filtered_opportunities:
                opp_type = opp.get("type", "Unknown")
                severity = opp.get("severity", "LOW")

                if opp_type not in summary["by_type"]:
                    summary["by_type"][opp_type] = 0
                summary["by_type"][opp_type] += 1
                summary["by_severity"][severity] += 1

            # Apply pagination
            total_count = len(filtered_opportunities)
            paginated_opportunities = filtered_opportunities[offset:offset + limit]
            count_returned = len(paginated_opportunities)

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "summary": summary,
                "opportunities": paginated_opportunities,
                "total_count": total_count,
                "count_returned": count_returned,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count,
                "next_offset": offset + limit if (offset + limit) < total_count else None,
                "severity_threshold": severity_threshold,
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "summary": {},
                "opportunities": [],
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_inline_class_opportunities(
        self,
        instance_name: str,
        max_methods: int = 3,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect small, trivial classes that should be inlined into their clients.

        This implements Fowler's "Inline Class" refactoring (Chapter 7).

        Detection signals:
        - Class has <= max_methods methods (default: 3)
        - Class is used by only 1 or 2 client classes
        - Class is a data class (only getters/setters, no business logic)
        - Class has no subclasses (can't inline if subclassed)

        Args:
            instance_name: RETER instance name
            max_methods: Maximum method count for small class (default: 3)
            limit: Maximum results to return (pagination)
            offset: Starting offset for pagination

        Returns:
            dict with success, inline_class_opportunities (list), pagination metadata
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query 1: Find small classes with method count
            # Use inFile (works for all languages) instead of inModule
            query_small_classes = f"""
            SELECT ?class ?className ?file (COUNT(?method) AS ?methodCount)
            WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?class inFile ?file .
                ?method type {method_concept} .
                ?method inClass ?class .
            }}
            GROUP BY ?class ?className ?file
            HAVING (COUNT(?method) <= {max_methods})
            ORDER BY ?methodCount
            """

            result_small_classes = self.reter.reql(query_small_classes)
            small_classes = self._query_to_list(result_small_classes)

            if not small_classes:
                time_ms = (time.time() - start_time) * 1000
                return {
                    "success": True,
                    "inline_class_opportunities": [],
                    "count": 0,
                    "total_count": 0,
                    "count_returned": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "next_offset": None,
                    "queries": [query_small_classes],
                    "time_ms": time_ms
                }

            # Query 2: Find all call relationships to build client map
            query_calls = f"""
            SELECT ?callerClass ?targetClass
            WHERE {{
                ?caller type {method_concept} .
                ?caller inClass ?callerClass .
                ?callee type {method_concept} .
                ?callee inClass ?targetClass .
                ?caller calls ?callee .
            }}
            """

            result_calls = self.reter.reql(query_calls)
            calls = self._query_to_list(result_calls)

            # Build client map: target class -> set of client classes
            client_map = defaultdict(set)
            for caller_class, target_class in calls:
                if caller_class != target_class:  # Exclude self-calls
                    client_map[target_class].add(caller_class)

            # Query 3: Find classes with subclasses (can't inline if subclassed)
            query_subclasses = f"""
            SELECT ?class ?subclass
            WHERE {{
                ?subclass type {class_concept} .
                ?subclass inheritsFrom ?class .
            }}
            """

            result_subclasses = self.reter.reql(query_subclasses)
            subclassed = set()
            for parent_class, _ in self._query_to_list(result_subclasses):
                subclassed.add(parent_class)

            # Analyze each small class
            opportunities = []

            for class_id, class_name, file, method_count in small_classes:
                # Skip if class has subclasses
                if class_id in subclassed:
                    continue

                # Count unique clients
                clients = client_map.get(class_id, set())
                client_count = len(clients)

                # Skip if no clients or too many clients
                if client_count == 0 or client_count > 2:
                    continue

                # Determine severity based on method count and client count
                if method_count == 1 and client_count == 1:
                    severity = "HIGH"
                    confidence = "high"
                elif method_count <= 2 and client_count == 1:
                    severity = "MEDIUM"
                    confidence = "medium"
                else:
                    severity = "LOW"
                    confidence = "low"

                # Build client list
                client_names = []
                for client_id in clients:
                    # Extract client class name from ID
                    client_name = client_id.split(".")[-1] if "." in client_id else client_id
                    client_names.append(client_name)

                opportunity = {
                    "class": class_id,
                    "class_name": class_name,
                    "file": file,
                    "method_count": method_count,
                    "client_count": client_count,
                    "client_classes": client_names,
                    "severity": severity,
                    "confidence": confidence,
                    "smell": "Lazy Class / Trivial Class",
                    "refactoring": "Inline Class",
                    "suggestion": (
                        f"Class '{class_name}' has only {method_count} method(s) and is used by "
                        f"only {client_count} client class(es): {', '.join(client_names)}. "
                        f"Consider inlining it into the client class. "
                        f"This reduces unnecessary abstraction and simplifies the codebase."
                    ),
                    "estimated_effort": "simple" if method_count <= 2 else "moderate",
                    "recommendation": (
                        f"1. Move all methods from '{class_name}' into '{client_names[0]}'\n"
                        f"2. Move all data fields into '{client_names[0]}'\n"
                        f"3. Update references to use '{client_names[0]}' directly\n"
                        f"4. Delete '{class_name}'"
                    )
                }

                opportunities.append(opportunity)

            # Sort by severity and method count
            severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            opportunities.sort(
                key=lambda x: (-severity_order[x["severity"]], x["method_count"])
            )

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            count_returned = len(paginated)

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "inline_class_opportunities": paginated,
                "count": count_returned,
                "total_count": total_count,
                "count_returned": count_returned,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count,
                "next_offset": offset + limit if (offset + limit) < total_count else None,
                "queries": [query_small_classes, query_calls, query_subclasses],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "inline_class_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_primitive_obsession(
        self,
        instance_name: str,
        min_usages: int = 5,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect primitive parameters (str, int, float) that should be value objects.

        This implements detection for Fowler's "Replace Primitive with Object"
        refactoring (Chapter 7).

        Detection signals:
        - Same primitive parameter name appears in many functions
        - High fan-out (>= min_usages functions)
        - Parameter name suggests domain concept (email, phone, money, etc.)

        Args:
            instance_name: RETER instance name
            min_usages: Minimum function count to flag (default: 5)
            limit: Maximum results to return (pagination)
            offset: Starting offset for pagination

        Returns:
            dict with success, primitive_obsession_cases (list), pagination metadata
        """
        start_time = time.time()

        # Domain concept keywords for enhanced detection
        DOMAIN_CONCEPTS = {
            "email": ["email", "mail", "e_mail"],
            "phone": ["phone", "telephone", "mobile", "cell"],
            "url": ["url", "link", "uri", "href", "endpoint"],
            "money": ["price", "amount", "cost", "fee", "balance", "salary", "payment"],
            "percentage": ["percent", "rate", "ratio"],
            "identifier": ["id", "uuid", "guid", "code", "key"],
            "priority": ["priority", "level", "rank", "severity"],
            "status": ["status", "state", "stage"],
            "date": ["date", "time", "timestamp", "datetime", "when"],
            "address": ["address", "location", "addr"],
            "name": ["name", "title", "label"],
        }

        def normalize_param_name(name: str) -> str:
            """Normalize parameter name to detect variants."""
            name_lower = name.lower()

            # Check domain concepts
            for concept, keywords in DOMAIN_CONCEPTS.items():
                for keyword in keywords:
                    if keyword in name_lower:
                        return concept

            # Return original if no match
            return name_lower

        def suggest_class_name(normalized_name: str, param_type: str) -> str:
            """Suggest a value class name based on normalized parameter name."""
            # Capitalize and remove underscores
            if normalized_name in DOMAIN_CONCEPTS:
                # Use concept name
                if normalized_name == "email":
                    return "EmailAddress"
                elif normalized_name == "phone":
                    return "PhoneNumber"
                elif normalized_name == "url":
                    return "URL"
                elif normalized_name == "money":
                    return "Money"
                elif normalized_name == "identifier":
                    return "Identifier"
                else:
                    return normalized_name.capitalize()
            else:
                # Use parameter name
                return "".join(word.capitalize() for word in normalized_name.split("_"))

        try:
            param_concept = self._concept('Parameter')

            # Query: Find all primitive-typed parameters
            # Use inFile (works for all languages) instead of inModule
            query_params = f"""
            SELECT ?param ?paramName ?paramType ?func ?funcName ?file
            WHERE {{
                ?param type {param_concept} .
                ?param name ?paramName .
                ?param hasType ?paramType .
                ?param ofFunction ?func .
                ?func name ?funcName .
                ?func inFile ?file .
                FILTER(?paramType = "str" || ?paramType = "int" || ?paramType = "float" ||
                       ?paramType = "bool")
            }}
            """

            result_params = self.reter.reql(query_params)
            params = self._query_to_list(result_params)

            if not params:
                time_ms = (time.time() - start_time) * 1000
                return {
                    "success": True,
                    "primitive_obsession_cases": [],
                    "count": 0,
                    "total_count": 0,
                    "count_returned": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "next_offset": None,
                    "queries": [query_params],
                    "time_ms": time_ms
                }

            # Group parameters by normalized name and type
            param_groups = defaultdict(lambda: {
                "functions": [],
                "files": set(),
                "original_names": set(),
                "param_type": None
            })

            for param_id, param_name, param_type, func_id, func_name, file in params:
                normalized = normalize_param_name(param_name)
                key = (normalized, param_type)

                param_groups[key]["functions"].append({
                    "func_id": func_id,
                    "func_name": func_name,
                    "file": file
                })
                param_groups[key]["files"].add(file)
                param_groups[key]["original_names"].add(param_name)
                param_groups[key]["param_type"] = param_type

            # Analyze each parameter group
            opportunities = []

            for (normalized_name, param_type), data in param_groups.items():
                usage_count = len(data["functions"])
                file_count = len(data["files"])

                # Skip if usage below threshold
                if usage_count < min_usages:
                    continue

                # Determine severity based on usage count
                if usage_count >= 15:
                    severity = "CRITICAL"
                    confidence = "high"
                elif usage_count >= 10:
                    severity = "HIGH"
                    confidence = "high"
                elif usage_count >= 5:
                    severity = "MEDIUM"
                    confidence = "medium"
                else:
                    severity = "LOW"
                    confidence = "low"

                # Check if it's a recognized domain concept
                is_domain_concept = normalized_name in DOMAIN_CONCEPTS

                # Suggest value class name
                suggested_class = suggest_class_name(normalized_name, param_type)

                # Build function list (limit to first 10 for readability)
                function_list = []
                for func_data in data["functions"][:10]:
                    func_name = func_data["func_name"]
                    file = func_data["file"]
                    # Extract short file name
                    short_file = file.split("/")[-1].split("\\")[-1].replace(".py", "")
                    function_list.append(f"{func_name} ({short_file})")

                opportunity = {
                    "normalized_name": normalized_name,
                    "param_type": param_type,
                    "original_names": sorted(data["original_names"]),
                    "usage_count": usage_count,
                    "file_count": file_count,
                    "is_domain_concept": is_domain_concept,
                    "suggested_class_name": suggested_class,
                    "severity": severity,
                    "confidence": confidence,
                    "smell": "Primitive Obsession",
                    "refactoring": "Replace Primitive with Object",
                    "suggestion": (
                        f"Primitive parameter '{', '.join(sorted(data['original_names'])[:3])}: {param_type}' "
                        f"is used in {usage_count} functions across {file_count} file(s). "
                        f"Consider creating a value class '{suggested_class}' to encapsulate this domain concept. "
                        f"This provides type safety, centralized validation, and domain-specific operations."
                    ),
                    "estimated_effort": "moderate" if usage_count >= 10 else "simple",
                    "functions_sample": function_list,
                    "total_functions": usage_count,
                    "recommendation": (
                        f"1. Create value class '{suggested_class}':\n"
                        f"   class {suggested_class}:\n"
                        f"       def __init__(self, value: {param_type}):\n"
                        f"           self._value = self._validate(value)\n"
                        f"       def _validate(self, value):\n"
                        f"           # Add validation logic\n"
                        f"           return value\n"
                        f"       def __str__(self):\n"
                        f"           return str(self._value)\n"
                        f"2. Replace all {usage_count} uses of '{normalized_name}: {param_type}' with '{normalized_name}: {suggested_class}'\n"
                        f"3. Add domain-specific methods to '{suggested_class}' as needed"
                    )
                }

                opportunities.append(opportunity)

            # Sort by severity and usage count
            severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            opportunities.sort(
                key=lambda x: (-severity_order[x["severity"]], -x["usage_count"])
            )

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            count_returned = len(paginated)

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "primitive_obsession_cases": paginated,
                "count": count_returned,
                "total_count": total_count,
                "count_returned": count_returned,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count,
                "next_offset": offset + limit if (offset + limit) < total_count else None,
                "queries": [query_params],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "primitive_obsession_cases": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_encapsulate_collection_opportunities(
        self,
        instance_name: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect methods that return mutable collections without encapsulation.

        Fowler Chapter 7: Encapsulate Collection

        Detection signals (heuristic approach using type hints):
        - Method returns List, Dict, or Set type
        - No corresponding add_X or remove_X methods exist
        - Suggests wrapping collection access in proper methods

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return (pagination)
            offset: Number of results to skip (pagination)

        Returns:
            dict with success, encapsulate_collection_opportunities list,
            pagination metadata, time_ms

        Example:
            # BEFORE (bad - direct collection access)
            class Person:
                def get_courses(self) -> List[Course]:
                    return self._courses  # Clients can modify!

            # AFTER (good - encapsulated)
            class Person:
                def get_courses(self) -> List[Course]:
                    return self._courses.copy()  # Return copy

                def add_course(self, course: Course):
                    self._courses.append(course)

                def remove_course(self, course: Course):
                    self._courses.remove(course)
        """
        start_time = time.time()

        try:
            method_concept = self._concept('Method')

            # Query: Find methods returning collection types
            # Use inFile (works for all languages) instead of inModule
            query_collection_getters = f"""
            SELECT ?method ?methodName ?returnType ?class ?className ?file
            WHERE {{
                ?method type {method_concept} .
                ?method name ?methodName .
                ?method inClass ?class .
                ?class name ?className .
                ?class inFile ?file .
                ?method hasReturnType ?returnType .
                FILTER(?returnType = "List" || ?returnType = "list" ||
                       ?returnType = "Dict" || ?returnType = "dict" ||
                       ?returnType = "Set" || ?returnType = "set" ||
                       ?returnType = "List[" || ?returnType = "Dict[" || ?returnType = "Set[")
            }}
            ORDER BY ?className ?methodName
            """

            result = self.reter.reql(query_collection_getters)
            collection_getters = self._query_to_list(result)

            # Single query to get all methods for all classes (for modifier detection)
            query_all_methods = f"""
            SELECT ?class ?className ?methodName
            WHERE {{
                ?method type {method_concept} .
                ?method name ?methodName .
                ?method inClass ?class .
                ?class name ?className .
            }}
            """
            all_methods_result = self.reter.reql(query_all_methods)
            all_methods = self._query_to_list(all_methods_result)

            # Build mapping: class_name -> list of method names
            import re
            class_methods: Dict[str, List[str]] = {}
            for class_id, class_name, method_name in all_methods:
                if class_name not in class_methods:
                    class_methods[class_name] = []
                class_methods[class_name].append(method_name)

            # Build opportunities by analyzing each getter
            opportunities = []

            for method_id, method_name, return_type, class_id, class_name, file in collection_getters:
                # Parse element name from method name
                # e.g., get_courses -> "course", get_items -> "item", students -> "student"
                element_name = self._extract_element_name(method_name)

                if not element_name:
                    continue

                # Find add/remove methods in the same class using Python regex (no query needed)
                pattern = re.compile(f"^(add_|remove_|append_|delete_){element_name}")
                methods_in_class = class_methods.get(class_name, [])
                modifiers = [(None, m) for m in methods_in_class if pattern.match(m)]

                has_add = any("add" in m_name or "append" in m_name
                              for _, m_name in modifiers)
                has_remove = any("remove" in m_name or "delete" in m_name
                                for _, m_name in modifiers)

                # Flag if no modification methods exist
                if not has_add and not has_remove:
                    severity = "HIGH"  # No encapsulation at all
                elif not has_add or not has_remove:
                    severity = "MEDIUM"  # Partial encapsulation
                else:
                    continue  # Both exist, properly encapsulated

                # Determine collection type
                if "List" in return_type or "list" in return_type:
                    collection_type = "List"
                elif "Dict" in return_type or "dict" in return_type:
                    collection_type = "Dict"
                elif "Set" in return_type or "set" in return_type:
                    collection_type = "Set"
                else:
                    collection_type = return_type

                # Generate suggestions
                suggestions = []
                if not has_add:
                    suggestions.append(f"Add 'add_{element_name}()' method for controlled insertion")
                if not has_remove:
                    suggestions.append(f"Add 'remove_{element_name}()' method for controlled deletion")
                suggestions.append(f"Consider returning a copy: 'return self._{element_name}s.copy()'")
                if collection_type == "List":
                    suggestions.append(f"Or return immutable tuple: 'return tuple(self._{element_name}s)'")

                opportunity = {
                    "class": class_name,
                    "file": file,
                    "method": method_name,
                    "return_type": return_type,
                    "collection_type": collection_type,
                    "element_name": element_name,
                    "severity": severity,
                    "has_add_method": has_add,
                    "has_remove_method": has_remove,
                    "existing_modifiers": [m_name for _, m_name in modifiers],
                    "suggestions": suggestions,
                    "refactoring": "Encapsulate Collection (Fowler Chapter 7)",
                    "reason": f"Method '{method_name}' returns mutable {collection_type} without protective methods. "
                              f"Clients can directly modify the collection, breaking encapsulation."
                }

                opportunities.append(opportunity)

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "encapsulate_collection_opportunities": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "time_ms": time_ms,
                "note": "Requires type hints on return types. May miss collections returned without type annotations."
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "encapsulate_collection_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_encapsulate_field_opportunities(
        self,
        instance_name: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find public attributes that should be private with getters/setters.

        Fowler Chapter 7: Encapsulate Field

        Detection signals:
        - Public attributes (not starting with _) in non-dataclass classes
        - Classes with business logic (not simple data holders)
        - Suggests making field private and adding @property accessor

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return (pagination)
            offset: Number of results to skip (pagination)

        Returns:
            dict with success, encapsulate_field_opportunities list,
            pagination metadata, time_ms

        Example:
            # BEFORE (bad - direct field access)
            class BankAccount:
                def __init__(self):
                    self.balance = 0  # Public! Can be set to negative!

            # AFTER (good - encapsulated with validation)
            class BankAccount:
                def __init__(self):
                    self._balance = 0  # Protected

                @property
                def balance(self):
                    return self._balance

                @balance.setter
                def balance(self, value):
                    if value < 0:
                        raise ValueError("Balance cannot be negative")
                    self._balance = value
        """
        start_time = time.time()

        try:
            attr_concept = self._concept('Attribute')
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query: Find public attributes in classes with business logic
            # Use inFile (works for all languages) instead of inModule
            query_public_attrs = f"""
            SELECT ?attr ?attrName ?class ?className ?file ?type ?line
            WHERE {{
                ?attr type {attr_concept} .
                ?attr definedIn ?class .
                ?attr name ?attrName .
                ?attr visibility "public" .
                ?class type {class_concept} .
                ?class name ?className .
                ?class inFile ?file .
                OPTIONAL {{ ?attr hasType ?type }}
                OPTIONAL {{ ?attr atLine ?line }}
                FILTER NOT EXISTS {{
                    ?class hasDecorator "dataclass"
                }}
            }}
            ORDER BY ?className ?attrName
            """

            result = self.reter.reql(query_public_attrs)
            # Use padded list: 7 columns (OPTIONAL hasType and atLine may not be returned)
            public_attrs = self._query_to_list_padded(result, 7)

            # Single query to get all methods for all classes (excluding __init__)
            query_all_methods = f"""
            SELECT ?class ?methodName
            WHERE {{
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?method name ?methodName .
                FILTER(?methodName != "__init__")
            }}
            """
            all_methods_result = self.reter.reql(query_all_methods)
            all_methods = self._query_to_list(all_methods_result)

            # Build mapping: class_id -> list of method names
            class_to_methods: Dict[str, List[str]] = {}
            for class_id_m, method_name in all_methods:
                if class_id_m not in class_to_methods:
                    class_to_methods[class_id_m] = []
                class_to_methods[class_id_m].append(method_name)

            opportunities = []

            for attr_id, attr_name, class_id, class_name, file, attr_type, line in public_attrs:
                # Skip attributes that are already somewhat protected
                if attr_name.startswith('_'):
                    continue

                # Check if class has business logic (methods beyond __init__) using pre-fetched data
                methods = class_to_methods.get(class_id, [])

                # If class has business logic methods, this is a candidate
                if len(methods) > 0:
                    # Determine severity based on type and name
                    severity = "MEDIUM"
                    if attr_type and ("Manager" in attr_type or "Config" in attr_type or "State" in attr_type):
                        severity = "HIGH"  # Important infrastructure fields
                    elif any(keyword in attr_name.lower() for keyword in ["balance", "total", "count", "amount", "password", "token"]):
                        severity = "HIGH"  # Sensitive data fields

                    opportunity = {
                        "class": class_name,
                        "file": file,
                        "field": attr_name,
                        "qualified_name": attr_id,
                        "type": attr_type if attr_type else "Unknown",
                        "visibility": "public",
                        "line": int(line) if line else None,
                        "method_count": len(methods),
                        "severity": severity,
                        "refactoring": "Encapsulate Field (Fowler Chapter 7)",
                        "reason": "Public mutable field in class with business logic",
                        "suggestion": (
                            f"Make '{attr_name}' private and add property accessor:\n"
                            f"1. Rename field: self.{attr_name} -> self._{attr_name}\n"
                            f"2. Add @property getter:\n"
                            f"   @property\n"
                            f"   def {attr_name}(self):\n"
                            f"       return self._{attr_name}\n"
                            f"3. Add setter with validation if needed:\n"
                            f"   @{attr_name}.setter\n"
                            f"   def {attr_name}(self, value):\n"
                            f"       # Add validation here\n"
                            f"       self._{attr_name} = value"
                        )
                    }

                    opportunities.append(opportunity)

            # Sort by severity
            severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            opportunities.sort(
                key=lambda x: (-severity_order[x["severity"]], x["class"], x["field"])
            )

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "encapsulate_field_opportunities": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "time_ms": time_ms,
                "note": "Excludes @dataclass classes and fields starting with underscore."
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "encapsulate_field_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_encapsulate_record_opportunities(
        self,
        instance_name: str,
        min_accesses: int = 5,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect dict/record usage that should be encapsulated in a class.

        Based on Fowler Chapter 7: Encapsulate Record

        Detection signals:
        - Variables assigned dict literals (e.g., data = {"key": value})
        - Functions with many dict-literal assignments
        - Dicts with multiple keys suggest a structured record

        Args:
            instance_name: RETER instance name
            min_accesses: Minimum number of dict-related assignments to flag
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            Dict with encapsulate_record_opportunities

        Example:
            # BEFORE (bad - unstructured dict)
            person = {"name": "John", "age": 30, "email": "john@example.com"}
            print(person["name"])

            # AFTER (good - encapsulated in class)
            @dataclass
            class Person:
                name: str
                age: int
                email: str

            person = Person("John", 30, "john@example.com")
            print(person.name)
        """
        start_time = time.time()

        try:
            assign_concept = self._concept('Assignment')

            # Find variables assigned multiple times in functions (record-like patterns)
            # Note: Direct dict literal detection would require C++ AST to track isDictLiteral
            # Use inFile (works for all languages) instead of inModule
            query_multi_assigns = f"""
            SELECT ?func ?funcName ?file ?target (COUNT(?assign) AS ?count)
            WHERE {{
                ?assign type {assign_concept} .
                ?assign inFunction ?func .
                ?assign target ?target .
                ?func name ?funcName .
                ?func inFile ?file .
            }}
            GROUP BY ?func ?funcName ?file ?target
            HAVING (?count >= {min_accesses})
            ORDER BY DESC(?count)
            LIMIT 200
            """

            result = self.reter.reql(query_multi_assigns)
            multi_assigns = self._query_to_list(result)

            # Group by function
            func_data: Dict[str, Dict] = {}
            for func_id, func_name, file, target, count in multi_assigns:
                count = int(count)
                if func_id not in func_data:
                    func_data[func_id] = {
                        "func_name": func_name,
                        "file": file,
                        "targets": {}
                    }
                func_data[func_id]["targets"][target] = count

            opportunities = []

            # Create opportunities for functions with reassigned variables
            for func_id, data in func_data.items():
                total_reassigns = sum(data["targets"].values())
                num_vars = len(data["targets"])

                severity = "HIGH" if total_reassigns >= 15 or num_vars >= 5 else \
                           "MEDIUM" if total_reassigns >= 8 else "LOW"

                details = [
                    {"variable": var, "assignment_count": cnt}
                    for var, cnt in sorted(data["targets"].items(), key=lambda x: -x[1])
                ]

                opportunity = {
                    "location_type": "function",
                    "function": data["func_name"],
                    "file": data["file"],
                    "total_reassignments": total_reassigns,
                    "variable_count": num_vars,
                    "reassigned_variables": list(data["targets"].keys()),
                    "details": details,
                    "severity": severity,
                    "refactoring": "Encapsulate Record (Fowler Chapter 7)",
                    "reason": f"Function '{data['func_name']}' has {num_vars} variable(s) reassigned {total_reassigns} times",
                    "suggestions": [
                        f"Create a @dataclass named '{self._suggest_class_name_from_func(data['func_name'])}' to group related data",
                        "Multiple assignments to same variable may indicate building a record",
                        "Consider using a structured object instead of reassigning variables",
                        "Replace dict/variable mutation with immutable records"
                    ]
                }
                opportunities.append(opportunity)

            # Sort by severity
            severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            opportunities.sort(
                key=lambda x: (-severity_order.get(x["severity"], 0), x.get("function", x.get("module", "")))
            )

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "encapsulate_record_opportunities": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "time_ms": time_ms,
                "note": "Detects dict literal assignments that could benefit from @dataclass or named classes."
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "encapsulate_record_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def _suggest_class_name_from_func(self, func_name: str) -> str:
        """
        Suggest a class name based on a single function name.

        Used for Encapsulate Record refactoring suggestions.

        Examples:
            process_user_data -> UserData
            get_config -> Config
            _build_response -> Response
        """
        # Remove common prefixes
        name = func_name
        for prefix in ["process_", "get_", "create_", "build_", "make_", "_", "__"]:
            if name.startswith(prefix):
                name = name[len(prefix):]

        # Remove common suffixes
        for suffix in ["_data", "_dict", "_config", "_info", "_result"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

        # Convert to PascalCase
        parts = name.split("_")
        class_name = "".join(part.capitalize() for part in parts if part)

        # Add "Data" suffix if name is too short
        if len(class_name) < 4:
            class_name += "Data"

        return class_name or "RecordData"

    def _extract_element_name(self, method_name: str) -> str:
        """
        Extract element name from getter method name.

        Examples:
            get_courses -> course
            get_items -> item
            students -> student
            getCourses -> course
        """
        # Remove common prefixes
        name = method_name
        for prefix in ["get_", "fetch_", "retrieve_", "load_", "list_"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Handle camelCase (getCourses -> Courses -> courses)
        if method_name[0].islower() and any(c.isupper() for c in method_name):
            # Find first uppercase letter
            for i, c in enumerate(method_name):
                if c.isupper():
                    name = method_name[i:].lower()
                    break

        # Singularize (simple approach)
        if name.endswith("ses"):  # courses -> course
            return name[:-2]
        elif name.endswith("ies"):  # entries -> entry
            return name[:-3] + "y"
        elif name.endswith("s") and len(name) > 1:  # items -> item
            return name[:-1]
        else:
            return name

    def find_hide_delegate_opportunities(
        self,
        instance_name: str,
        min_client_calls: int = 3,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect classes that should add delegating methods (inverse of Middle Man).

        Fowler Chapter 7: Hide Delegate (pragmatic version)

        Detection signals:
        - Class has dependencies (calls other classes)
        - Clients frequently access those dependencies
        - No delegating methods exist to hide the dependencies
        - High coupling to external classes

        This is the INVERSE of Remove Middle Man:
        - Middle Man: Too much delegation → remove it
        - Hide Delegate: No delegation + high coupling → add it

        Args:
            instance_name: RETER instance name
            min_client_calls: Minimum client calls to flag (default: 3)
            limit: Maximum results to return (pagination)
            offset: Number of results to skip (pagination)

        Returns:
            dict with success, hide_delegate_opportunities list,
            pagination metadata, time_ms

        Example:
            # BEFORE (Law of Demeter violation)
            class Person:
                def __init__(self):
                    self.department = Department()

            # Client code (bad - knows about department)
            manager = person.department.manager
            budget = person.department.get_budget()

            # AFTER (Hide Delegate)
            class Person:
                def get_manager(self):
                    return self.department.manager

                def get_department_budget(self):
                    return self.department.get_budget()

            # Client code (good - simpler)
            manager = person.get_manager()
            budget = person.get_department_budget()
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            call_concept = self._concept('Call')
            method_concept = self._concept('Method')

            # Query 1: Find classes and their external dependencies
            # Use inFile (works for all languages) instead of inModule
            query_dependencies = f"""
            SELECT ?class ?className ?file ?targetClass ?targetClassName (COUNT(?call) AS ?callCount)
            WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?class inFile ?file .
                ?call type {call_concept} .
                ?call fromClass ?class .
                ?call toClass ?targetClass .
                ?targetClass type {class_concept} .
                ?targetClass name ?targetClassName .
                FILTER(?class != ?targetClass)
            }}
            GROUP BY ?class ?className ?file ?targetClass ?targetClassName
            ORDER BY DESC(?callCount)
            """

            result = self.reter.reql(query_dependencies)
            dependencies = self._query_to_list(result)

            # Query 2: Find classes with client relationships (who calls whom)
            query_clients = f"""
            SELECT ?class ?className ?clientClass ?clientClassName (COUNT(?call) AS ?clientCallCount)
            WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?call type {call_concept} .
                ?call toClass ?class .
                ?call fromClass ?clientClass .
                ?clientClass type {class_concept} .
                ?clientClass name ?clientClassName .
                FILTER(?class != ?clientClass)
            }}
            GROUP BY ?class ?className ?clientClass ?clientClassName
            ORDER BY DESC(?clientCallCount)
            """

            client_result = self.reter.reql(query_clients)
            client_relationships = self._query_to_list(client_result)

            # Build client map: class -> list of clients
            client_map = {}
            for class_id, class_name, client_class_id, client_name, call_count in client_relationships:
                if class_name not in client_map:
                    client_map[class_name] = []
                client_map[class_name].append({
                    "client": client_name,
                    "call_count": int(call_count)
                })

            # Query 3: Check for delegation methods (methods that just call other classes)
            # A delegating method typically has a single call to another class
            query_delegating_methods = f"""
            SELECT ?method ?methodName ?class ?className
            WHERE {{
                ?method type {method_concept} .
                ?method name ?methodName .
                ?method inClass ?class .
                ?class name ?className .
                ?call type {call_concept} .
                ?call fromFunction ?method .
            }}
            """

            delegation_result = self.reter.reql(query_delegating_methods)
            all_methods_with_calls = self._query_to_list(delegation_result)

            # Build map: class -> count of methods with calls
            delegation_map = {}
            for method_id, method_name, class_id, class_name in all_methods_with_calls:
                delegation_map[class_name] = delegation_map.get(class_name, 0) + 1

            # Analyze opportunities
            opportunities = []

            for class_id, class_name, file, target_class_id, target_class, call_count in dependencies:
                call_count = int(call_count)

                # Get clients of this class
                clients = client_map.get(class_name, [])
                client_count = len(clients)

                # Check if this class has delegating methods
                delegation_count = delegation_map.get(class_name, 0)

                # Heuristic: Suggest Hide Delegate if:
                # 1. Class has external dependencies (call_count > 0)
                # 2. Class has multiple clients (client_count >= min_client_calls)
                # 3. Class has few or no delegating methods (delegation_count < 3)

                if call_count > 0 and client_count >= min_client_calls and delegation_count < 3:
                    severity = "HIGH" if delegation_count == 0 else "MEDIUM"

                    opportunity = {
                        "class": class_name,
                        "file": file,
                        "target_dependency": target_class,
                        "dependency_call_count": call_count,
                        "client_count": client_count,
                        "clients": [c["client"] for c in clients[:5]],  # Top 5 clients
                        "delegation_method_count": delegation_count,
                        "severity": severity,
                        "refactoring": "Hide Delegate (Fowler Chapter 7)",
                        "reason": f"Class '{class_name}' has {client_count} clients and {call_count} calls to '{target_class}', "
                                  f"but only {delegation_count} delegating methods. "
                                  f"Consider adding delegation methods to hide '{target_class}' from clients.",
                        "suggestions": [
                            f"Add delegation methods in '{class_name}' to hide '{target_class}'",
                            f"This reduces coupling between clients and '{target_class}'",
                            f"Clients currently depend on '{class_name}' AND '{target_class}' (Law of Demeter violation)",
                            f"Example: Add methods like 'get_X()' that delegate to '{target_class}.get_X()'"
                        ]
                    }

                    opportunities.append(opportunity)

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "hide_delegate_opportunities": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "time_ms": time_ms,
                "note": "Pragmatic heuristic approach. True Law of Demeter violations require tracking attribute access chains."
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "hide_delegate_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_split_loop_opportunities(
        self,
        instance_name: str,
        min_operations: int = 2,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find loops doing multiple things (Fowler Chapter 8: Split Loop).

        Placeholder implementation - returns empty results.
        Full implementation requires control flow analysis.
        """
        start_time = time.time()
        try:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "split_loop_opportunities": [],
                "count": 0,
                "total_count": 0,
                "has_more": False,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": None
                },
                "time_ms": time_ms,
                "note": "Placeholder implementation. Full loop analysis requires AST-level control flow analysis."
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "split_loop_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_pipeline_conversion_opportunities(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find loops replaceable with collection pipelines (Fowler Chapter 8).

        Placeholder implementation - returns empty results.
        Full implementation requires control flow and pattern analysis.
        """
        start_time = time.time()
        try:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "pipeline_conversion_opportunities": [],
                "count": 0,
                "total_count": 0,
                "has_more": False,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": None
                },
                "time_ms": time_ms,
                "note": "Placeholder implementation. Full pipeline detection requires AST-level pattern matching."
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "pipeline_conversion_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_move_function_opportunities(
        self,
        instance_name: str,
        coupling_threshold: float = 0.5,
        min_external_refs: int = 5,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find functions that should move to a different class (Fowler Chapter 8: Move Function).

        Placeholder implementation - returns empty results.
        Full implementation requires coupling analysis.
        """
        start_time = time.time()
        try:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "move_function_opportunities": [],
                "count": 0,
                "total_count": 0,
                "has_more": False,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": None
                },
                "time_ms": time_ms,
                "note": "Placeholder implementation. Full analysis requires coupling metrics and class cohesion analysis."
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "move_function_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_move_field_opportunities(
        self,
        instance_name: str,
        access_ratio_threshold: float = 0.6,
        min_external_accesses: int = 3,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find fields that should move to a different class (Fowler Chapter 8: Move Field).

        Placeholder implementation - returns empty results.
        Full implementation requires field access pattern analysis.
        """
        start_time = time.time()
        try:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "move_field_opportunities": [],
                "count": 0,
                "total_count": 0,
                "has_more": False,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": None
                },
                "time_ms": time_ms,
                "note": "Placeholder implementation. Full analysis requires attribute access tracking and coupling analysis."
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "move_field_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

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

        Based on Martin Fowler's "Refactoring" Chapter 9: Split Variable.

        Args:
            instance_name: RETER instance name
            min_assignments: Minimum number of assignments to flag (default: 2)
            include_loop_vars: Whether to include loop variables (default: False)
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            Dict with split variable opportunities:
            {
                "success": bool,
                "split_variable_opportunities": [
                    {
                        "function": str,              # Function ID
                        "function_name": str,         # Function name
                        "variable_name": str,         # Variable that should be split
                        "assignment_count": int,      # Number of assignments
                        "assignment_locations": list, # Where variable is assigned
                        "is_augmented": bool,         # Is it a collecting variable (+=, etc)
                        "severity": str,              # LOW/MEDIUM/HIGH
                        "refactoring": str,
                        "suggestion": str
                    }
                ],
                "count": int,
                "total_count": int
            }
        """
        start_time = time.time()

        try:
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')
            assign_concept = self._concept('Assignment')

            # OPTIMIZATION: Use GROUP BY to pre-filter (func, target) pairs with multiple assignments
            # This avoids fetching all assignments just to filter them in Python

            # Phase 1: Find (func, target) pairs with count >= min_assignments using GROUP BY
            query_func_counts = f"""
            SELECT ?func ?target (COUNT(?assignment) AS ?count)
            WHERE {{
                ?func type {func_concept} .
                ?assignment type {assign_concept} .
                ?assignment inFunction ?func .
                ?assignment target ?target .
            }}
            GROUP BY ?func ?target
            """

            query_method_counts = f"""
            SELECT ?func ?target (COUNT(?assignment) AS ?count)
            WHERE {{
                ?func type {method_concept} .
                ?assignment type {assign_concept} .
                ?assignment inFunction ?func .
                ?assignment target ?target .
            }}
            GROUP BY ?func ?target
            """

            # Run aggregation queries (fast)
            result_func_counts = self.reter.reql(query_func_counts)
            result_method_counts = self.reter.reql(query_method_counts)

            # Build set of (func, target) pairs that meet the threshold
            candidate_pairs = set()
            for row in self._query_to_list(result_func_counts) + self._query_to_list(result_method_counts):
                func_id, target, count = row
                if int(count) >= min_assignments:
                    candidate_pairs.add((func_id, target))

            # If no candidates, return early
            if not candidate_pairs:
                return {
                    "success": True,
                    "split_variable_opportunities": [],
                    "count": 0,
                    "total_count": 0,
                    "time_ms": (time.time() - start_time) * 1000
                }

            # Phase 2: Get details only for candidate pairs
            # Add LIMIT to prevent huge result sets - we filter by candidate_pairs anyway
            max_assignments = 5000  # Safety limit

            # Query for function assignments
            # Use inFile (works for all languages) instead of inModule
            query_functions = f"""
            SELECT ?func ?funcName ?file ?assignment ?target ?line
            WHERE {{
                ?func type {func_concept} .
                ?func name ?funcName .
                ?func inFile ?file .
                ?assignment type {assign_concept} .
                ?assignment inFunction ?func .
                ?assignment target ?target .
                ?assignment atLine ?line .
            }}
            ORDER BY ?func ?target ?line
            LIMIT {max_assignments}
            """

            # Query for method assignments
            # Use inFile (works for all languages) instead of inModule
            query_methods = f"""
            SELECT ?func ?funcName ?file ?assignment ?target ?line
            WHERE {{
                ?func type {method_concept} .
                ?func name ?funcName .
                ?func inFile ?file .
                ?assignment type {assign_concept} .
                ?assignment inFunction ?func .
                ?assignment target ?target .
                ?assignment atLine ?line .
            }}
            ORDER BY ?func ?target ?line
            LIMIT {max_assignments}
            """

            # Query for augmented assignments
            query_augmented = f"""
            SELECT ?assignment
            WHERE {{
                ?assignment type {assign_concept} .
                ?assignment isAugmented "true" .
            }}
            """

            # Run queries
            result1 = self.reter.reql(query_functions)
            result2 = self.reter.reql(query_methods)
            result_augmented = self.reter.reql(query_augmented)

            # Build set of augmented assignment IDs for fast lookup
            augmented_set = set()
            for row in self._query_to_list(result_augmented):
                augmented_set.add(row[0])

            # Combine and FILTER assignment results - only keep candidates
            raw_assignments = self._query_to_list(result1) + self._query_to_list(result2)

            # Transform to expected format with isAugmented, filtering by candidate pairs
            assignments = []
            for func_id, func_name, file_path, assignment_id, target, line in raw_assignments:
                # Skip non-candidate pairs (major optimization)
                if (func_id, target) not in candidate_pairs:
                    continue
                is_augmented = assignment_id in augmented_set
                assignments.append((func_id, func_name, file_path, target, line, is_augmented))

            # Group assignments by function and variable
            from collections import defaultdict
            var_assignments = defaultdict(lambda: {
                "func_id": None,
                "func_name": None,
                "file": None,
                "assignments": []
            })

            for func_id, func_name, file_path, target, line, is_augmented in assignments:
                key = (func_id, target)
                if var_assignments[key]["func_id"] is None:
                    var_assignments[key]["func_id"] = func_id
                    var_assignments[key]["func_name"] = func_name
                    var_assignments[key]["file"] = file_path

                var_assignments[key]["assignments"].append({
                    "line": int(line),
                    "is_augmented": is_augmented  # Already a boolean from our processing
                })

            # Find split variable opportunities
            opportunities = []

            for (func_id, var_name), data in var_assignments.items():
                assignment_count = len(data["assignments"])

                # Skip if below threshold
                if assignment_count < min_assignments:
                    continue

                # Check if it's a collecting variable (all assignments are augmented)
                augmented_count = sum(1 for a in data["assignments"] if a["is_augmented"])
                is_collecting_var = augmented_count == assignment_count

                # Skip collecting variables (accumulators like total += x)
                if is_collecting_var:
                    continue

                # Check if any assignments are augmented (mixed usage)
                has_some_augmented = augmented_count > 0

                # Build assignment location strings
                assignment_locations = [
                    f"{data['file']}:{a['line']}"
                    for a in sorted(data["assignments"], key=lambda x: x["line"])
                ]

                # Determine severity
                if assignment_count >= 5:
                    severity = "HIGH"
                elif assignment_count >= 3:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"

                # Create suggestion
                if has_some_augmented:
                    suggestion = (
                        f"Variable '{var_name}' in function '{data['func_name']}' is assigned {assignment_count} times "
                        f"with mixed usage (some augmented assignments). Consider splitting into separate variables "
                        f"for each distinct purpose."
                    )
                else:
                    suggestion = (
                        f"Variable '{var_name}' in function '{data['func_name']}' is assigned {assignment_count} times "
                        f"with different values. Each assignment likely represents a different responsibility. "
                        f"Consider splitting into separate variables (e.g., 'primaryAcc' and 'secondaryAcc')."
                    )

                opportunity = {
                    "function": func_id,
                    "function_name": data["func_name"],
                    "module": data["module"],
                    "variable_name": var_name,
                    "assignment_count": assignment_count,
                    "assignment_locations": assignment_locations,
                    "is_augmented": is_collecting_var,
                    "has_mixed_usage": has_some_augmented and not is_collecting_var,
                    "severity": severity,
                    "refactoring": "Split Variable",
                    "suggestion": suggestion,
                    "fowler_reference": "Refactoring (2nd Edition), Chapter 9: Split Variable, p.240"
                }

                opportunities.append(opportunity)

            # Sort by severity (HIGH > MEDIUM > LOW) then by assignment count
            severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            opportunities.sort(key=lambda x: (severity_order[x["severity"]], -x["assignment_count"]))

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "split_variable_opportunities": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "split_variable_opportunities": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    # =========================================================================
    # MESSAGE CHAINS (Fowler Chapter 3)
    # =========================================================================

    def find_message_chains(
        self,
        instance_name: str,
        min_chain_length: int = 3,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect long method call chains (Law of Demeter violations).

        Message chains occur when code calls through a long chain of objects:
        a.b().c().d(). This creates coupling and makes code brittle.

        Example:
            customer.getAddress().getStreet().getName()

        Refactoring: Hide Delegate or Extract Method

        Args:
            instance_name: RETER instance name
            min_chain_length: Minimum chain depth to flag (default: 3)
            limit: Max results to return
            offset: Pagination offset

        Returns:
            dict with message_chains list and metadata
        """
        start_time = time.time()

        try:
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')

            # Query all call relationships
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?caller ?callerName ?callerFile ?callee ?calleeName
                WHERE {{
                    ?caller type ?callerType .
                    ?caller name ?callerName .
                    ?caller inFile ?callerFile .
                    ?caller calls ?callee .
                    ?callee type ?calleeType .
                    ?callee name ?calleeName .
                    FILTER(?callerType = "{func_concept}" || ?callerType = "{method_concept}")
                    FILTER(?calleeType = "{func_concept}" || ?calleeType = "{method_concept}")
                }}
            """

            result = self.reter.reql(query)
            calls = self._query_to_list(result)

            # Build call graph
            call_graph = defaultdict(set)
            function_names = {}
            function_files = {}

            for caller, caller_name, caller_file, callee, callee_name in calls:
                call_graph[caller].add(callee)
                function_names[caller] = caller_name
                function_names[callee] = callee_name
                function_files[caller] = caller_file

            # Find chains using DFS
            chains = []

            def find_chains_from(start, path, visited):
                """Recursively find call chains."""
                if len(path) >= min_chain_length:
                    chains.append(list(path))

                if len(path) >= 10:  # Prevent infinite recursion
                    return

                current = path[-1]
                for next_func in call_graph.get(current, []):
                    if next_func not in visited:
                        visited.add(next_func)
                        find_chains_from(start, path + [next_func], visited)
                        visited.remove(next_func)

            # Start DFS from each function
            for func in call_graph:
                visited = {func}
                find_chains_from(func, [func], visited)

            # Convert to opportunity format
            opportunities = []
            seen_chains = set()

            for chain in chains:
                chain_key = tuple(chain)
                if chain_key in seen_chains:
                    continue
                seen_chains.add(chain_key)

                chain_length = len(chain)
                root_func = chain[0]

                # Determine severity
                if chain_length >= 5:
                    severity = "HIGH"
                elif chain_length >= 4:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"

                chain_names = " → ".join([function_names.get(f, "?") for f in chain])

                opportunity = {
                    "root_function": root_func,
                    "root_function_name": function_names.get(root_func, "?"),
                    "file": function_files.get(root_func, "?"),
                    "chain_length": chain_length,
                    "chain": [function_names.get(f, "?") for f in chain],
                    "chain_display": chain_names,
                    "severity": severity,
                    "refactoring": "Hide Delegate / Extract Method",
                    "suggestion": (
                        f"Function '{function_names.get(root_func)}' has a call chain of depth {chain_length} "
                        f"({chain_names}). Consider hiding intermediary calls by adding delegate methods "
                        f"or extracting the chain into a well-named method."
                    ),
                    "fowler_reference": "Refactoring (2nd Edition), Chapter 3: Message Chains, p.81"
                }
                opportunities.append(opportunity)

            # Sort by severity and chain length
            severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            opportunities.sort(key=lambda x: (severity_order[x["severity"]], -x["chain_length"]))

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "message_chains": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "message_chains": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    # =========================================================================
    # GLOBAL DATA (Fowler Chapter 3)
    # =========================================================================

    def find_global_data(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect module-level mutable assignments (global data).

        Global data is one of the most insidious code smells. Module-level
        variables that are assigned (not just constants) create hidden
        coupling and make testing difficult.

        Example:
            # At module level
            user_cache = {}
            request_count = 0

        Refactoring: Encapsulate Variable

        Args:
            instance_name: RETER instance name
            limit: Max results to return
            offset: Pagination offset

        Returns:
            dict with global_data list and metadata
        """
        start_time = time.time()

        try:
            assign_concept = self._concept('Assignment')

            # Query assignments NOT in any function or class
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?assignment ?target ?value ?file ?line
                WHERE {{
                    ?assignment type {assign_concept} .
                    ?assignment target ?target .
                    ?assignment value ?value .
                    ?assignment inFile ?file .
                    ?assignment atLine ?line .
                    FILTER NOT EXISTS {{ ?assignment inFunction ?func }}
                    FILTER NOT EXISTS {{ ?assignment inClass ?cls }}
                }}
                ORDER BY ?file ?line
            """

            result = self.reter.reql(query)
            assignments = self._query_to_list(result)

            # Group by module and variable
            opportunities = []

            for assignment_id, target, value, file, line in assignments:
                # Heuristics for identifying constants (should be excluded)
                is_likely_constant = (
                    target.isupper() or  # CONSTANT_NAME
                    value in ['None', 'True', 'False'] or
                    (value.startswith('"') or value.startswith("'")) or  # String literal
                    (value.isdigit())  # Numeric literal
                )

                # Determine severity
                if not is_likely_constant:
                    severity = "HIGH"  # Mutable global data
                else:
                    severity = "LOW"   # Likely a constant, but still module-level

                opportunity = {
                    "file": file,
                    "variable_name": target,
                    "initial_value": value,
                    "line": line,
                    "is_likely_constant": is_likely_constant,
                    "severity": severity,
                    "refactoring": "Encapsulate Variable",
                    "suggestion": (
                        f"Module-level {'constant' if is_likely_constant else 'variable'} '{target}' in module '{module}'. "
                        + ("Consider making this a true constant (immutable) or encapsulating it." if is_likely_constant
                           else "Global mutable data! Encapsulate this in a class or function to avoid hidden coupling.")
                    ),
                    "fowler_reference": "Refactoring (2nd Edition), Chapter 3: Global Data, p.74"
                }
                opportunities.append(opportunity)

            # Sort by severity (mutable first)
            severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            opportunities.sort(key=lambda x: (severity_order[x["severity"]], x["module"], x["line"]))

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "global_data": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "global_data": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    # =========================================================================
    # SPECULATIVE GENERALITY (Fowler Chapter 3)
    # =========================================================================

    def find_speculative_generality(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect abstract classes with only 1 subclass (over-engineering).

        Speculative generality occurs when abstractions are created "just in case"
        we need them in the future, but currently serve no purpose.

        Example:
            class AbstractPaymentProcessor(ABC):  # Only 1 subclass
                pass
            class StripePaymentProcessor(AbstractPaymentProcessor):
                pass

        Refactoring: Collapse Hierarchy / Inline Class

        Args:
            instance_name: RETER instance name
            limit: Max results to return
            offset: Pagination offset

        Returns:
            dict with speculative_generality list and metadata
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')

            # Query all inheritance relationships
            query = f"""
                SELECT ?parent ?parentName ?child ?childName
                WHERE {{
                    ?parent type {class_concept} .
                    ?parent name ?parentName .
                    ?child type {class_concept} .
                    ?child name ?childName .
                    ?child inheritsFrom ?parent .
                }}
            """

            result = self.reter.reql(query)
            inheritance = self._query_to_list(result)

            # Count subclasses per parent
            subclass_count = defaultdict(list)
            parent_names = {}

            for parent, parent_name, child, child_name in inheritance:
                subclass_count[parent].append((child, child_name))
                parent_names[parent] = parent_name

            # Find parents with exactly 1 subclass
            opportunities = []

            for parent, subclasses in subclass_count.items():
                if len(subclasses) == 1:
                    child, child_name = subclasses[0]
                    parent_name = parent_names[parent]

                    # Check if parent is likely an abstract class
                    is_likely_abstract = (
                        "Abstract" in parent_name or
                        "Base" in parent_name or
                        "Interface" in parent_name or
                        parent_name.startswith("I") and parent_name[1].isupper()  # IPayment pattern
                    )

                    severity = "MEDIUM" if is_likely_abstract else "LOW"

                    opportunity = {
                        "parent_class": parent,
                        "parent_name": parent_name,
                        "child_class": child,
                        "child_name": child_name,
                        "is_likely_abstract": is_likely_abstract,
                        "severity": severity,
                        "refactoring": "Collapse Hierarchy / Inline Class",
                        "suggestion": (
                            f"Class '{parent_name}' has only 1 subclass ('{child_name}'). "
                            f"This abstraction may be premature. Consider collapsing the hierarchy "
                            f"unless you have concrete plans to add more subclasses soon."
                        ),
                        "fowler_reference": "Refactoring (2nd Edition), Chapter 3: Speculative Generality, p.85"
                    }
                    opportunities.append(opportunity)

            # Sort by severity and name
            severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            opportunities.sort(key=lambda x: (severity_order[x["severity"]], x["parent_name"]))

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "speculative_generality": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "speculative_generality": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    # =========================================================================
    # PARALLEL INHERITANCE HIERARCHIES (Fowler Chapter 3)
    # =========================================================================

    def find_parallel_inheritance_hierarchies(
        self,
        instance_name: str,
        min_similarity: float = 0.6,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect class hierarchies that mirror each other (parallel hierarchies).

        When you see class pairs like OrderProcessor/OrderValidator,
        ShippingProcessor/ShippingValidator, they likely should be merged
        or share a common abstraction.

        Example:
            OrderProcessor → ShippingOrderProcessor
            OrderValidator → ShippingOrderValidator

        Refactoring: Move Method / Consolidate Hierarchies

        Args:
            instance_name: RETER instance name
            min_similarity: Name similarity threshold (0-1)
            limit: Max results to return
            offset: Pagination offset

        Returns:
            dict with parallel_hierarchies list and metadata
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')

            # Query all inheritance relationships
            query = f"""
                SELECT ?parent ?parentName ?child ?childName
                WHERE {{
                    ?parent type {class_concept} .
                    ?parent name ?parentName .
                    ?child type {class_concept} .
                    ?child name ?childName .
                    ?child inheritsFrom ?parent .
                }}
            """

            result = self.reter.reql(query)
            inheritance = self._query_to_list(result)

            # Build hierarchy trees
            hierarchies = defaultdict(list)
            parent_names = {}

            for parent, parent_name, child, child_name in inheritance:
                hierarchies[parent].append((child, child_name))
                parent_names[parent] = parent_name

            # Find hierarchy pairs with similar naming patterns
            def extract_suffix(name):
                """Extract common suffix patterns like Processor, Validator, etc."""
                common_suffixes = ["Processor", "Validator", "Handler", "Manager", "Service",
                                   "Controller", "Factory", "Builder", "Strategy", "Observer"]
                for suffix in common_suffixes:
                    if name.endswith(suffix):
                        return suffix
                return None

            def name_similarity(name1, name2):
                """Calculate simple name similarity (0-1)."""
                # Remove common suffixes
                suffix1 = extract_suffix(name1)
                suffix2 = extract_suffix(name2)

                if suffix1 and suffix2 and suffix1 == suffix2:
                    # Same suffix pattern
                    prefix1 = name1[:-len(suffix1)]
                    prefix2 = name2[:-len(suffix2)]

                    # Check if prefixes are similar
                    if prefix1 == prefix2:
                        return 1.0

                    # Levenshtein-like simple check
                    common = sum(1 for a, b in zip(prefix1, prefix2) if a == b)
                    return common / max(len(prefix1), len(prefix2))

                return 0.0

            opportunities = []
            seen_pairs = set()

            # Compare all hierarchy pairs
            hierarchy_list = list(hierarchies.items())
            for i, (parent1, children1) in enumerate(hierarchy_list):
                for parent2, children2 in hierarchy_list[i+1:]:
                    name1 = parent_names.get(parent1, "")
                    name2 = parent_names.get(parent2, "")

                    # Check if parent names have similar patterns
                    similarity = name_similarity(name1, name2)

                    if similarity >= min_similarity:
                        # Check if children also follow the pattern
                        matching_children = 0
                        for child1, child_name1 in children1:
                            for child2, child_name2 in children2:
                                if name_similarity(child_name1, child_name2) >= min_similarity:
                                    matching_children += 1

                        if matching_children > 0:
                            pair_key = tuple(sorted([parent1, parent2]))
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)

                                severity = "MEDIUM" if matching_children >= 2 else "LOW"

                                opportunity = {
                                    "hierarchy1_root": parent1,
                                    "hierarchy1_name": name1,
                                    "hierarchy1_children": [c[1] for c in children1],
                                    "hierarchy2_root": parent2,
                                    "hierarchy2_name": name2,
                                    "hierarchy2_children": [c[1] for c in children2],
                                    "similarity_score": round(similarity, 2),
                                    "matching_subclasses": matching_children,
                                    "severity": severity,
                                    "refactoring": "Move Method / Consolidate Hierarchies",
                                    "suggestion": (
                                        f"Class hierarchies '{name1}' and '{name2}' appear to be parallel "
                                        f"(similarity: {similarity:.1%}, {matching_children} matching subclass patterns). "
                                        f"Consider consolidating these hierarchies or moving methods to eliminate duplication."
                                    ),
                                    "fowler_reference": "Refactoring (2nd Edition), Chapter 3: Alternative Classes with Different Interfaces, p.83"
                                }
                                opportunities.append(opportunity)

            # Sort by matching children and similarity
            opportunities.sort(key=lambda x: (-x["matching_subclasses"], -x["similarity_score"]))

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "parallel_hierarchies": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "parallel_hierarchies": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    # =========================================================================
    # MUTABLE DATA ACROSS FUNCTIONS (Cross-function coupling)
    # =========================================================================

    def find_mutable_data_across_functions(
        self,
        instance_name: str,
        min_functions: int = 3,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect variables assigned in multiple different functions.

        When the same variable name is assigned in many functions, it suggests
        shared mutable state or poorly scoped data access.

        Example:
            def func1(): user_cache[id] = user
            def func2(): user_cache.clear()
            def func3(): del user_cache[id]

        Refactoring: Encapsulate Variable / Move to Class

        Args:
            instance_name: RETER instance name
            min_functions: Minimum functions assigning the variable
            limit: Max results to return
            offset: Pagination offset

        Returns:
            dict with mutable_data list and metadata
        """
        start_time = time.time()

        try:
            assign_concept = self._concept('Assignment')
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')

            # Query assignments in functions
            # Use inFile (works for all languages) instead of inModule
            query_functions = f"""
                SELECT ?assignment ?target ?func ?funcName ?file ?line
                WHERE {{
                    ?assignment type {assign_concept} .
                    ?assignment target ?target .
                    ?assignment inFunction ?func .
                    ?assignment atLine ?line .
                    ?func type {func_concept} .
                    ?func name ?funcName .
                    ?func inFile ?file
                }}
            """

            # Query assignments in methods
            # Use inFile (works for all languages) instead of inModule
            query_methods = f"""
                SELECT ?assignment ?target ?func ?funcName ?file ?line
                WHERE {{
                    ?assignment type {assign_concept} .
                    ?assignment target ?target .
                    ?assignment inFunction ?func .
                    ?assignment atLine ?line .
                    ?func type {method_concept} .
                    ?func name ?funcName .
                    ?func inFile ?file
                }}
            """

            # Run both queries and combine results
            result_functions = self.reter.reql(query_functions)
            result_methods = self.reter.reql(query_methods)

            assignments = self._query_to_list(result_functions) + self._query_to_list(result_methods)
            query = query_functions  # For logging

            # Group by variable name
            var_usage = defaultdict(lambda: {"functions": set(), "assignments": []})

            for assignment_id, target, func, func_name, file, line in assignments:
                var_usage[target]["functions"].add((func, func_name, file))
                var_usage[target]["assignments"].append({
                    "function": func_name,
                    "file": file,
                    "line": line
                })

            # Find variables used in multiple functions
            opportunities = []

            for var_name, data in var_usage.items():
                func_count = len(data["functions"])

                if func_count >= min_functions:
                    # Determine severity
                    if func_count >= 5:
                        severity = "HIGH"
                    elif func_count >= 4:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    functions_list = [f[1] for f in sorted(data["functions"])]

                    opportunity = {
                        "variable_name": var_name,
                        "function_count": func_count,
                        "functions": functions_list,
                        "assignment_count": len(data["assignments"]),
                        "assignments": data["assignments"][:10],  # Limit to first 10
                        "severity": severity,
                        "refactoring": "Encapsulate Variable / Move to Class",
                        "suggestion": (
                            f"Variable '{var_name}' is assigned in {func_count} different functions. "
                            f"This suggests shared mutable state. Consider encapsulating this in a class "
                            f"or making the data flow explicit through parameters and return values."
                        ),
                        "fowler_reference": "Refactoring (2nd Edition), Chapter 3: Mutable Data, p.75"
                    }
                    opportunities.append(opportunity)

            # Sort by function count
            opportunities.sort(key=lambda x: -x["function_count"])

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "mutable_data": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "mutable_data": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    # =========================================================================
    # ALTERNATIVE CLASSES WITH DIFFERENT INTERFACES (Fowler Chapter 3)
    # =========================================================================

    def find_alternative_classes_with_different_interfaces(
        self,
        instance_name: str,
        min_method_similarity: float = 0.5,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Detect classes with similar responsibilities but different interfaces.

        Classes that do similar things but have different method names create
        confusion and make it harder to swap implementations.

        Example:
            class PayPalGateway:
                def charge_card(...): pass
            class StripeGateway:
                def process_payment(...): pass  # Same thing, different name!

        Refactoring: Rename Method / Extract Superclass

        Args:
            instance_name: RETER instance name
            min_method_similarity: Method count similarity threshold (0-1)
            limit: Max results to return
            offset: Pagination offset

        Returns:
            dict with alternative_classes list and metadata
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query all classes and their methods
            query = f"""
                SELECT ?class ?className ?method ?methodName
                WHERE {{
                    ?class type {class_concept} .
                    ?class name ?className .
                    ?method type {method_concept} .
                    ?method definedIn ?class .
                    ?method name ?methodName .
                }}
            """

            result = self.reter.reql(query)
            class_methods = self._query_to_list(result)

            # Group methods by class
            classes = defaultdict(lambda: {"methods": [], "method_count": 0})

            for cls, cls_name, method, method_name in class_methods:
                classes[cls]["name"] = cls_name
                classes[cls]["methods"].append(method_name)
                classes[cls]["method_count"] += 1

            # Compare all class pairs
            def method_count_similarity(count1, count2):
                """Calculate method count similarity (0-1)."""
                if count1 == 0 or count2 == 0:
                    return 0.0
                smaller = min(count1, count2)
                larger = max(count1, count2)
                return smaller / larger

            def has_similar_suffix(name1, name2):
                """Check if class names suggest similar purpose."""
                common_suffixes = ["Gateway", "Processor", "Handler", "Manager", "Service",
                                   "Controller", "Repository", "Provider", "Client", "Adapter"]
                for suffix in common_suffixes:
                    if name1.endswith(suffix) and name2.endswith(suffix):
                        return True
                return False

            opportunities = []
            seen_pairs = set()

            class_list = list(classes.items())
            for i, (cls1, data1) in enumerate(class_list):
                for cls2, data2 in class_list[i+1:]:
                    # Check method count similarity
                    similarity = method_count_similarity(data1["method_count"], data2["method_count"])

                    if similarity >= min_method_similarity:
                        name1 = data1["name"]
                        name2 = data2["name"]

                        # Check if names suggest similar purpose
                        has_suffix = has_similar_suffix(name1, name2)

                        # Calculate method name overlap
                        methods1 = set(data1["methods"])
                        methods2 = set(data2["methods"])
                        common_methods = methods1 & methods2
                        overlap = len(common_methods) / max(len(methods1), len(methods2))

                        # Low overlap with similar count suggests different interfaces
                        if overlap < 0.3 and has_suffix:
                            pair_key = tuple(sorted([cls1, cls2]))
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)

                                severity = "MEDIUM" if similarity > 0.7 else "LOW"

                                opportunity = {
                                    "class1": cls1,
                                    "class1_name": name1,
                                    "class1_methods": data1["methods"][:10],  # First 10
                                    "class1_method_count": data1["method_count"],
                                    "class2": cls2,
                                    "class2_name": name2,
                                    "class2_methods": data2["methods"][:10],
                                    "class2_method_count": data2["method_count"],
                                    "method_count_similarity": round(similarity, 2),
                                    "method_name_overlap": round(overlap, 2),
                                    "common_methods": list(common_methods),
                                    "severity": severity,
                                    "refactoring": "Rename Method / Extract Superclass / Change Function Declaration",
                                    "suggestion": (
                                        f"Classes '{name1}' and '{name2}' have similar method counts "
                                        f"({data1['method_count']} vs {data2['method_count']}) but different interfaces "
                                        f"({overlap:.0%} method name overlap). They likely serve similar purposes. "
                                        f"Consider unifying their interfaces or extracting a common superclass."
                                    ),
                                    "fowler_reference": "Refactoring (2nd Edition), Chapter 3: Alternative Classes with Different Interfaces, p.83"
                                }
                                opportunities.append(opportunity)

            # Sort by similarity
            opportunities.sort(key=lambda x: -x["method_count_similarity"])

            # Apply pagination
            total_count = len(opportunities)
            paginated = opportunities[offset:offset + limit]
            has_more = (offset + limit) < total_count

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "alternative_classes": paginated,
                "count": len(paginated),
                "total_count": total_count,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "alternative_classes": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_flag_arguments(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find boolean parameters (flag arguments) that control function behavior.

        Fowler Chapter 11: Remove Flag Argument

        Detects parameters that are likely boolean flags based on:
        1. Type hint is 'bool'
        2. Name matches boolean patterns (is_, has_, should_, enable_, etc.)

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            {
                "success": bool,
                "flag_arguments": [
                    {
                        "parameter": str (param ID),
                        "parameter_name": str,
                        "type_hint": str or None,
                        "function": str (function ID),
                        "function_name": str,
                        "module": str,
                        "file": str,
                        "line": int,
                        "detection_reason": str ("type_hint" | "naming_pattern" | "both"),
                        "severity": str ("high" | "medium")
                    }
                ],
                "count": int,
                "has_more": bool
            }
        """
        start_time = time.time()

        try:
            param_concept = self._concept('Parameter')
            func_concept = self._concept('Function')
            method_concept = self._concept('Method')

            # Query all parameters with optional type hints
            # Use inFile (works for all languages) instead of inModule
            query = f"""
            SELECT ?param ?paramName ?typeHint ?function ?funcName ?file ?line
            WHERE {{
                ?param type {param_concept} .
                ?param name ?paramName .
                ?param ofFunction ?function .
                ?function type ?funcType .
                ?function name ?funcName .
                ?function inFile ?file .
                ?function atLine ?line .
                OPTIONAL {{ ?param typeHint ?typeHint }}
                FILTER(?funcType = "{func_concept}" || ?funcType = "{method_concept}")
            }}
            ORDER BY ?file ?funcName ?paramName
            """

            result = self.reter.reql(query)
            # Use padded list: 7 columns (OPTIONAL typeHint may not be returned)
            rows = self._query_to_list_padded(result, 7)

            # Boolean naming patterns
            import re
            bool_pattern = re.compile(
                r'^(is_|has_|should_|enable_|disable_|use_|allow_|can_|will_|does_)'
                r'|.*(_flag|_enabled|_disabled|_mode)$',
                re.IGNORECASE
            )

            flag_arguments = []
            for param, param_name, type_hint, function, func_name, file, line in rows:
                # Skip 'self' and 'cls' parameters
                if param_name in ('self', 'cls'):
                    continue

                has_bool_type = type_hint == 'bool'
                has_bool_name = bool_pattern.match(param_name)

                if has_bool_type or has_bool_name:
                    detection_reason = "both" if (has_bool_type and has_bool_name) else \
                                     "type_hint" if has_bool_type else "naming_pattern"

                    # High severity if both type and name indicate boolean
                    severity = "high" if detection_reason == "both" else "medium"

                    flag_arguments.append({
                        "parameter": param,
                        "parameter_name": param_name,
                        "type_hint": type_hint,
                        "function": function,
                        "function_name": func_name,
                        "file": file,
                        "line": line,
                        "detection_reason": detection_reason,
                        "severity": severity,
                        "recommendation": f"Replace flag parameter '{param_name}' with separate functions"
                    })

            # Pagination
            total = len(flag_arguments)
            paginated = flag_arguments[offset:offset + limit]
            has_more = offset + limit < total

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "flag_arguments": paginated,
                "count": len(paginated),
                "total_count": total,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "detector": "Remove Flag Argument (Fowler Chapter 11)",
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "flag_arguments": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_setting_methods(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find setter methods that could be removed for immutability.

        Fowler Chapter 11: Remove Setting Method

        Detects methods that:
        1. Name starts with 'set_' or 'set' followed by uppercase
        2. Have exactly 2 parameters (self + 1 value parameter)
        3. Belong to a class

        These indicate mutable state that could be set only in constructor.

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            {
                "success": bool,
                "setting_methods": [
                    {
                        "method": str (method ID),
                        "method_name": str,
                        "class": str (class ID),
                        "class_name": str,
                        "parameter_count": int,
                        "file": str,
                        "line": int,
                        "severity": str ("high" | "medium"),
                        "naming_style": str ("snake_case" | "camelCase")
                    }
                ],
                "count": int,
                "has_more": bool
            }
        """
        start_time = time.time()

        try:
            method_concept = self._concept('Method')
            param_concept = self._concept('Parameter')

            # Query all setter methods with their classes
            query = f"""
            SELECT ?method ?methodName ?class ?className ?file ?line
            WHERE {{
                ?method type {method_concept} .
                ?method name ?methodName .
                ?method definedIn ?class .
                ?class name ?className .
                ?method inFile ?file .
                ?method atLine ?line .
                FILTER(REGEX(?methodName, "^set[_A-Z]"))
            }}
            ORDER BY ?className ?methodName
            """

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Now count parameters for each method
            param_query = f"""
            SELECT ?method (COUNT(?param) AS ?paramCount)
            WHERE {{
                ?method type {method_concept} .
                ?param type {param_concept} .
                ?param ofFunction ?method .
            }}
            GROUP BY ?method
            """
            param_result = self.reter.reql(param_query)

            # Build parameter count map
            param_counts = {}
            if param_result.num_rows > 0:
                for i in range(param_result.num_rows):
                    method_uri = param_result.column(0)[i].as_py()
                    count = param_result.column(1)[i].as_py()
                    param_counts[method_uri] = count

            setting_methods = []
            for method, method_name, class_id, class_name, file, line in rows:
                # Check parameter count (should be exactly 2: self + value)
                param_count = param_counts.get(method, 0)
                if param_count != 2:
                    continue

                # Determine naming style
                if method_name.startswith("set_"):
                    naming_style = "snake_case"
                    severity = "high"  # Python convention uses snake_case
                elif method_name.startswith("set") and len(method_name) > 3 and method_name[3].isupper():
                    naming_style = "camelCase"
                    severity = "medium"  # Less Pythonic but valid
                else:
                    continue  # Skip if doesn't match setter pattern

                setting_methods.append({
                    "method": method,
                    "method_name": method_name,
                    "class": class_id,
                    "class_name": class_name,
                    "parameter_count": param_count,
                    "file": file,
                    "line": line,
                    "severity": severity,
                    "naming_style": naming_style,
                    "recommendation": f"Consider making {class_name} immutable by removing setter"
                })

            # Pagination
            total = len(setting_methods)
            paginated = setting_methods[offset:offset + limit]
            has_more = offset + limit < total

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "setting_methods": paginated,
                "count": len(paginated),
                "total_count": total,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "detector": "Remove Setting Method (Fowler Chapter 11)",
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "setting_methods": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_trivial_commands(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find trivial command objects that should be functions.

        Fowler Chapter 11: Replace Command with Function

        Detects classes with:
        1. Exactly 1 non-dunder method (besides __init__)
        2. Method name is execute/run/__call__/handle/perform/do

        These are over-engineered and should just be functions.

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            {
                "success": bool,
                "trivial_commands": [
                    {
                        "class": str (class ID),
                        "class_name": str,
                        "command_method": str,
                        "method_count": int (non-dunder),
                        "file": str,
                        "line": int,
                        "severity": str ("high" | "medium"),
                        "pattern_confidence": str ("strong" | "weak")
                    }
                ],
                "count": int,
                "has_more": bool
            }
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query all classes and their methods
            query = f"""
            SELECT ?class ?className ?method ?methodName ?file ?line
            WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?class inFile ?file .
                ?class atLine ?line .
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?method name ?methodName .
            }}
            ORDER BY ?className
            """

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Group methods by class
            from collections import defaultdict
            class_data = defaultdict(lambda: {
                "methods": [],
                "file": None,
                "line": None,
                "class_id": None,
                "class_name": None
            })

            for class_id, class_name, method, method_name, file, line in rows:
                class_data[class_id]["methods"].append(method_name)
                class_data[class_id]["file"] = file
                class_data[class_id]["line"] = line
                class_data[class_id]["class_id"] = class_id
                class_data[class_id]["class_name"] = class_name

            # Command method names (common patterns)
            command_methods = {
                'execute', 'run', '__call__', 'handle',
                'perform', 'do', 'invoke', 'call'
            }

            trivial_commands = []
            for class_id, data in class_data.items():
                # Filter out dunder methods (except __call__ which is valid for commands)
                non_dunder = [m for m in data["methods"] if not m.startswith('__') or m == '__call__']

                # Check if exactly 1 non-dunder method and it's a command method
                if len(non_dunder) == 1:
                    method = non_dunder[0]

                    if method in command_methods:
                        # Strong pattern: classic command method name
                        pattern_confidence = "strong"
                        severity = "high"
                    elif method.endswith('_command') or 'execute' in method.lower():
                        # Weak pattern: method name suggests command but not exact match
                        pattern_confidence = "weak"
                        severity = "medium"
                    else:
                        # Not a command pattern
                        continue

                    trivial_commands.append({
                        "class": data["class_id"],
                        "class_name": data["class_name"],
                        "command_method": method,
                        "method_count": len(non_dunder),
                        "file": data["file"],
                        "line": data["line"],
                        "severity": severity,
                        "pattern_confidence": pattern_confidence,
                        "recommendation": f"Replace {data['class_name']} command class with a simple function"
                    })

            # Pagination
            total = len(trivial_commands)
            paginated = trivial_commands[offset:offset + limit]
            has_more = offset + limit < total

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "trivial_commands": paginated,
                "count": len(paginated),
                "total_count": total,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "detector": "Replace Command with Function (Fowler Chapter 11)",
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "trivial_commands": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    # =========================================================================
    # CHAPTER 12: Dealing with Inheritance Refactoring Detectors
    # =========================================================================

    def find_pull_up_method_candidates(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find duplicate methods in sibling classes that should be pulled up.

        Fowler Chapter 12: Pull Up Method

        Detects methods with same name in sibling classes (classes sharing same parent).

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            {
                "success": bool,
                "candidates": [
                    {
                        "method_name": str,
                        "parent_class": str,
                        "sibling_classes": [str],  # Classes with duplicate method
                        "occurrences": int,
                        "file": str,
                        "recommendation": str
                    }
                ],
                "count": int,
                "has_more": bool
            }
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query sibling classes with methods of same name
            query = f"""
            SELECT ?parent ?parentName ?class1 ?class1Name ?class2 ?class2Name
                   ?method1 ?method2 ?methodName ?file ?line
            WHERE {{
                ?class1 type {class_concept} .
                ?class2 type {class_concept} .
                ?class1 inheritsFrom ?parent .
                ?class2 inheritsFrom ?parent .
                ?parent name ?parentName .
                ?class1 name ?class1Name .
                ?class2 name ?class2Name .
                ?method1 type {method_concept} .
                ?method2 type {method_concept} .
                ?method1 definedIn ?class1 .
                ?method2 definedIn ?class2 .
                ?method1 name ?methodName .
                ?method2 name ?methodName .
                ?method1 inFile ?file .
                ?method1 atLine ?line .
                FILTER(?class1 != ?class2)
                FILTER(?methodName != "__init__")
            }}
            ORDER BY ?parentName ?methodName
            """

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Group by parent and method name
            from collections import defaultdict
            grouped = defaultdict(lambda: {
                "parent_name": None,
                "method_name": None,
                "siblings": set(),
                "file": None,
                "line": None
            })

            for parent, parent_name, c1, c1_name, c2, c2_name, m1, m2, method_name, file, line in rows:
                key = (parent, method_name)
                grouped[key]["parent_name"] = parent_name
                grouped[key]["method_name"] = method_name
                grouped[key]["siblings"].add(c1_name)
                grouped[key]["siblings"].add(c2_name)
                grouped[key]["file"] = file
                grouped[key]["line"] = line

            candidates = []
            for (parent, method_name), data in grouped.items():
                siblings = sorted(list(data["siblings"]))
                if len(siblings) >= 2:  # At least 2 siblings have this method
                    candidates.append({
                        "method_name": data["method_name"],
                        "parent_class": data["parent_name"],
                        "sibling_classes": siblings,
                        "occurrences": len(siblings),
                        "file": data["file"],
                        "line": data["line"],
                        "recommendation": f"Pull up '{data['method_name']}' from {len(siblings)} subclasses to {data['parent_name']}"
                    })

            # Sort by occurrences (more duplicates = higher priority)
            candidates.sort(key=lambda x: x["occurrences"], reverse=True)

            # Pagination
            total = len(candidates)
            paginated = candidates[offset:offset + limit]
            has_more = offset + limit < total

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "candidates": paginated,
                "count": len(paginated),
                "total_count": total,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "detector": "Pull Up Method (Fowler Chapter 12)",
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "candidates": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_push_down_method_candidates(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find superclass methods only used by some subclasses.

        Fowler Chapter 12: Push Down Method

        Detects methods in superclass that are only called by subset of subclasses.

        Args:
            instance_name: RETER instance name
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            {
                "success": bool,
                "candidates": [
                    {
                        "method_name": str,
                        "superclass": str,
                        "total_subclasses": int,
                        "using_subclasses": [str],
                        "unused_subclasses": [str],
                        "file": str,
                        "line": int,
                        "recommendation": str
                    }
                ],
                "count": int,
                "has_more": bool
            }
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query superclass methods and their subclasses
            query1 = f"""
            SELECT ?superclass ?superName ?subclass ?subName
            WHERE {{
                ?superclass type {class_concept} .
                ?superclass name ?superName .
                ?subclass type {class_concept} .
                ?subclass inheritsFrom ?superclass .
                ?subclass name ?subName .
            }}
            """

            # Query methods in superclasses and which subclass methods call them
            query2 = f"""
            SELECT ?superMethod ?superMethodName ?superclass ?superName
                   ?callerMethod ?callerClass ?file ?line
            WHERE {{
                ?superclass type {class_concept} .
                ?superclass name ?superName .
                ?superMethod type {method_concept} .
                ?superMethod definedIn ?superclass .
                ?superMethod name ?superMethodName .
                ?superMethod inFile ?file .
                ?superMethod atLine ?line .
                OPTIONAL {{
                    ?callerMethod type {method_concept} .
                    ?callerMethod calls ?superMethod .
                    ?callerMethod definedIn ?callerClass .
                }}
            }}
            """

            result1 = self.reter.reql(query1)
            result2 = self.reter.reql(query2)

            # Build inheritance tree
            from collections import defaultdict
            subclasses_map = defaultdict(set)  # superclass -> set of subclasses

            rows1 = self._query_to_list(result1)
            for superclass, super_name, subclass, sub_name in rows1:
                subclasses_map[superclass].add(sub_name)

            # Build method usage map
            method_usage = defaultdict(lambda: {
                "superclass": None,
                "method_name": None,
                "file": None,
                "line": None,
                "calling_subclasses": set()
            })

            # Use padded list: 8 columns (OPTIONAL callerMethod, callerClass may not be returned)
            rows2 = self._query_to_list_padded(result2, 8)
            for super_method, method_name, superclass, super_name, caller_method, caller_class, file, line in rows2:
                key = super_method
                method_usage[key]["superclass"] = super_name
                method_usage[key]["method_name"] = method_name
                method_usage[key]["file"] = file
                method_usage[key]["line"] = line

                # Track which subclass calls this method
                if caller_class and caller_class in subclasses_map:
                    # The caller is in a subclass
                    for sub_name in subclasses_map[caller_class]:
                        method_usage[key]["calling_subclasses"].add(sub_name)

            candidates = []
            for method_id, data in method_usage.items():
                superclass = data["superclass"]
                if not superclass:
                    continue

                total_subs = len(subclasses_map.get(superclass, []))
                calling_subs = list(data["calling_subclasses"])
                unused_subs = list(set(subclasses_map.get(superclass, [])) - data["calling_subclasses"])

                # Only report if not all subclasses use the method
                if total_subs > 0 and len(calling_subs) < total_subs and len(calling_subs) > 0:
                    candidates.append({
                        "method_name": data["method_name"],
                        "superclass": superclass,
                        "total_subclasses": total_subs,
                        "using_subclasses": calling_subs,
                        "unused_subclasses": unused_subs,
                        "file": data["file"],
                        "line": data["line"],
                        "recommendation": f"Push down '{data['method_name']}' to {len(calling_subs)}/{total_subs} subclasses that actually use it"
                    })

            # Sort by waste (more unused subclasses = higher priority)
            candidates.sort(key=lambda x: len(x["unused_subclasses"]), reverse=True)

            # Pagination
            total = len(candidates)
            paginated = candidates[offset:offset + limit]
            has_more = offset + limit < total

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "candidates": paginated,
                "count": len(paginated),
                "total_count": total,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "detector": "Push Down Method (Fowler Chapter 12)",
                "queries": [query1, query2],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "candidates": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_remove_subclass_candidates(
        self,
        instance_name: str,
        max_methods: int = 2,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find trivial subclasses that add little value.

        Fowler Chapter 12: Remove Subclass

        Detects subclasses with very few methods (0-2 non-dunder methods).

        Args:
            instance_name: RETER instance name
            max_methods: Maximum non-dunder methods for trivial subclass
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            {
                "success": bool,
                "candidates": [
                    {
                        "subclass": str,
                        "superclass": str,
                        "method_count": int,
                        "methods": [str],
                        "file": str,
                        "line": int,
                        "severity": str,
                        "recommendation": str
                    }
                ],
                "count": int,
                "has_more": bool
            }
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query all subclasses and their methods
            query = f"""
            SELECT ?subclass ?subName ?superclass ?superName ?method ?methodName ?file ?line
            WHERE {{
                ?subclass type {class_concept} .
                ?subclass name ?subName .
                ?subclass inheritsFrom ?superclass .
                ?superclass name ?superName .
                ?subclass inFile ?file .
                ?subclass atLine ?line .
                OPTIONAL {{
                    ?method type {method_concept} .
                    ?method definedIn ?subclass .
                    ?method name ?methodName .
                }}
            }}
            """

            result = self.reter.reql(query)
            # Use padded list: 8 columns (OPTIONAL method, methodName may not be returned)
            rows = self._query_to_list_padded(result, 8)

            # Group by subclass
            from collections import defaultdict
            subclass_data = defaultdict(lambda: {
                "subclass_name": None,
                "superclass_name": None,
                "methods": [],
                "file": None,
                "line": None
            })

            for subclass, sub_name, superclass, super_name, method, method_name, file, line in rows:
                subclass_data[subclass]["subclass_name"] = sub_name
                subclass_data[subclass]["superclass_name"] = super_name
                subclass_data[subclass]["file"] = file
                subclass_data[subclass]["line"] = line
                if method:
                    subclass_data[subclass]["methods"].append(method_name)

            candidates = []
            for subclass_id, data in subclass_data.items():
                # Filter out dunder methods
                non_dunder = [m for m in data["methods"] if not m.startswith('__') or m == '__init__']

                if len(non_dunder) <= max_methods:
                    severity = "high" if len(non_dunder) == 0 else "medium"

                    candidates.append({
                        "subclass": data["subclass_name"],
                        "superclass": data["superclass_name"],
                        "method_count": len(non_dunder),
                        "methods": non_dunder,
                        "file": data["file"],
                        "line": data["line"],
                        "severity": severity,
                        "recommendation": f"Remove trivial subclass {data['subclass_name']} and merge into {data['superclass_name']}"
                    })

            # Sort by method count (fewer methods = more trivial)
            candidates.sort(key=lambda x: x["method_count"])

            # Pagination
            total = len(candidates)
            paginated = candidates[offset:offset + limit]
            has_more = offset + limit < total

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "candidates": paginated,
                "count": len(paginated),
                "total_count": total,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "detector": "Remove Subclass (Fowler Chapter 12)",
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "candidates": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_extract_superclass_candidates(
        self,
        instance_name: str,
        min_shared_methods: int = 2,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find classes with similar methods that should share superclass.

        Fowler Chapter 12: Extract Superclass

        Detects unrelated classes with multiple methods of same name.

        Args:
            instance_name: RETER instance name
            min_shared_methods: Minimum shared methods to report
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            {
                "success": bool,
                "candidates": [
                    {
                        "class1": str,
                        "class2": str,
                        "shared_methods": [str],
                        "shared_count": int,
                        "similarity_score": float,
                        "recommendation": str
                    }
                ],
                "count": int,
                "has_more": bool
            }
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query all classes and their methods
            query = f"""
            SELECT ?class ?className ?method ?methodName
            WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?method name ?methodName .
                FILTER(?methodName != "__init__")
            }}
            """

            result = self.reter.reql(query)
            rows = self._query_to_list(result)

            # Build method sets per class
            from collections import defaultdict
            class_methods = defaultdict(set)

            for class_id, class_name, method, method_name in rows:
                class_methods[class_name].add(method_name)

            # Find pairs with shared methods
            candidates = []
            class_names = sorted(class_methods.keys())

            for i, class1 in enumerate(class_names):
                for class2 in class_names[i+1:]:
                    shared = class_methods[class1] & class_methods[class2]

                    if len(shared) >= min_shared_methods:
                        # Calculate similarity
                        union = class_methods[class1] | class_methods[class2]
                        similarity = len(shared) / len(union) if union else 0

                        candidates.append({
                            "class1": class1,
                            "class2": class2,
                            "shared_methods": sorted(list(shared)),
                            "shared_count": len(shared),
                            "similarity_score": round(similarity, 3),
                            "recommendation": f"Extract superclass from {class1} and {class2} with {len(shared)} common methods"
                        })

            # Sort by shared count (more shared = higher priority)
            candidates.sort(key=lambda x: x["shared_count"], reverse=True)

            # Pagination
            total = len(candidates)
            paginated = candidates[offset:offset + limit]
            has_more = offset + limit < total

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "candidates": paginated,
                "count": len(paginated),
                "total_count": total,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "detector": "Extract Superclass (Fowler Chapter 12)",
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "candidates": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_collapse_hierarchy_candidates(
        self,
        instance_name: str,
        max_additional_methods: int = 2,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find nearly identical parent-child pairs that should be merged.

        Fowler Chapter 12: Collapse Hierarchy

        Detects subclasses that add very few methods beyond parent.

        Args:
            instance_name: RETER instance name
            max_additional_methods: Max additional methods in child
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            {
                "success": bool,
                "candidates": [
                    {
                        "subclass": str,
                        "superclass": str,
                        "subclass_methods": int,
                        "additional_methods": int,
                        "similarity": str,
                        "file": str,
                        "line": int,
                        "recommendation": str
                    }
                ],
                "count": int,
                "has_more": bool
            }
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # This is similar to Remove Subclass but focuses on similarity
            query = f"""
            SELECT ?subclass ?subName ?superclass ?superName ?method ?methodName ?file ?line
            WHERE {{
                ?subclass type {class_concept} .
                ?subclass name ?subName .
                ?subclass inheritsFrom ?superclass .
                ?superclass name ?superName .
                ?subclass inFile ?file .
                ?subclass atLine ?line .
                OPTIONAL {{
                    ?method type {method_concept} .
                    ?method definedIn ?subclass .
                    ?method name ?methodName .
                    FILTER(?methodName != "__init__")
                }}
            }}
            """

            result = self.reter.reql(query)
            # Use padded list: 8 columns (OPTIONAL method, methodName may not be returned)
            rows = self._query_to_list_padded(result, 8)

            # Group by subclass
            from collections import defaultdict
            subclass_data = defaultdict(lambda: {
                "subclass_name": None,
                "superclass_name": None,
                "methods": [],
                "file": None,
                "line": None
            })

            for subclass, sub_name, superclass, super_name, method, method_name, file, line in rows:
                subclass_data[subclass]["subclass_name"] = sub_name
                subclass_data[subclass]["superclass_name"] = super_name
                subclass_data[subclass]["file"] = file
                subclass_data[subclass]["line"] = line
                if method:
                    subclass_data[subclass]["methods"].append(method_name)

            candidates = []
            for subclass_id, data in subclass_data.items():
                additional = len(data["methods"])

                if additional <= max_additional_methods:
                    if additional == 0:
                        similarity = "identical"
                        severity = "high"
                    elif additional == 1:
                        similarity = "nearly_identical"
                        severity = "high"
                    else:
                        similarity = "very_similar"
                        severity = "medium"

                    candidates.append({
                        "subclass": data["subclass_name"],
                        "superclass": data["superclass_name"],
                        "subclass_methods": additional,
                        "additional_methods": additional,
                        "similarity": similarity,
                        "file": data["file"],
                        "line": data["line"],
                        "recommendation": f"Collapse {data['subclass_name']} into {data['superclass_name']} (only {additional} additional methods)"
                    })

            # Sort by additional methods (fewer = more similar)
            candidates.sort(key=lambda x: x["additional_methods"])

            # Pagination
            total = len(candidates)
            paginated = candidates[offset:offset + limit]
            has_more = offset + limit < total

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "candidates": paginated,
                "count": len(paginated),
                "total_count": total,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "detector": "Collapse Hierarchy (Fowler Chapter 12)",
                "queries": [query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "candidates": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

    def find_replace_with_delegate_candidates(
        self,
        instance_name: str,
        max_coupling_ratio: float = 0.3,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find inheritance with low coupling (refused bequest pattern).

        Fowler Chapter 12: Replace Subclass/Superclass with Delegate

        Detects subclasses that use few inherited methods (low coupling).

        Args:
            instance_name: RETER instance name
            max_coupling_ratio: Maximum coupling ratio to report
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            {
                "success": bool,
                "candidates": [
                    {
                        "subclass": str,
                        "superclass": str,
                        "parent_methods": int,
                        "calls_to_parent": int,
                        "coupling_ratio": float,
                        "file": str,
                        "line": int,
                        "severity": str,
                        "recommendation": str
                    }
                ],
                "count": int,
                "has_more": bool
            }
        """
        start_time = time.time()

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Query subclass-superclass pairs and their method calls
            query = f"""
            SELECT ?subclass ?subName ?superclass ?superName ?subMethod ?superMethod ?file ?line
            WHERE {{
                ?subclass type {class_concept} .
                ?subclass name ?subName .
                ?subclass inheritsFrom ?superclass .
                ?superclass name ?superName .
                ?subclass inFile ?file .
                ?subclass atLine ?line .
                ?subMethod type {method_concept} .
                ?subMethod definedIn ?subclass .
                OPTIONAL {{
                    ?superMethod type {method_concept} .
                    ?superMethod definedIn ?superclass .
                    ?subMethod calls ?superMethod .
                }}
            }}
            """

            result = self.reter.reql(query)
            # Use padded list: 8 columns (OPTIONAL superMethod may not be returned)
            rows = self._query_to_list_padded(result, 8)

            # Count parent methods separately
            parent_methods_query = f"""
            SELECT ?superclass ?superName (COUNT(?method) AS ?methodCount)
            WHERE {{
                ?superclass type {class_concept} .
                ?superclass name ?superName .
                ?method type {method_concept} .
                ?method definedIn ?superclass .
                ?method name ?methodName .
                FILTER(?methodName != "__init__")
            }}
            GROUP BY ?superclass ?superName
            """

            parent_result = self.reter.reql(parent_methods_query)

            # Build parent method counts
            parent_method_counts = {}
            if parent_result.num_rows > 0:
                for i in range(parent_result.num_rows):
                    superclass = parent_result.column(0)[i].as_py()
                    count = parent_result.column(2)[i].as_py()
                    parent_method_counts[superclass] = count

            # Group by subclass
            from collections import defaultdict
            subclass_data = defaultdict(lambda: {
                "subclass_name": None,
                "superclass": None,
                "superclass_name": None,
                "calls_to_parent": 0,
                "file": None,
                "line": None
            })

            for subclass, sub_name, superclass, super_name, sub_method, super_method, file, line in rows:
                subclass_data[subclass]["subclass_name"] = sub_name
                subclass_data[subclass]["superclass"] = superclass
                subclass_data[subclass]["superclass_name"] = super_name
                subclass_data[subclass]["file"] = file
                subclass_data[subclass]["line"] = line
                if super_method:
                    subclass_data[subclass]["calls_to_parent"] += 1

            candidates = []
            for subclass_id, data in subclass_data.items():
                superclass = data["superclass"]
                parent_methods = parent_method_counts.get(superclass, 0)

                if parent_methods > 0:
                    coupling = data["calls_to_parent"] / parent_methods

                    if coupling <= max_coupling_ratio:
                        if coupling == 0:
                            severity = "high"
                        elif coupling < 0.2:
                            severity = "high"
                        else:
                            severity = "medium"

                        candidates.append({
                            "subclass": data["subclass_name"],
                            "superclass": data["superclass_name"],
                            "parent_methods": parent_methods,
                            "calls_to_parent": data["calls_to_parent"],
                            "coupling_ratio": round(coupling, 3),
                            "file": data["file"],
                            "line": data["line"],
                            "severity": severity,
                            "recommendation": f"Replace inheritance with delegation - {data['subclass_name']} uses only {data['calls_to_parent']}/{parent_methods} parent methods"
                        })

            # Sort by coupling (lower = worse)
            candidates.sort(key=lambda x: x["coupling_ratio"])

            # Pagination
            total = len(candidates)
            paginated = candidates[offset:offset + limit]
            has_more = offset + limit < total

            time_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "candidates": paginated,
                "count": len(paginated),
                "total_count": total,
                "has_more": has_more,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "next_offset": offset + limit if has_more else None
                },
                "detector": "Replace with Delegate (Fowler Chapter 12)",
                "queries": [query, parent_methods_query],
                "time_ms": time_ms
            }

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "candidates": [],
                "count": 0,
                "total_count": 0,
                "time_ms": time_ms
            }

