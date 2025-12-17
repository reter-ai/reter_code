"""
Data Clump Detection Tools

Detects data clumps (parameters/attributes appearing together) that suggest
the need for Introduce Parameter Object or Introduce Value Object refactoring.

Based on Martin Fowler's "Refactoring" patterns.
"""

from typing import Dict, Any, List
from collections import defaultdict
import time
from .base import AdvancedToolsBase


class DataClumpDetectionTools(AdvancedToolsBase):
    """
    Data clump detection tools for refactoring opportunities.

    Detects:
    - Parameter data clumps (same params in multiple functions)
    - Attribute data clumps (same attrs in multiple classes)
    """

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

            # Efficient algorithm: O(N^2 * M) instead of O(N * 2^M)
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

        except (KeyError, IndexError, TypeError, ValueError, RuntimeError):
            # KeyError: Missing qualified_name
            # IndexError: Empty query results
            # TypeError: Invalid data types
            # ValueError: Conversion errors
            # RuntimeError: REQL query errors
            return False
