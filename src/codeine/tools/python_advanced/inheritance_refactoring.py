"""
Inheritance Refactoring Tools

Provides tools for detecting inheritance-related refactoring opportunities
based on Martin Fowler's "Refactoring" Chapter 12.

Detects:
- Pull Up Method (duplicate methods in siblings)
- Push Down Method (superclass methods not used by all subclasses)
- Remove Subclass (trivial subclasses)
- Extract Superclass (similar unrelated classes)
- Collapse Hierarchy (nearly identical parent-child)
- Replace with Delegate (low coupling inheritance)
"""

from typing import Dict, Any
from collections import defaultdict
import time
from .base import AdvancedToolsBase


class InheritanceRefactoringTools(AdvancedToolsBase):
    """
    Inheritance refactoring detection tools.

    Detects opportunities for refactoring class hierarchies.
    """

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
            subclasses_map = defaultdict(set)  # superclass -> set of subclasses

            if result1.num_rows > 0:
                for i in range(result1.num_rows):
                    superclass = result1.column(0)[i].as_py()
                    sub_name = result1.column(3)[i].as_py()
                    subclasses_map[superclass].add(sub_name)

            # Build method usage map
            method_usage = defaultdict(lambda: {
                "superclass": None,
                "method_name": None,
                "file": None,
                "line": None,
                "calling_subclasses": set()
            })

            if result2.num_rows > 0:
                for i in range(result2.num_rows):
                    super_method = result2.column(0)[i].as_py()
                    method_name = result2.column(1)[i].as_py()
                    superclass = result2.column(2)[i].as_py()
                    super_name = result2.column(3)[i].as_py()
                    caller_class = result2.column(5)[i].as_py()
                    file = result2.column(6)[i].as_py()
                    line = result2.column(7)[i].as_py()

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
            rows = self._query_to_list(result)

            # Group by subclass
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
            rows = self._query_to_list(result)

            # Group by subclass
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
                    elif additional == 1:
                        similarity = "nearly_identical"
                    else:
                        similarity = "very_similar"

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
            rows = self._query_to_list(result)

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
