"""
Refactoring to Patterns Plugin

Implements refactoring pattern detectors from "Refactoring to Patterns"
by Joshua Kerievsky, adapted for Python.

Provides two high-level tools:
1. refactoring_to_patterns_prepare - Creates recommendations for running all pattern detectors
2. refactoring_to_patterns_detector - Runs a specific pattern detector and stores findings as recommendations

Pattern Detectors:
- detect_chain_constructors: Find complex __init__ methods needing factory methods
- detect_encapsulate_classes_with_factory: Find classes that should be hidden behind factories
- detect_encapsulate_composite_with_builder: Find composites needing Builder pattern
- detect_extract_adapter: Find version-specific code needing Adapter pattern
- detect_extract_composite: Find duplicate child management needing Composite extraction
- detect_extract_parameter: Find hardcoded instantiation needing dependency injection
- detect_form_template_method: Find duplicate algorithm structures needing Template Method
- detect_inline_singleton: Find unnecessary singletons that can be inlined
- detect_unify_interfaces: Find inconsistent interfaces across class hierarchies
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from codeine.tools.base import ToolMetadata, ToolDefinition
from codeine.tools.refactoring.base import RefactoringToolBase


# =============================================================================
# PATTERN DETECTOR REGISTRY
# =============================================================================

PATTERN_DETECTORS = {
    "detect_chain_constructors": {
        "description": "Detect Chain Constructors pattern opportunities (Fowler/Kerievsky). Find classes with complex __init__ methods that could benefit from factory methods, telescoping constructor anti-pattern",
        "category": "creation",
        "severity": "medium",
        "default_params": {"min_parameters": 5, "include_defaults": True}
    },
    "detect_encapsulate_classes_with_factory": {
        "description": "Detect Encapsulate Classes With Factory pattern opportunities. Find classes sharing a base that should be hidden behind a factory",
        "category": "creation",
        "severity": "medium",
        "default_params": {"min_related_classes": 2}
    },
    "detect_encapsulate_composite_with_builder": {
        "description": "Detect Encapsulate Composite With Builder pattern opportunities. Find classes with add/append methods that need a Builder",
        "category": "creation",
        "severity": "medium",
        "default_params": {"min_add_methods": 2}
    },
    "detect_extract_adapter": {
        "description": "Detect Extract Adapter pattern opportunities. Find classes handling multiple API versions with conditionals",
        "category": "structural",
        "severity": "medium",
        "default_params": {"min_version_methods": 2}
    },
    "detect_extract_composite": {
        "description": "Detect Extract Composite pattern opportunities. Find sibling classes with duplicate child management methods",
        "category": "structural",
        "severity": "medium",
        "default_params": {"min_sibling_classes": 2, "similarity_threshold": 0.7}
    },
    "detect_extract_parameter": {
        "description": "Detect Extract Parameter pattern opportunities. Find hardcoded object instantiation that should be parameterized (dependency injection)",
        "category": "behavioral",
        "severity": "high",
        "default_params": {"include_init": True, "include_methods": True}
    },
    "detect_form_template_method": {
        "description": "Detect Form Template Method pattern opportunities. Find sibling classes with methods performing similar steps",
        "category": "behavioral",
        "severity": "medium",
        "default_params": {"min_common_steps": 3, "similarity_threshold": 0.7}
    },
    "detect_inline_singleton": {
        "description": "Detect Inline Singleton pattern opportunities. Find unnecessary singletons that could be replaced with direct instantiation",
        "category": "simplification",
        "severity": "low",
        "default_params": {"include_usage_analysis": True}
    },
    "detect_unify_interfaces": {
        "description": "Detect Unify Interfaces pattern opportunities. Find superclasses lacking methods that multiple subclasses have",
        "category": "behavioral",
        "severity": "medium",
        "default_params": {"min_subclasses": 2, "include_private": False}
    },
}


class RefactoringToPatternsTool(RefactoringToolBase):
    """
    Refactoring to Patterns plugin implementation.

    Inherits common functionality from RefactoringToolBase.
    """

    def get_metadata(self) -> ToolMetadata:
        """Return plugin metadata."""
        return ToolMetadata(
            name="refactoring_to_patterns",
            version="2.0.0",
            description="Detects refactoring opportunities from 'Refactoring to Patterns' by Joshua Kerievsky - provides prepare/detector interface for pattern analysis",
            author="RETER Team",
            requires_reter=True,
            dependencies=["recommendations"],
            categories=["python", "refactoring", "patterns", "analysis"]
        )

    def get_tools(self) -> List[ToolDefinition]:
        """Return list of tools provided by this plugin."""
        return [
            ToolDefinition(
                name="prepare",
                description="Generate recommendations for running all available pattern detectors. Creates a work plan stored in the unified session.",
                handler=self.prepare,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "instance_name": {
                            "type": "string",
                            "description": "RETER instance to analyze",
                            "default": "default"
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter detectors by category (creation, structural, behavioral, simplification). Default: all"
                        },
                        "severities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter detectors by severity (low, medium, high). Default: all"
                        },
                        "session_instance": {
                            "type": "string",
                            "description": "Unified session instance name",
                            "default": "default"
                        }
                    }
                }
            ),
            ToolDefinition(
                name="detector",
                description="Run a specific pattern detector and store findings as recommendations. Use 'prepare' first to see available detectors.",
                handler=self.detector,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "instance_name": {
                            "type": "string",
                            "description": "RETER instance to analyze",
                            "default": "default"
                        },
                        "detector_name": {
                            "type": "string",
                            "description": "Name of the detector to run (e.g., 'detect_chain_constructors', 'detect_extract_adapter')"
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional parameters to override detector defaults",
                            "additionalProperties": True
                        },
                        "session_instance": {
                            "type": "string",
                            "description": "Unified session instance name",
                            "default": "default"
                        },
                        "create_tasks": {
                            "type": "boolean",
                            "description": "Auto-create tasks from high-priority findings",
                            "default": False
                        },
                        "link_to_thought": {
                            "type": "string",
                            "description": "Link recommendations to a thought ID"
                        }
                    },
                    "required": ["detector_name"]
                }
            ),
        ]

    # =========================================================================
    # High-Level Tool Implementations
    # =========================================================================

    def prepare(
        self,
        instance_name: str = "default",
        categories: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        session_instance: str = "default"
    ) -> Dict[str, Any]:
        """
        Generate recommendations for running all pattern detectors.

        Creates one recommendation per detector, stored in the unified session.
        """
        try:
            # Get unified store
            store = self._get_unified_store()
            if not store:
                return {"success": False, "error": "UnifiedStore not available"}

            # Get or create session
            session_id = self._get_or_create_session(store, session_instance)
            if not session_id:
                return {"success": False, "error": "Failed to create session"}

            # Filter detectors - only include ones that actually have implementations
            filtered_detectors = {}
            for name, info in PATTERN_DETECTORS.items():
                if categories and info["category"] not in categories:
                    continue
                if severities and info["severity"] not in severities:
                    continue

                # Check if the method actually exists (methods are _<detector_name>)
                if not hasattr(self, f"_{name}"):
                    continue

                filtered_detectors[name] = info

            # Generate timestamp for this prepare run
            run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Create recommendations for each detector
            created = []
            for detector_name, detector_info in filtered_detectors.items():
                # Create recommendation text
                text = f"Run pattern detector: {detector_name}"
                description = (
                    f"{detector_info['description']}\n\n"
                    f"Category: {detector_info['category']}\n"
                    f"Default params: {detector_info['default_params']}\n\n"
                    f"Execute with: refactoring_detector(detector_name='{detector_name}')"
                )

                # Create task in unified store (Design Docs approach)
                item_id = store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=text,
                    summary=description,
                    category="refactor",
                    priority=self._severity_to_priority(detector_info["severity"]),
                    status="pending",
                    source_tool="refactoring_to_patterns:prepare",
                    metadata={
                        "detector_name": detector_name,
                        "run_timestamp": run_timestamp,
                        "default_params": detector_info["default_params"],
                        "pattern_category": detector_info["category"]
                    }
                )

                created.append({
                    "item_id": item_id,
                    "detector": detector_name,
                    "category": detector_info["category"],
                    "severity": detector_info["severity"]
                })

            # Group by category for summary (counts only)
            by_category = {}
            for item in created:
                cat = item["category"]
                by_category[cat] = by_category.get(cat, 0) + 1

            # Build detector list for response
            detectors = [
                {
                    "name": name,
                    "description": info["description"],
                    "category": info["category"],
                    "severity": info["severity"],
                    "default_params": info["default_params"]
                }
                for name, info in filtered_detectors.items()
            ]

            return {
                "success": True,
                "message": f"Created {len(created)} pattern detector tasks",
                "tasks_created": len(created),
                "detector_count": len(created),
                "detectors": detectors,
                "by_category": by_category,
                "session_instance": session_instance,
                "note": "Call items(item_type='task', category='refactor') to view details"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def detector(
        self,
        detector_name: str,
        instance_name: str = "default",
        params: Optional[Dict[str, Any]] = None,
        session_instance: str = "default",
        create_tasks: bool = False,
        link_to_thought: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a specific pattern detector and store findings as recommendations.
        """
        try:
            # Validate detector exists
            if detector_name not in PATTERN_DETECTORS:
                available = list(PATTERN_DETECTORS.keys())
                return {
                    "success": False,
                    "error": f"Unknown detector: {detector_name}",
                    "available_detectors": available
                }

            detector_info = PATTERN_DETECTORS[detector_name]

            # Merge default params with provided params
            effective_params = dict(detector_info["default_params"])
            if params:
                effective_params.update(params)

            # Get the detector method
            method = getattr(self, f"_{detector_name}", None)
            if not method:
                return {"success": False, "error": f"Method _{detector_name} not found"}

            # Run the detector
            result = method(instance_name, **effective_params)

            if not result.get("success", True):
                return result

            # Get unified store
            store = self._get_unified_store()
            if not store:
                # Return raw results if no store available
                return {
                    "success": True,
                    "detector": detector_name,
                    "raw_result": result,
                    "recommendations_created": 0,
                    "warning": "UnifiedStore not available"
                }

            # Get or create session
            session_id = self._get_or_create_session(store, session_instance)
            if not session_id:
                return {
                    "success": True,
                    "detector": detector_name,
                    "raw_result": result,
                    "recommendations_created": 0,
                    "warning": "Failed to create session"
                }

            # Convert findings to unified items
            items_result = self._findings_to_items(
                detector_name=detector_name,
                detector_info=detector_info,
                result=result,
                store=store,
                session_id=session_id,
                category_prefix="pattern:",
                link_to_thought=link_to_thought,
                create_tasks=create_tasks
            )

            # Mark any pending "Run pattern detector: X" tasks as completed
            try:
                pending_items = store.get_items(
                    session_id=session_id,
                    item_type="task",
                    status="pending"
                )
                for item in pending_items:
                    content = item.get("content", "")
                    if content == f"Run pattern detector: {detector_name}":
                        store.update_item(item["item_id"], status="completed")
                        break
            except (KeyError, TypeError, AttributeError):
                pass  # Non-critical - continue even if we can't update status

            return {
                "success": True,
                "detector": detector_name,
                "params_used": effective_params,
                "findings_count": self._count_findings(result),
                "tasks_created": items_result["items_created"] + items_result["tasks_created"],
                "relations_created": items_result["relations_created"],
                "session_instance": session_instance,
                "note": "Call items(item_type='task', category='refactor') to view details"
            }

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    # =========================================================================
    # Helper Methods (most inherited from RefactoringToolBase)
    # =========================================================================

    def _query_to_list(self, query_result) -> List:
        """Convert REQL query result to list of tuples."""
        if hasattr(query_result, 'rows'):
            return query_result.rows
        return []

    # =========================================================================
    # Pattern Detector Implementations
    # =========================================================================

    def _detect_chain_constructors(
        self,
        instance_name: str,
        min_parameters: int = 5,
        include_defaults: bool = True,
        class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect Chain Constructors pattern opportunities."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            if class_name:
                params_query = f"""
                SELECT ?class ?className ?init ?param ?paramName ?hasDefault WHERE {{
                    ?class type {class_concept} .
                    ?class name ?className .
                    FILTER(?className = "{class_name}") .
                    ?init type {method_concept} .
                    ?init name "__init__" .
                    ?init definedIn ?class .
                    ?init hasParameter ?param .
                    ?param name ?paramName .
                    OPTIONAL {{ ?param hasDefaultValue ?hasDefault }}
                }}
                """
            else:
                params_query = f"""
                SELECT ?class ?className ?init ?param ?paramName ?hasDefault WHERE {{
                    ?class type {class_concept} .
                    ?class name ?className .
                    ?init type {method_concept} .
                    ?init name "__init__" .
                    ?init definedIn ?class .
                    ?init hasParameter ?param .
                    ?param name ?paramName .
                    OPTIONAL {{ ?param hasDefaultValue ?hasDefault }}
                }}
                """

            params_result = reter.reql(params_query)

            if params_result.num_rows == 0:
                return {"success": True, "opportunities": [], "count": 0}

            inits_by_class = {}
            for row in self._query_to_list(params_result):
                class_id, cls_name, init_id, param_id, param_name, has_default = row

                if init_id not in inits_by_class:
                    inits_by_class[init_id] = {
                        "class_id": class_id,
                        "class_name": cls_name,
                        "params": [],
                        "default_count": 0
                    }

                if param_name != "self":
                    is_optional = has_default is not None
                    if is_optional:
                        inits_by_class[init_id]["default_count"] += 1
                    inits_by_class[init_id]["params"].append({
                        "name": param_name,
                        "has_default": is_optional
                    })

            opportunities = []

            for init_id, init_data in inits_by_class.items():
                cls_name = init_data["class_name"]
                params_info = init_data["params"]
                default_count = init_data["default_count"]
                actual_param_count = len(params_info)

                if actual_param_count < min_parameters:
                    continue

                has_telescoping = default_count > 2
                severity = "high" if actual_param_count >= 8 else ("medium" if actual_param_count >= 6 else "low")

                recommendations = []
                if has_telescoping:
                    recommendations.append("Consider using @classmethod factory methods for common initialization patterns")
                if actual_param_count >= 7:
                    recommendations.append("Consider using a Builder pattern or dataclass")

                opportunities.append({
                    "class_name": cls_name,
                    "parameter_count": actual_param_count,
                    "defaults_count": default_count,
                    "has_telescoping": has_telescoping,
                    "severity": severity,
                    "parameters": [p["name"] for p in params_info],
                    "recommendations": recommendations
                })

            return {
                "success": True,
                "opportunities": opportunities,
                "count": len(opportunities)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_encapsulate_classes_with_factory(
        self,
        instance_name: str,
        min_related_classes: int = 2,
        module_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect Encapsulate Classes With Factory pattern opportunities."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        try:
            class_concept = self._concept('Class')
            query = f"""
            SELECT ?class ?className ?module ?moduleName ?base ?baseName WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?class inModule ?module .
                ?module name ?moduleName .
                ?class inheritsFrom ?base .
                ?base name ?baseName
            }}
            """
            result = reter.reql(query)

            if result.num_rows == 0:
                return {"success": True, "opportunities": [], "count": 0}

            # Group by module and base class
            module_hierarchies = {}
            for row in self._query_to_list(result):
                class_id, cls_name, mod_id, mod_name, base_id, base_name = row

                if module_name and mod_name != module_name:
                    continue

                key = (mod_name, base_name)
                if key not in module_hierarchies:
                    module_hierarchies[key] = {"module": mod_name, "base": base_name, "classes": []}
                module_hierarchies[key]["classes"].append(cls_name)

            opportunities = []
            for key, data in module_hierarchies.items():
                if len(data["classes"]) >= min_related_classes:
                    opportunities.append({
                        "module": data["module"],
                        "base_class": data["base"],
                        "related_classes": data["classes"],
                        "class_count": len(data["classes"]),
                        "recommendation": f"Create a factory method in {data['base']} or a dedicated Factory class"
                    })

            return {
                "success": True,
                "opportunities": opportunities,
                "count": len(opportunities)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_encapsulate_composite_with_builder(
        self,
        instance_name: str,
        min_add_methods: int = 2,
        class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect Encapsulate Composite With Builder pattern opportunities."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')
            query = f"""
            SELECT ?class ?className ?method ?methodName WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?method name ?methodName
            }}
            """
            result = reter.reql(query)

            if result.num_rows == 0:
                return {"success": True, "opportunities": [], "count": 0}

            # Group methods by class
            classes_methods = {}
            for row in self._query_to_list(result):
                class_id, cls_name, method_id, method_name = row

                if class_name and cls_name != class_name:
                    continue

                if cls_name not in classes_methods:
                    classes_methods[cls_name] = []
                classes_methods[cls_name].append(method_name)

            opportunities = []
            add_patterns = ["add", "append", "insert", "push", "attach", "register"]

            for cls_name, methods in classes_methods.items():
                add_methods = [m for m in methods if any(m.startswith(p) or m.endswith(p) for p in add_patterns)]

                if len(add_methods) >= min_add_methods:
                    opportunities.append({
                        "class_name": cls_name,
                        "add_methods": add_methods,
                        "add_method_count": len(add_methods),
                        "recommendation": f"Consider creating a {cls_name}Builder class to encapsulate construction"
                    })

            return {
                "success": True,
                "opportunities": opportunities,
                "count": len(opportunities)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_extract_adapter(
        self,
        instance_name: str,
        min_version_methods: int = 2,
        class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect Extract Adapter pattern opportunities."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')
            query = f"""
            SELECT ?class ?className ?method ?methodName WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?method name ?methodName
            }}
            """
            result = reter.reql(query)

            if result.num_rows == 0:
                return {"success": True, "opportunities": [], "count": 0}

            # Group methods by class
            classes_methods = {}
            for row in self._query_to_list(result):
                class_id, cls_name, method_id, method_name = row

                if class_name and cls_name != class_name:
                    continue

                if cls_name not in classes_methods:
                    classes_methods[cls_name] = []
                classes_methods[cls_name].append(method_name)

            opportunities = []
            version_patterns = ["_v1", "_v2", "_v3", "_old", "_new", "_legacy", "_modern"]

            for cls_name, methods in classes_methods.items():
                version_methods = [m for m in methods if any(p in m.lower() for p in version_patterns)]

                if len(version_methods) >= min_version_methods:
                    opportunities.append({
                        "class_name": cls_name,
                        "version_methods": version_methods,
                        "version_count": len(version_methods),
                        "recommendation": "Extract version-specific logic into separate Adapter classes"
                    })

            return {
                "success": True,
                "opportunities": opportunities,
                "count": len(opportunities)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_extract_composite(
        self,
        instance_name: str,
        min_sibling_classes: int = 2,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Detect Extract Composite pattern opportunities."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')
            query = f"""
            SELECT ?class ?className ?base ?baseName ?method ?methodName WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?class inheritsFrom ?base .
                ?base name ?baseName .
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?method name ?methodName
            }}
            """
            result = reter.reql(query)

            if result.num_rows == 0:
                return {"success": True, "opportunities": [], "count": 0}

            # Group siblings by base class
            base_siblings = {}
            for row in self._query_to_list(result):
                class_id, cls_name, base_id, base_name, method_id, method_name = row

                if base_name not in base_siblings:
                    base_siblings[base_name] = {}
                if cls_name not in base_siblings[base_name]:
                    base_siblings[base_name][cls_name] = set()
                base_siblings[base_name][cls_name].add(method_name)

            opportunities = []
            child_patterns = ["add_child", "remove_child", "add", "remove", "append", "children"]

            for base_name, siblings in base_siblings.items():
                if len(siblings) < min_sibling_classes:
                    continue

                # Find common child-management methods
                sibling_list = list(siblings.items())
                common_child_methods = set()

                for cls_name, methods in sibling_list:
                    child_methods = {m for m in methods if any(p in m.lower() for p in child_patterns)}
                    if not common_child_methods:
                        common_child_methods = child_methods
                    else:
                        common_child_methods &= child_methods

                if common_child_methods:
                    opportunities.append({
                        "base_class": base_name,
                        "sibling_classes": list(siblings.keys()),
                        "common_child_methods": list(common_child_methods),
                        "recommendation": f"Extract common composite behavior to {base_name}"
                    })

            return {
                "success": True,
                "opportunities": opportunities,
                "count": len(opportunities)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_extract_parameter(
        self,
        instance_name: str,
        include_init: bool = True,
        include_methods: bool = True,
        class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect Extract Parameter pattern opportunities."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')
            query = f"""
            SELECT ?class ?className ?method ?methodName ?attr ?attrName WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?method name ?methodName .
                ?class hasAttribute ?attr .
                ?attr name ?attrName
            }}
            """
            result = reter.reql(query)

            if result.num_rows == 0:
                return {"success": True, "opportunities": [], "count": 0}

            # Group by class
            class_data = {}
            for row in self._query_to_list(result):
                c_id, cls_name, m_id, method_name, a_id, attr_name = row

                if class_name and cls_name != class_name:
                    continue

                if cls_name not in class_data:
                    class_data[cls_name] = {"methods": set(), "attrs": set()}
                class_data[cls_name]["methods"].add(method_name)
                class_data[cls_name]["attrs"].add(attr_name)

            opportunities = []
            for cls_name, data in class_data.items():
                methods = data["methods"]
                attrs = data["attrs"]

                # Check for __init__ if requested
                if include_init and "__init__" in methods:
                    # Look for attributes that could be parameters
                    injectable_attrs = [a for a in attrs if not a.startswith("_")]
                    if injectable_attrs:
                        opportunities.append({
                            "class_name": cls_name,
                            "method": "__init__",
                            "injectable_attributes": injectable_attrs,
                            "recommendation": "Consider accepting these as __init__ parameters for dependency injection"
                        })

            return {
                "success": True,
                "opportunities": opportunities,
                "count": len(opportunities)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_form_template_method(
        self,
        instance_name: str,
        min_common_steps: int = 3,
        similarity_threshold: float = 0.7,
        base_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect Form Template Method pattern opportunities."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')
            query = f"""
            SELECT ?class ?className ?base ?baseName ?method ?methodName WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?class inheritsFrom ?base .
                ?base name ?baseName .
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?method name ?methodName
            }}
            """
            result = reter.reql(query)

            if result.num_rows == 0:
                return {"success": True, "opportunities": [], "count": 0}

            # Group by base class
            base_siblings = {}
            for row in self._query_to_list(result):
                c_id, cls_name, b_id, b_name, m_id, method_name = row

                if base_class and b_name != base_class:
                    continue

                if b_name not in base_siblings:
                    base_siblings[b_name] = {}
                if cls_name not in base_siblings[b_name]:
                    base_siblings[b_name][cls_name] = set()
                base_siblings[b_name][cls_name].add(method_name)

            opportunities = []
            for b_name, siblings in base_siblings.items():
                if len(siblings) < 2:
                    continue

                # Find common methods across siblings
                all_methods = [m for ms in siblings.values() for m in ms]
                method_counts = {}
                for m in all_methods:
                    method_counts[m] = method_counts.get(m, 0) + 1

                # Methods present in multiple siblings
                common_methods = [m for m, c in method_counts.items() if c >= 2 and not m.startswith("_")]

                if len(common_methods) >= min_common_steps:
                    opportunities.append({
                        "base_class": b_name,
                        "sibling_classes": list(siblings.keys()),
                        "common_methods": common_methods,
                        "recommendation": f"Form a template method in {b_name} with abstract steps"
                    })

            return {
                "success": True,
                "opportunities": opportunities,
                "count": len(opportunities)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_inline_singleton(
        self,
        instance_name: str,
        include_usage_analysis: bool = True,
        class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect Inline Singleton pattern opportunities."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')
            query = f"""
            SELECT ?class ?className ?attr ?attrName ?method ?methodName WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?class hasAttribute ?attr .
                ?attr name ?attrName .
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?method name ?methodName
            }}
            """
            result = reter.reql(query)

            if result.num_rows == 0:
                return {"success": True, "singletons": [], "count": 0}

            # Group by class
            class_data = {}
            for row in self._query_to_list(result):
                c_id, cls_name, a_id, attr_name, m_id, method_name = row

                if class_name and cls_name != class_name:
                    continue

                if cls_name not in class_data:
                    class_data[cls_name] = {"attrs": set(), "methods": set()}
                class_data[cls_name]["attrs"].add(attr_name)
                class_data[cls_name]["methods"].add(method_name)

            singletons = []
            singleton_indicators = ["_instance", "instance", "_singleton"]
            singleton_methods = ["get_instance", "getInstance", "instance"]

            for cls_name, data in class_data.items():
                is_singleton = False
                indicators = []

                # Check for singleton attributes
                for attr in data["attrs"]:
                    if any(ind in attr.lower() for ind in singleton_indicators):
                        is_singleton = True
                        indicators.append(f"attribute: {attr}")

                # Check for singleton methods
                for method in data["methods"]:
                    if any(method.lower() == sm.lower() for sm in singleton_methods):
                        is_singleton = True
                        indicators.append(f"method: {method}")

                if is_singleton:
                    singletons.append({
                        "class_name": cls_name,
                        "indicators": indicators,
                        "recommendation": "Consider if singleton is necessary - could inline into dependent classes"
                    })

            return {
                "success": True,
                "singletons": singletons,
                "count": len(singletons)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_unify_interfaces(
        self,
        instance_name: str,
        min_subclasses: int = 2,
        include_private: bool = False,
        base_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect Unify Interfaces pattern opportunities."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        try:
            class_concept = self._concept('Class')
            method_concept = self._concept('Method')

            # Get base class methods
            base_query = f"""
            SELECT ?base ?baseName ?method ?methodName WHERE {{
                ?base type {class_concept} .
                ?base name ?baseName .
                ?method type {method_concept} .
                ?method definedIn ?base .
                ?method name ?methodName
            }}
            """
            base_result = reter.reql(base_query)

            # Get subclass methods
            sub_query = f"""
            SELECT ?sub ?subName ?base ?baseName ?method ?methodName WHERE {{
                ?sub type {class_concept} .
                ?sub name ?subName .
                ?sub inheritsFrom ?base .
                ?base name ?baseName .
                ?method type {method_concept} .
                ?method definedIn ?sub .
                ?method name ?methodName
            }}
            """
            sub_result = reter.reql(sub_query)

            if sub_result.num_rows == 0:
                return {"success": True, "opportunities": [], "count": 0}

            # Collect base class methods
            base_methods = {}
            for row in self._query_to_list(base_result):
                b_id, b_name, m_id, method_name = row
                if b_name not in base_methods:
                    base_methods[b_name] = set()
                base_methods[b_name].add(method_name)

            # Collect subclass methods by base
            base_subclasses = {}
            for row in self._query_to_list(sub_result):
                s_id, sub_name, b_id, b_name, m_id, method_name = row

                if base_class and b_name != base_class:
                    continue

                if not include_private and method_name.startswith("_"):
                    continue

                if b_name not in base_subclasses:
                    base_subclasses[b_name] = {}
                if sub_name not in base_subclasses[b_name]:
                    base_subclasses[b_name][sub_name] = set()
                base_subclasses[b_name][sub_name].add(method_name)

            opportunities = []
            for b_name, subclasses in base_subclasses.items():
                if len(subclasses) < min_subclasses:
                    continue

                # Find methods in multiple subclasses but not in base
                b_methods = base_methods.get(b_name, set())
                all_sub_methods = {}

                for sub_name, methods in subclasses.items():
                    for m in methods:
                        if m not in b_methods:
                            if m not in all_sub_methods:
                                all_sub_methods[m] = []
                            all_sub_methods[m].append(sub_name)

                # Methods in >= min_subclasses but not in base
                missing_in_base = {m: subs for m, subs in all_sub_methods.items() if len(subs) >= min_subclasses}

                if missing_in_base:
                    opportunities.append({
                        "base_class": b_name,
                        "subclasses": list(subclasses.keys()),
                        "missing_methods": {m: subs for m, subs in missing_in_base.items()},
                        "recommendation": f"Add these methods to {b_name} with default/abstract implementation"
                    })

            return {
                "success": True,
                "opportunities": opportunities,
                "count": len(opportunities)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
