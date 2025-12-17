"""
Test Coverage Tool

Provides detectors for analyzing test coverage gaps and generating recommendations.
Follows the same pattern as RefactoringTool for consistency.
"""

import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List
from codeine.tools.base import ToolMetadata, ToolDefinition, BaseTool
from .matcher import TestMatcher


# =============================================================================
# DETECTOR REGISTRY
# =============================================================================

DETECTORS = {
    # Coverage Gaps - Find untested code
    "untested_classes": {
        "description": "Find classes with no corresponding test class",
        "category": "coverage_gaps",
        "severity": "high",
        "default_params": {"min_methods": 2, "exclude_private": True},
    },
    "untested_functions": {
        "description": "Find public functions without tests",
        "category": "coverage_gaps",
        "severity": "high",
        "default_params": {"exclude_private": True, "exclude_dunder": True},
    },
    "untested_methods": {
        "description": "Find public methods without test coverage",
        "category": "coverage_gaps",
        "severity": "medium",
        "default_params": {"exclude_private": True, "exclude_dunder": True},
    },
    "partial_class_coverage": {
        "description": "Find classes with some but not all methods tested",
        "category": "coverage_gaps",
        "severity": "medium",
        "default_params": {"coverage_threshold": 0.5},
    },

    # Risk-Based Priority - High-risk untested code
    "complex_untested": {
        "description": "Find complex code (high method count) without tests",
        "category": "risk_priority",
        "severity": "critical",
        "default_params": {"complexity_threshold": 10},
    },
    "high_fanout_untested": {
        "description": "Find functions with high fan-out (many calls) without tests",
        "category": "risk_priority",
        "severity": "high",
        "default_params": {"fanout_threshold": 5},
    },
    "public_api_untested": {
        "description": "Find public API classes/functions without tests",
        "category": "risk_priority",
        "severity": "critical",
        "default_params": {"api_patterns": ["api", "service", "handler"]},
    },

    # Test Quality - Improve existing tests
    "shallow_tests": {
        "description": "Find test classes with too few test methods",
        "category": "test_quality",
        "severity": "medium",
        "default_params": {"min_tests_per_class": 3},
    },
    "large_untested_modules": {
        "description": "Find modules with many classes but few tests",
        "category": "test_quality",
        "severity": "high",
        "default_params": {"min_classes": 3, "max_test_ratio": 0.3},
    },
}


class TestCoverageTool(BaseTool):
    """
    Test Coverage analysis tool.

    Provides:
    - prepare(): List available detectors and create recommendations to run them
    - detector(): Run a specific detector and store findings
    """

    def __init__(self, instance_manager):
        """Initialize with RETER instance manager."""
        self.instance_manager = instance_manager

    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test_coverage",
            description="Analyze test coverage gaps and recommend tests",
            version="1.0.0",
            author="reter",
            categories=["testing", "quality"]
        )

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="prepare",
                description="Generate recommendations for running all available test coverage detectors",
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
                            "description": "Filter by category (coverage_gaps, risk_priority, test_quality)"
                        },
                        "severities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by severity (critical, high, medium, low)"
                        },
                        "session_instance": {
                            "type": "string",
                            "description": "Session instance for recommendations",
                            "default": "default"
                        }
                    }
                }
            ),
            ToolDefinition(
                name="detector",
                description="Run a specific test coverage detector",
                handler=self.detector,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "detector_name": {
                            "type": "string",
                            "description": f"Detector to run: {', '.join(DETECTORS.keys())}"
                        },
                        "instance_name": {
                            "type": "string",
                            "description": "RETER instance to analyze",
                            "default": "default"
                        },
                        "params": {
                            "type": "object",
                            "description": "Override default parameters"
                        },
                        "session_instance": {
                            "type": "string",
                            "description": "Session instance for recommendations",
                            "default": "default"
                        },
                        "create_tasks": {
                            "type": "boolean",
                            "description": "Auto-create tasks for critical findings",
                            "default": False
                        },
                        "link_to_thought": {
                            "type": "string",
                            "description": "Thought ID to link findings to"
                        }
                    },
                    "required": ["detector_name"]
                }
            )
        ]

    def _get_reter(self, instance_name: str):
        """Get RETER instance."""
        return self.instance_manager.get_or_create_instance(instance_name)

    def _get_unified_store(self):
        """Get UnifiedStore for creating items."""
        try:
            from ..unified.store import UnifiedStore
            return UnifiedStore()
        except (ImportError, OSError):
            # ImportError: Module not available
            # OSError: Database file issues
            return None

    def _get_python_tools(self, instance_name: str):
        """Get PythonAnalysisTools for code inspection."""
        from ..python_basic.python_tools import PythonAnalysisTools
        reter = self.instance_manager.get_or_create_instance(instance_name)
        return PythonAnalysisTools(reter)

    def _get_advanced_tools(self, instance_name: str):
        """Get AdvancedPythonTools for complexity analysis."""
        from ..python_advanced.advanced_python_tools import AdvancedPythonTools
        reter = self.instance_manager.get_or_create_instance(instance_name)
        return AdvancedPythonTools(reter)

    def _severity_to_priority(self, severity: str) -> str:
        """Map detector severity to item priority."""
        mapping = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
        }
        return mapping.get(severity, "medium")

    # =========================================================================
    # PREPARE - List detectors
    # =========================================================================

    def prepare(
        self,
        instance_name: str = "default",
        categories: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        session_instance: str = "default"
    ) -> Dict[str, Any]:
        """
        List available test coverage detectors.

        Args:
            instance_name: RETER instance to analyze
            categories: Filter by category (coverage_gaps, risk_priority, test_quality)
            severities: Filter by severity (critical, high, medium, low)
            session_instance: Session instance for recommendations

        Returns:
            Dict with available detectors and recommendations created
        """
        # Filter detectors
        filtered = {}
        for name, info in DETECTORS.items():
            if categories and info["category"] not in categories:
                continue
            if severities and info["severity"] not in severities:
                continue
            filtered[name] = info

        # Build detector list
        detectors = []
        for name, info in filtered.items():
            detectors.append({
                "name": name,
                "description": info["description"],
                "category": info["category"],
                "severity": info["severity"],
                "default_params": info["default_params"]
            })

        # Create recommendations to run each detector
        recommendations_created = 0
        store = self._get_unified_store()
        if store:
            try:
                session_id = store.get_or_create_session(session_instance)
                for det in detectors:
                    store.add_item(
                        session_id=session_id,
                        item_type="recommendation",
                        content=f"Run test coverage detector: {det['name']} - {det['description']}",
                        category=f"test_coverage:{det['category']}",
                        priority=self._severity_to_priority(det['severity']),
                        status="pending",
                        source_tool=f"test_coverage:prepare",
                        metadata={"detector": det['name'], "action": "run_detector"}
                    )
                    recommendations_created += 1
            except Exception as e:
                pass  # Store not available

        return {
            "success": True,
            "detectors": detectors,
            "detector_count": len(detectors),
            "recommendations_created": recommendations_created,
            "session_instance": session_instance
        }

    # =========================================================================
    # DETECTOR - Run specific detector
    # =========================================================================

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
        Run a specific test coverage detector.

        Args:
            detector_name: Name of detector to run
            instance_name: RETER instance to analyze
            params: Override default parameters
            session_instance: Session for storing recommendations
            create_tasks: Auto-create tasks for critical findings
            link_to_thought: Link findings to a thought ID

        Returns:
            Detection results and recommendations created
        """
        if detector_name not in DETECTORS:
            return {
                "success": False,
                "error": f"Unknown detector: {detector_name}",
                "available_detectors": list(DETECTORS.keys())
            }

        detector_info = DETECTORS[detector_name]
        effective_params = dict(detector_info.get("default_params", {}))
        if params:
            effective_params.update(params)

        # Get tools
        python_tools = self._get_python_tools(instance_name)
        advanced_tools = self._get_advanced_tools(instance_name)

        # Run the detector
        try:
            if detector_name == "untested_classes":
                result = self._detect_untested_classes(
                    instance_name, python_tools, **effective_params
                )
            elif detector_name == "untested_functions":
                result = self._detect_untested_functions(
                    instance_name, python_tools, **effective_params
                )
            elif detector_name == "untested_methods":
                result = self._detect_untested_methods(
                    instance_name, python_tools, **effective_params
                )
            elif detector_name == "partial_class_coverage":
                result = self._detect_partial_coverage(
                    instance_name, python_tools, **effective_params
                )
            elif detector_name == "complex_untested":
                result = self._detect_complex_untested(
                    instance_name, python_tools, advanced_tools, **effective_params
                )
            elif detector_name == "high_fanout_untested":
                result = self._detect_high_fanout_untested(
                    instance_name, python_tools, advanced_tools, **effective_params
                )
            elif detector_name == "public_api_untested":
                result = self._detect_public_api_untested(
                    instance_name, python_tools, **effective_params
                )
            elif detector_name == "shallow_tests":
                result = self._detect_shallow_tests(
                    instance_name, python_tools, **effective_params
                )
            elif detector_name == "large_untested_modules":
                result = self._detect_large_untested_modules(
                    instance_name, python_tools, **effective_params
                )
            else:
                return {"success": False, "error": f"Detector {detector_name} not implemented"}

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

        # Store findings as recommendations
        items_created = self._findings_to_items(
            detector_name=detector_name,
            detector_info=detector_info,
            result=result,
            session_instance=session_instance,
            create_tasks=create_tasks,
            link_to_thought=link_to_thought
        )

        return {
            "success": True,
            "detector": detector_name,
            "params_used": effective_params,
            "findings": result.get("findings", []),
            "findings_count": len(result.get("findings", [])),
            "recommendations_created": items_created.get("items_created", 0),
            "tasks_created": items_created.get("tasks_created", 0),
            "session_instance": session_instance
        }

    # =========================================================================
    # DETECTOR IMPLEMENTATIONS
    # =========================================================================

    def _get_test_classes(self, instance_name: str, python_tools) -> List[Dict]:
        """Get all test classes from the codebase."""
        result = python_tools.list_classes(instance_name=instance_name, limit=500)
        classes = result.get("classes", [])
        return [c for c in classes if c.get("name", "").startswith("Test")]

    def _get_test_functions(self, instance_name: str, python_tools) -> List[Dict]:
        """Get all test functions from the codebase."""
        result = python_tools.list_functions(instance_name=instance_name, limit=1000)
        functions = result.get("functions", [])
        return [f for f in functions if f.get("name", "").startswith("test_")]

    def _detect_untested_classes(
        self,
        instance_name: str,
        python_tools,
        min_methods: int = 2,
        exclude_private: bool = True
    ) -> Dict[str, Any]:
        """Find classes without corresponding test classes."""
        # Get all classes
        result = python_tools.list_classes(instance_name=instance_name, limit=500)
        all_classes = result.get("classes", [])

        # Get test classes
        test_classes = self._get_test_classes(instance_name, python_tools)

        # Create matcher
        matcher = TestMatcher(test_classes)

        findings = []
        for cls in all_classes:
            name = cls.get("name", "")

            # Skip test classes
            if name.startswith("Test"):
                continue

            # Skip private classes if configured
            if exclude_private and name.startswith("_"):
                continue

            # Get method count
            method_count = cls.get("method_count", 0)
            if method_count < min_methods:
                continue

            # Check for test coverage
            match = matcher.find_test_for_class(name, cls.get("module", ""))
            if not match.is_tested:
                findings.append({
                    "class_name": name,
                    "module": cls.get("module", ""),
                    "file": cls.get("file", ""),
                    "method_count": method_count,
                    "suggested_test_file": matcher.suggest_test_file(cls.get("module", "")),
                    "suggested_test_class": matcher.suggest_test_class(name),
                    "priority": "high" if method_count > 10 else "medium"
                })

        return {
            "success": True,
            "findings": findings,
            "total_classes": len(all_classes),
            "test_classes": len(test_classes)
        }

    def _detect_untested_functions(
        self,
        instance_name: str,
        python_tools,
        exclude_private: bool = True,
        exclude_dunder: bool = True
    ) -> Dict[str, Any]:
        """Find public functions without tests."""
        result = python_tools.list_functions(instance_name=instance_name, limit=1000)
        all_functions = result.get("functions", [])

        test_functions = self._get_test_functions(instance_name, python_tools)
        test_classes = self._get_test_classes(instance_name, python_tools)

        matcher = TestMatcher(test_classes, test_functions)

        findings = []
        for func in all_functions:
            name = func.get("name", "")

            # Skip test functions
            if name.startswith("test_"):
                continue

            # Skip private
            if exclude_private and name.startswith("_") and not name.startswith("__"):
                continue

            # Skip dunder methods
            if exclude_dunder and name.startswith("__") and name.endswith("__"):
                continue

            # Check for test coverage
            match = matcher.find_test_for_function(name, func.get("module", ""))
            if not match.is_tested:
                findings.append({
                    "function_name": name,
                    "module": func.get("module", ""),
                    "file": func.get("file", ""),
                    "suggested_test": f"test_{name}",
                    "priority": "medium"
                })

        return {
            "success": True,
            "findings": findings,
            "total_functions": len(all_functions),
            "test_functions": len(test_functions)
        }

    def _detect_untested_methods(
        self,
        instance_name: str,
        python_tools,
        exclude_private: bool = True,
        exclude_dunder: bool = True
    ) -> Dict[str, Any]:
        """Find public methods without test coverage."""
        # Get classes with their methods
        result = python_tools.list_classes(instance_name=instance_name, limit=500)
        all_classes = result.get("classes", [])

        test_classes = self._get_test_classes(instance_name, python_tools)
        test_functions = self._get_test_functions(instance_name, python_tools)
        matcher = TestMatcher(test_classes, test_functions)

        findings = []
        for cls in all_classes:
            class_name = cls.get("name", "")
            if class_name.startswith("Test"):
                continue

            # Check if class has tests
            class_match = matcher.find_test_for_class(class_name)
            if not class_match.is_tested:
                continue  # Already covered by untested_classes

            # Get methods for this class
            try:
                class_info = python_tools.describe_class(
                    class_name=class_name,
                    instance_name=instance_name
                )
                methods = class_info.get("methods", [])
            except (KeyError, AttributeError, TypeError):
                # Class info retrieval failed - skip this class
                continue

            for method in methods:
                method_name = method.get("name", "")

                if exclude_private and method_name.startswith("_") and not method_name.startswith("__"):
                    continue
                if exclude_dunder and method_name.startswith("__"):
                    continue

                # Check for method test
                method_match = matcher.find_test_for_method(method_name, class_name)
                if not method_match.is_tested:
                    findings.append({
                        "method_name": method_name,
                        "class_name": class_name,
                        "module": cls.get("module", ""),
                        "suggested_test": f"test_{method_name}",
                        "test_class": class_match.test_name,
                        "priority": "low"
                    })

        return {"success": True, "findings": findings}

    def _detect_partial_coverage(
        self,
        instance_name: str,
        python_tools,
        coverage_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Find classes with partial test coverage."""
        result = python_tools.list_classes(instance_name=instance_name, limit=500)
        all_classes = result.get("classes", [])

        test_classes = self._get_test_classes(instance_name, python_tools)
        test_functions = self._get_test_functions(instance_name, python_tools)
        matcher = TestMatcher(test_classes, test_functions)

        findings = []
        for cls in all_classes:
            class_name = cls.get("name", "")
            if class_name.startswith("Test"):
                continue

            class_match = matcher.find_test_for_class(class_name)
            if not class_match.is_tested:
                continue  # No test class at all

            # Count tested vs untested methods
            try:
                class_info = python_tools.describe_class(
                    class_name=class_name,
                    instance_name=instance_name
                )
                methods = class_info.get("methods", [])
            except (KeyError, AttributeError, TypeError):
                # Class info retrieval failed - skip this class
                continue

            public_methods = [m for m in methods if not m.get("name", "").startswith("_")]
            if not public_methods:
                continue

            tested = 0
            untested_methods = []
            for method in public_methods:
                method_name = method.get("name", "")
                match = matcher.find_test_for_method(method_name, class_name)
                if match.is_tested:
                    tested += 1
                else:
                    untested_methods.append(method_name)

            coverage = tested / len(public_methods)
            if 0 < coverage < coverage_threshold:
                findings.append({
                    "class_name": class_name,
                    "module": cls.get("module", ""),
                    "coverage": round(coverage, 2),
                    "tested_methods": tested,
                    "total_methods": len(public_methods),
                    "untested_methods": untested_methods[:5],  # Top 5
                    "test_class": class_match.test_name,
                    "priority": "medium"
                })

        return {"success": True, "findings": findings}

    def _detect_complex_untested(
        self,
        instance_name: str,
        python_tools,
        advanced_tools,
        complexity_threshold: int = 10
    ) -> Dict[str, Any]:
        """Find complex classes without tests."""
        # Get complexity metrics
        complexity = advanced_tools.get_complexity_metrics(instance_name=instance_name)
        class_complexity = complexity.get("class_complexity", {})
        top_classes = class_complexity.get("top_10_largest", [])

        test_classes = self._get_test_classes(instance_name, python_tools)
        matcher = TestMatcher(test_classes)

        findings = []
        for cls in top_classes:
            class_name = cls.get("class_name", "")
            method_count = cls.get("method_count", 0)

            if class_name.startswith("Test"):
                continue
            if method_count < complexity_threshold:
                continue

            match = matcher.find_test_for_class(class_name)
            if not match.is_tested:
                findings.append({
                    "class_name": class_name,
                    "method_count": method_count,
                    "qualified_name": cls.get("class", ""),
                    "suggested_test_class": matcher.suggest_test_class(class_name),
                    "priority": "critical",
                    "reason": f"High complexity ({method_count} methods) without tests"
                })

        return {"success": True, "findings": findings}

    def _detect_high_fanout_untested(
        self,
        instance_name: str,
        python_tools,
        advanced_tools,
        fanout_threshold: int = 5
    ) -> Dict[str, Any]:
        """Find high fan-out functions without tests."""
        complexity = advanced_tools.get_complexity_metrics(instance_name=instance_name)
        call_complexity = complexity.get("call_complexity", {})
        top_callers = call_complexity.get("top_10_highest_fanout", [])

        test_functions = self._get_test_functions(instance_name, python_tools)
        test_classes = self._get_test_classes(instance_name, python_tools)
        matcher = TestMatcher(test_classes, test_functions)

        findings = []
        for func in top_callers:
            func_name = func.get("function_name", "")
            calls_count = func.get("calls_count", 0)

            if func_name.startswith("test_"):
                continue
            if calls_count < fanout_threshold:
                continue

            match = matcher.find_test_for_function(func_name)
            if not match.is_tested:
                findings.append({
                    "function_name": func_name,
                    "calls_count": calls_count,
                    "qualified_name": func.get("function", ""),
                    "suggested_test": f"test_{func_name}",
                    "priority": "high",
                    "reason": f"High fan-out ({calls_count} calls) without tests"
                })

        return {"success": True, "findings": findings}

    def _detect_public_api_untested(
        self,
        instance_name: str,
        python_tools,
        api_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """Find public API classes without tests."""
        if api_patterns is None:
            api_patterns = ["api", "service", "handler", "controller", "endpoint"]

        result = python_tools.list_classes(instance_name=instance_name, limit=500)
        all_classes = result.get("classes", [])

        test_classes = self._get_test_classes(instance_name, python_tools)
        matcher = TestMatcher(test_classes)

        findings = []
        for cls in all_classes:
            class_name = cls.get("name", "")
            module = cls.get("module", "").lower()

            if class_name.startswith("Test"):
                continue

            # Check if module matches API patterns
            is_api = any(pattern in module for pattern in api_patterns)
            is_api = is_api or any(pattern in class_name.lower() for pattern in api_patterns)

            if not is_api:
                continue

            match = matcher.find_test_for_class(class_name)
            if not match.is_tested:
                findings.append({
                    "class_name": class_name,
                    "module": cls.get("module", ""),
                    "method_count": cls.get("method_count", 0),
                    "suggested_test_class": matcher.suggest_test_class(class_name),
                    "priority": "critical",
                    "reason": "Public API without tests"
                })

        return {"success": True, "findings": findings}

    def _detect_shallow_tests(
        self,
        instance_name: str,
        python_tools,
        min_tests_per_class: int = 3
    ) -> Dict[str, Any]:
        """Find test classes with too few test methods."""
        test_classes = self._get_test_classes(instance_name, python_tools)
        test_functions = self._get_test_functions(instance_name, python_tools)

        # Count test methods per test class
        test_method_counts = {}
        for func in test_functions:
            module = func.get("module", "")
            for tc in test_classes:
                if tc.get("module", "") == module:
                    tc_name = tc.get("name", "")
                    test_method_counts[tc_name] = test_method_counts.get(tc_name, 0) + 1
                    break

        findings = []
        for tc in test_classes:
            tc_name = tc.get("name", "")
            count = test_method_counts.get(tc_name, 0)

            if count < min_tests_per_class:
                # Infer source class name
                source_class = tc_name.replace("Test", "").replace("Tests", "")
                findings.append({
                    "test_class": tc_name,
                    "module": tc.get("module", ""),
                    "test_count": count,
                    "min_expected": min_tests_per_class,
                    "source_class": source_class,
                    "priority": "medium",
                    "reason": f"Only {count} tests for {source_class}"
                })

        return {"success": True, "findings": findings}

    def _detect_large_untested_modules(
        self,
        instance_name: str,
        python_tools,
        min_classes: int = 3,
        max_test_ratio: float = 0.3
    ) -> Dict[str, Any]:
        """Find modules with many classes but few tests."""
        result = python_tools.list_classes(instance_name=instance_name, limit=500)
        all_classes = result.get("classes", [])

        # Group by module
        module_classes = {}
        module_tests = {}

        for cls in all_classes:
            module = cls.get("module", "")
            name = cls.get("name", "")

            if name.startswith("Test"):
                module_tests[module] = module_tests.get(module, 0) + 1
            else:
                module_classes[module] = module_classes.get(module, 0) + 1

        findings = []
        for module, class_count in module_classes.items():
            if class_count < min_classes:
                continue

            # Find corresponding test module
            test_count = 0
            for test_module, count in module_tests.items():
                if module.split(".")[-1] in test_module:
                    test_count = count
                    break

            ratio = test_count / class_count if class_count > 0 else 0
            if ratio < max_test_ratio:
                findings.append({
                    "module": module,
                    "class_count": class_count,
                    "test_count": test_count,
                    "test_ratio": round(ratio, 2),
                    "priority": "high",
                    "reason": f"Low test ratio ({ratio:.0%}) for {class_count} classes"
                })

        return {"success": True, "findings": findings}

    # =========================================================================
    # FINDINGS TO ITEMS
    # =========================================================================

    def _findings_to_items(
        self,
        detector_name: str,
        detector_info: Dict[str, Any],
        result: Dict[str, Any],
        session_instance: str,
        create_tasks: bool = False,
        link_to_thought: Optional[str] = None
    ) -> Dict[str, int]:
        """Convert findings to unified store items."""
        findings = result.get("findings", [])
        if not findings:
            return {"items_created": 0, "tasks_created": 0}

        store = self._get_unified_store()
        if not store:
            return {"items_created": 0, "tasks_created": 0}

        try:
            session_id = store.get_or_create_session(session_instance)
        except (sqlite3.Error, OSError):
            # Database or file system error
            return {"items_created": 0, "tasks_created": 0}

        items_created = 0
        tasks_created = 0
        priority = self._severity_to_priority(detector_info.get("severity", "medium"))

        for finding in findings:
            # Build recommendation text
            text = self._finding_to_text(detector_name, finding)

            # Create recommendation
            rec_id = store.add_item(
                session_id=session_id,
                item_type="recommendation",
                content=text,
                category=f"test_coverage:{detector_info['category']}",
                priority=finding.get("priority", priority),
                status="pending",
                source_tool=f"test_coverage:{detector_name}",
                metadata=finding
            )
            items_created += 1

            # Link to thought if specified
            if link_to_thought:
                store.add_relation(rec_id, link_to_thought, "item", "traces")

            # Create task for critical/high priority
            if create_tasks and finding.get("priority", priority) in ("critical", "high"):
                task_text = f"Write tests: {text}"
                task_id = store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=task_text,
                    category=f"test_coverage:{detector_info['category']}",
                    priority=finding.get("priority", priority),
                    status="pending",
                    source_tool=f"test_coverage:{detector_name}"
                )
                store.add_relation(task_id, rec_id, "item", "traces")
                tasks_created += 1

        return {"items_created": items_created, "tasks_created": tasks_created}

    def _finding_to_text(self, detector_name: str, finding: Dict[str, Any]) -> str:
        """Convert finding to readable text."""
        if "class_name" in finding:
            name = finding["class_name"]
            module = finding.get("module", "")
            if finding.get("method_count"):
                return f"Add tests for {name} ({finding['method_count']} methods) in {module}"
            return f"Add tests for class {name} in {module}"

        if "function_name" in finding:
            name = finding["function_name"]
            module = finding.get("module", "")
            return f"Add test for function {name} in {module}"

        if "method_name" in finding:
            method = finding["method_name"]
            cls = finding.get("class_name", "")
            return f"Add test for {cls}.{method}"

        if "test_class" in finding:
            return f"Expand tests in {finding['test_class']}: {finding.get('reason', '')}"

        if "module" in finding:
            return f"Add tests for module {finding['module']}: {finding.get('reason', '')}"

        return f"{detector_name}: {str(finding)[:100]}"
