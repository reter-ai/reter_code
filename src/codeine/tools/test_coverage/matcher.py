"""
Test Matcher

Matches source files, classes, and functions to their corresponding tests.
Uses multiple naming conventions to find existing test coverage.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class TestMatch:
    """Represents a match between source and test."""
    source_name: str
    source_module: str
    source_type: str  # "class", "function", "method"
    test_name: Optional[str] = None
    test_module: Optional[str] = None
    test_file: Optional[str] = None
    is_tested: bool = False
    confidence: float = 0.0  # 0.0-1.0
    pattern_used: Optional[str] = None  # Pattern that matched


class TestMatcher:
    """
    Matches source code entities to their test counterparts.

    Supports multiple naming conventions:
    - Test files: test_*.py, *_test.py
    - Test classes: Test*, *Test, *Tests
    - Test functions: test_*, *_test
    """

    # Test file patterns (module name transformations)
    FILE_PATTERNS = [
        ("test_{name}", 1.0),      # test_foo.py
        ("{name}_test", 0.9),      # foo_test.py
        ("tests.test_{name}", 0.8),  # tests/test_foo.py
        ("test.test_{name}", 0.8),   # test/test_foo.py
    ]

    # Test class patterns
    CLASS_PATTERNS = [
        ("Test{name}", 1.0),       # TestFoo
        ("{name}Test", 0.9),       # FooTest
        ("{name}Tests", 0.85),     # FooTests
        ("Test{name}s", 0.8),      # TestFoos
    ]

    # Test function patterns
    FUNCTION_PATTERNS = [
        ("test_{name}", 1.0),      # test_foo
        ("test_{name}_*", 0.9),    # test_foo_returns_none
        ("{name}_test", 0.8),      # foo_test
    ]

    def __init__(self, test_classes: List[Dict], test_functions: List[Dict] = None):
        """
        Initialize matcher with available tests.

        Args:
            test_classes: List of test class dicts with 'name' and 'module' keys
            test_functions: List of test function dicts with 'name' and 'module' keys
        """
        self.test_classes = {c['name']: c for c in test_classes}
        self.test_functions = {f['name']: f for f in (test_functions or [])}

        # Build lookup sets for fast matching
        self._test_class_names = set(self.test_classes.keys())
        self._test_function_names = set(self.test_functions.keys())
        self._test_modules = set(c.get('module', '') for c in test_classes)

    def find_test_for_class(self, class_name: str, module: str = "") -> TestMatch:
        """
        Find test class for a source class.

        Args:
            class_name: Name of the source class (e.g., "UserService")
            module: Module path of the source class

        Returns:
            TestMatch with test info if found
        """
        match = TestMatch(
            source_name=class_name,
            source_module=module,
            source_type="class"
        )

        # Try each pattern
        for pattern, confidence in self.CLASS_PATTERNS:
            test_name = pattern.format(name=class_name)
            if test_name in self._test_class_names:
                test_info = self.test_classes[test_name]
                match.test_name = test_name
                match.test_module = test_info.get('module', '')
                match.test_file = test_info.get('file', '')
                match.is_tested = True
                match.confidence = confidence
                match.pattern_used = pattern
                return match

        # Try partial matching (class name contained in test name)
        # Only match if test name is exactly Test{ClassName} with optional suffix
        # e.g., TestUserService matches UserService, TestUserServiceIntegration matches UserService
        for test_name in self._test_class_names:
            if test_name.startswith('Test'):
                test_suffix = test_name[4:]  # Remove 'Test' prefix
                # Match only if the class name is at the START of the test suffix
                # This prevents TestThinking from matching ThinkingSession
                if test_suffix.startswith(class_name):
                    test_info = self.test_classes[test_name]
                    match.test_name = test_name
                    match.test_module = test_info.get('module', '')
                    match.test_file = test_info.get('file', '')
                    match.is_tested = True
                    match.confidence = 0.7  # Lower confidence for partial match
                    match.pattern_used = "partial_match"
                    return match

        # NOTE: Removed overly aggressive "reverse_partial_match" logic
        # The old logic matched TestThinking to ThinkingSession because "Thinking" was in "ThinkingSession"
        # This caused massive false positives in test coverage detection

        return match

    def find_test_for_function(self, func_name: str, module: str = "") -> TestMatch:
        """
        Find test function for a source function.

        Args:
            func_name: Name of the source function (e.g., "calculate_total")
            module: Module path of the source function

        Returns:
            TestMatch with test info if found
        """
        match = TestMatch(
            source_name=func_name,
            source_module=module,
            source_type="function"
        )

        # Try exact pattern match
        test_name = f"test_{func_name}"
        if test_name in self._test_function_names:
            test_info = self.test_functions[test_name]
            match.test_name = test_name
            match.test_module = test_info.get('module', '')
            match.is_tested = True
            match.confidence = 1.0
            match.pattern_used = "test_{name}"
            return match

        # Try prefix matching (test_funcname_*)
        for test_name in self._test_function_names:
            if test_name.startswith(f"test_{func_name}_") or test_name == f"test_{func_name}":
                test_info = self.test_functions[test_name]
                match.test_name = test_name
                match.test_module = test_info.get('module', '')
                match.is_tested = True
                match.confidence = 0.9
                match.pattern_used = "test_{name}_*"
                return match

        return match

    def find_test_for_method(
        self,
        method_name: str,
        class_name: str,
        module: str = ""
    ) -> TestMatch:
        """
        Find test for a class method.

        Args:
            method_name: Name of the method (e.g., "save")
            class_name: Name of the containing class
            module: Module path

        Returns:
            TestMatch with test info if found
        """
        match = TestMatch(
            source_name=f"{class_name}.{method_name}",
            source_module=module,
            source_type="method"
        )

        # First check if the class has a test class
        class_match = self.find_test_for_class(class_name, module)
        if not class_match.is_tested:
            return match

        # Look for test method in the test class
        # Pattern: test_method_name or test_class_method
        test_patterns = [
            f"test_{method_name}",
            f"test_{class_name.lower()}_{method_name}",
            f"test_{method_name}_",  # prefix for variations
        ]

        for test_name in self._test_function_names:
            for pattern in test_patterns:
                if test_name.startswith(pattern):
                    test_info = self.test_functions[test_name]
                    # Check if it's in the right test class module
                    if class_match.test_module and class_match.test_module in test_info.get('module', ''):
                        match.test_name = test_name
                        match.test_module = test_info.get('module', '')
                        match.is_tested = True
                        match.confidence = 0.8
                        match.pattern_used = pattern
                        return match

        return match

    def suggest_test_file(self, module: str) -> str:
        """
        Suggest a test file path for a source module.

        Args:
            module: Source module path (e.g., "mypackage.services.user")

        Returns:
            Suggested test file path (e.g., "tests/services/test_user.py")
        """
        parts = module.split('.')
        if not parts:
            return "tests/test_module.py"

        # Get the last part (module name)
        module_name = parts[-1]

        # Build test path
        if len(parts) > 1:
            # Remove package prefix, keep structure
            subpath = '/'.join(parts[1:-1])
            if subpath:
                return f"tests/{subpath}/test_{module_name}.py"

        return f"tests/test_{module_name}.py"

    def suggest_test_class(self, class_name: str) -> str:
        """Suggest a test class name for a source class."""
        return f"Test{class_name}"

    def suggest_test_methods(self, class_name: str, methods: List[str]) -> List[str]:
        """
        Suggest test method names for class methods.

        Args:
            class_name: Source class name
            methods: List of method names to generate tests for

        Returns:
            List of suggested test method names
        """
        suggestions = []
        for method in methods:
            if method.startswith('_') and not method.startswith('__'):
                continue  # Skip private methods
            if method.startswith('__') and method.endswith('__'):
                if method not in ('__init__', '__call__', '__enter__', '__exit__'):
                    continue  # Skip most dunder methods
            suggestions.append(f"test_{method}")
        return suggestions

    def get_coverage_stats(
        self,
        classes: List[Dict],
        functions: List[Dict]
    ) -> Dict:
        """
        Calculate test coverage statistics.

        Args:
            classes: List of source classes
            functions: List of source functions

        Returns:
            Coverage statistics dict
        """
        tested_classes = 0
        untested_classes = []

        for cls in classes:
            name = cls.get('name', '')
            if name.startswith('Test'):
                continue  # Skip test classes
            match = self.find_test_for_class(name, cls.get('module', ''))
            if match.is_tested:
                tested_classes += 1
            else:
                untested_classes.append(cls)

        tested_functions = 0
        untested_functions = []

        for func in functions:
            name = func.get('name', '')
            if name.startswith('test_'):
                continue  # Skip test functions
            if name.startswith('_'):
                continue  # Skip private functions
            match = self.find_test_for_function(name, func.get('module', ''))
            if match.is_tested:
                tested_functions += 1
            else:
                untested_functions.append(func)

        total_classes = len([c for c in classes if not c.get('name', '').startswith('Test')])
        total_functions = len([f for f in functions if not f.get('name', '').startswith(('test_', '_'))])

        return {
            "class_coverage": tested_classes / max(total_classes, 1),
            "function_coverage": tested_functions / max(total_functions, 1),
            "tested_classes": tested_classes,
            "total_classes": total_classes,
            "untested_classes": untested_classes,
            "tested_functions": tested_functions,
            "total_functions": total_functions,
            "untested_functions": untested_functions
        }
