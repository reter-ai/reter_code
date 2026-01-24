"""
Tests for the Test Coverage Recommender

Tests the TestMatcher and TestCoverageTool functionality.
"""

import pytest
from reter_code.tools.test_coverage.matcher import TestMatcher, TestMatch


class TestTestMatcher:
    """Tests for the TestMatcher class."""

    def test_find_test_for_class_exact_match(self):
        """Test finding test class with exact Test{Name} pattern."""
        test_classes = [
            {"name": "TestUserService", "module": "tests.test_user"},
            {"name": "TestOrderService", "module": "tests.test_order"},
        ]
        matcher = TestMatcher(test_classes)

        match = matcher.find_test_for_class("UserService", "services.user")

        assert match.is_tested
        assert match.test_name == "TestUserService"
        assert match.confidence == 1.0

    def test_find_test_for_class_not_found(self):
        """Test when no test class exists."""
        test_classes = [
            {"name": "TestOrderService", "module": "tests.test_order"},
        ]
        matcher = TestMatcher(test_classes)

        match = matcher.find_test_for_class("UserService", "services.user")

        assert not match.is_tested
        assert match.test_name is None

    def test_find_test_for_class_partial_match(self):
        """Test partial matching (class name contained in test name)."""
        test_classes = [
            {"name": "TestUserServiceIntegration", "module": "tests.test_user"},
        ]
        matcher = TestMatcher(test_classes)

        match = matcher.find_test_for_class("UserService", "services.user")

        assert match.is_tested
        assert match.test_name == "TestUserServiceIntegration"
        assert match.confidence == 0.7  # Lower confidence for partial

    def test_find_test_for_class_no_reverse_partial_match(self):
        """Test that reverse partial matching is NOT performed.

        The old behavior matched TestThinking to ThinkingSession because
        "Thinking" was contained in "ThinkingSession". This caused massive
        false positives in test coverage detection.

        Now we only match if the class name is at the START of the test suffix,
        e.g., TestThinkingSession matches ThinkingSession.
        """
        test_classes = [
            {"name": "TestThinking", "module": "tests.test_thinking_session"},
        ]
        matcher = TestMatcher(test_classes)

        # ThinkingSession should NOT match TestThinking
        match = matcher.find_test_for_class("ThinkingSession", "tools.unified.session")
        assert not match.is_tested
        assert match.test_name is None

        # But Thinking should still match TestThinking (exact match)
        match2 = matcher.find_test_for_class("Thinking", "tools.unified.session")
        assert match2.is_tested
        assert match2.test_name == "TestThinking"

    def test_find_test_for_function_exact_match(self):
        """Test finding test function with exact test_{name} pattern."""
        test_functions = [
            {"name": "test_calculate_total", "module": "tests.test_utils"},
        ]
        matcher = TestMatcher([], test_functions)

        match = matcher.find_test_for_function("calculate_total", "utils")

        assert match.is_tested
        assert match.test_name == "test_calculate_total"
        assert match.confidence == 1.0

    def test_find_test_for_function_prefix_match(self):
        """Test finding test function with prefix pattern."""
        test_functions = [
            {"name": "test_calculate_total_returns_zero", "module": "tests.test_utils"},
        ]
        matcher = TestMatcher([], test_functions)

        match = matcher.find_test_for_function("calculate_total", "utils")

        assert match.is_tested
        assert match.confidence == 0.9

    def test_find_test_for_function_not_found(self):
        """Test when no test function exists."""
        test_functions = [
            {"name": "test_other_function", "module": "tests.test_utils"},
        ]
        matcher = TestMatcher([], test_functions)

        match = matcher.find_test_for_function("calculate_total", "utils")

        assert not match.is_tested

    def test_suggest_test_file(self):
        """Test test file path suggestion."""
        matcher = TestMatcher([])

        assert matcher.suggest_test_file("mypackage.services.user") == "tests/services/test_user.py"
        assert matcher.suggest_test_file("mypackage.user") == "tests/test_user.py"
        assert matcher.suggest_test_file("user") == "tests/test_user.py"

    def test_suggest_test_class(self):
        """Test test class name suggestion."""
        matcher = TestMatcher([])

        assert matcher.suggest_test_class("UserService") == "TestUserService"
        assert matcher.suggest_test_class("Order") == "TestOrder"

    def test_suggest_test_methods(self):
        """Test test method name suggestions."""
        matcher = TestMatcher([])

        methods = ["save", "delete", "_private", "__init__"]
        suggestions = matcher.suggest_test_methods("User", methods)

        assert "test_save" in suggestions
        assert "test_delete" in suggestions
        assert "test___init__" in suggestions
        assert "test__private" not in suggestions  # Private skipped

    def test_get_coverage_stats(self):
        """Test coverage statistics calculation."""
        test_classes = [
            {"name": "TestUserService", "module": "tests.test_user"},
        ]
        matcher = TestMatcher(test_classes)

        classes = [
            {"name": "UserService", "module": "services.user"},
            {"name": "OrderService", "module": "services.order"},
            {"name": "TestUserService", "module": "tests.test_user"},  # Should be skipped
        ]
        functions = []

        stats = matcher.get_coverage_stats(classes, functions)

        assert stats["total_classes"] == 2  # Excludes test class
        assert stats["tested_classes"] == 1
        assert stats["class_coverage"] == 0.5
        assert len(stats["untested_classes"]) == 1
        assert stats["untested_classes"][0]["name"] == "OrderService"


class TestTestCoverageToolImport:
    """Test that TestCoverageTool can be imported."""

    def test_import_tool(self):
        """Test importing the tool."""
        from reter_code.tools.test_coverage import TestCoverageTool, DETECTORS

        assert TestCoverageTool is not None
        assert DETECTORS is not None
        assert len(DETECTORS) > 0

    def test_detector_registry(self):
        """Test detector registry structure."""
        from reter_code.tools.test_coverage import DETECTORS

        # Check required detectors exist
        assert "untested_classes" in DETECTORS
        assert "untested_functions" in DETECTORS
        assert "complex_untested" in DETECTORS

        # Check detector structure
        for name, info in DETECTORS.items():
            assert "description" in info
            assert "category" in info
            assert "severity" in info
            assert "default_params" in info
            assert info["category"] in ("coverage_gaps", "risk_priority", "test_quality")
            assert info["severity"] in ("critical", "high", "medium", "low")


class TestRecommenderIntegration:
    """Test recommender integration with test_coverage."""

    def test_recommender_types_include_test_coverage(self):
        """Test that test_coverage is in recommender types."""
        from reter_code.services.registrars.refactoring import RECOMMENDER_TYPES

        assert "test_coverage" in RECOMMENDER_TYPES
        assert "refactoring" in RECOMMENDER_TYPES

    def test_registrar_imports_test_coverage(self):
        """Test that registrar can import test_coverage tool."""
        from reter_code.services.registrars.refactoring import RecommenderToolsRegistrar

        assert RecommenderToolsRegistrar is not None


class TestTestMatch:
    """Test the TestMatch dataclass."""

    def test_default_values(self):
        """Test TestMatch default values."""
        match = TestMatch(
            source_name="MyClass",
            source_module="mymodule",
            source_type="class"
        )

        assert match.source_name == "MyClass"
        assert match.source_module == "mymodule"
        assert match.source_type == "class"
        assert match.test_name is None
        assert match.is_tested is False
        assert match.confidence == 0.0

    def test_tested_match(self):
        """Test TestMatch with test found."""
        match = TestMatch(
            source_name="MyClass",
            source_module="mymodule",
            source_type="class",
            test_name="TestMyClass",
            test_module="tests.test_mymodule",
            is_tested=True,
            confidence=1.0
        )

        assert match.is_tested
        assert match.test_name == "TestMyClass"
        assert match.confidence == 1.0
