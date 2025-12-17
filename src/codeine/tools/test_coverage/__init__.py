"""
Test Coverage Recommender Tool

Analyzes Python codebases to identify gaps in test coverage
and generates actionable recommendations for new tests.
"""

from .tool import TestCoverageTool, DETECTORS
from .matcher import TestMatcher

__all__ = ['TestCoverageTool', 'TestMatcher', 'DETECTORS']
