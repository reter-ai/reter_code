"""
Refactoring & Improving Plugin

Provides two high-level tools:
1. refactoring_improving_prepare - Creates recommendations for running all detectors
2. refactoring_improving_detector - Runs a specific detector and stores findings as recommendations

Uses the existing AdvancedPythonTools and RefactoringOpportunityDetector for actual analysis,
and the Recommendations plugin for storing results.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from codeine.tools.base import ToolMetadata, ToolDefinition
from codeine.tools.refactoring.base import RefactoringToolBase


# =============================================================================
# EXTRACTION ANALYSIS DATACLASSES
# =============================================================================
# These were moved from code_diff.py as part of RETER-based refactoring

@dataclass
class ParameterSuggestion:
    """A suggested parameter for the extracted function."""
    name: str
    inferred_type: str
    values: List[str]  # The different values found in the code blocks
    diff_type: str  # "name", "literal", "string"


@dataclass
class ExtractionAnalysis:
    """Result of analyzing two code blocks for extraction."""
    extractable: bool
    reason: str
    similarity_score: float
    parameters: List[ParameterSuggestion]
    common_pattern: str  # Template with placeholders
    suggested_name: str
    diff_count: int
    total_tokens: int


# =============================================================================
# DETECTOR REGISTRY
# =============================================================================
# Each detector has: name, description, category, severity, default_params, source_class

DETECTORS = {
    # Code Smells - Quality Issues
    # All methods are in AdvancedPythonTools - parameters match actual method signatures
    "find_large_classes": {
        "description": "Find classes with too many methods (God classes)",
        "category": "code_smell",
        "severity": "high",
        "default_params": {
            "threshold": 20,
            "exclude_test_files": True,
            "exclude_patterns": ["*Visitor*", "*FactExtraction*", "*ParserBase*", "*LexerBase*"]
        },
        "source": "advanced"
    },
    "find_long_parameter_lists": {
        "description": "Find functions/methods with too many parameters",
        "category": "code_smell",
        "severity": "medium",
        "default_params": {
            "threshold": 5,
            "exclude_test_files": True,
            "exclude_patterns": []
        },
        "source": "advanced"
    },
    "find_magic_numbers": {
        "description": "Find magic numbers (numeric literals) that should be named constants",
        "category": "code_smell",
        "severity": "medium",
        "default_params": {"exclude_common": True, "min_occurrences": 2, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    # NOTE: detect_long_functions, detect_data_classes, detect_feature_envy, detect_refused_bequest
    # were removed - methods don't exist in AdvancedPythonTools. Need to implement or keep removed.
    "find_message_chains": {
        "description": "Find long method call chains",
        "category": "code_smell",
        "severity": "medium",
        "default_params": {"min_chain_length": 3, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_global_data": {
        "description": "Find module-level mutable assignments",
        "category": "code_smell",
        "severity": "high",
        "default_params": {"limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_speculative_generality": {
        "description": "Find abstract classes with only 1 subclass",
        "category": "code_smell",
        "severity": "low",
        "default_params": {"limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_parallel_inheritance_hierarchies": {
        "description": "Find class hierarchies that mirror each other",
        "category": "code_smell",
        "severity": "medium",
        "default_params": {"min_similarity": 0.6, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_mutable_data_across_functions": {
        "description": "Find variables assigned in multiple different functions",
        "category": "code_smell",
        "severity": "high",
        "default_params": {"min_functions": 3, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_alternative_classes_with_different_interfaces": {
        "description": "Find classes with similar responsibilities but different interfaces",
        "category": "code_smell",
        "severity": "medium",
        "default_params": {"min_method_similarity": 0.5, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_flag_arguments": {
        "description": "Find boolean parameters that control function behavior",
        "category": "code_smell",
        "severity": "medium",
        "default_params": {"limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_setting_methods": {
        "description": "Find setter methods indicating immutability opportunities",
        "category": "code_smell",
        "severity": "low",
        "default_params": {"limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_trivial_commands": {
        "description": "Find trivial command objects that should be functions",
        "category": "code_smell",
        "severity": "low",
        "default_params": {"limit": 100, "offset": 0},
        "source": "advanced"
    },

    # Refactoring Opportunities - all in AdvancedPythonTools
    "find_data_clumps": {
        "description": "Detect parameter groups that appear together in multiple functions",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_params": 3, "min_functions": 2},
        "source": "advanced"
    },
    "find_attribute_data_clumps": {
        "description": "Detect groups of attributes that appear together in multiple classes",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_attrs": 3, "min_classes": 2, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_function_groups": {
        "description": "Identify groups of functions operating on shared data",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_shared_params": 2, "min_functions": 3},
        "source": "advanced"
    },
    "find_extract_function_opportunities": {
        "description": "Find functions that are candidates for Extract Function",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_lines": 20, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_inline_function_candidates": {
        "description": "Find trivial functions called from only one location",
        "category": "refactoring",
        "severity": "low",
        "default_params": {"max_lines": 5, "limit": 100},
        "source": "advanced"
    },
    "find_duplicate_parameter_lists": {
        "description": "Find functions with identical parameter signatures",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_params": 2, "min_functions": 2, "limit": 100},
        "source": "advanced"
    },
    "find_shotgun_surgery": {
        "description": "Detect functions/classes with high fan-in from many modules",
        "category": "refactoring",
        "severity": "high",
        "default_params": {"min_callers": 5, "min_modules": 3},
        "source": "advanced"
    },
    "find_middle_man": {
        "description": "Detect classes/methods that just delegate to other classes",
        "category": "refactoring",
        "severity": "low",
        "default_params": {"max_lines": 10, "min_delegation_ratio": 0.5},
        "source": "advanced"
    },
    "find_extract_class_opportunities": {
        "description": "Detect classes that should be split into multiple classes",
        "category": "refactoring",
        "severity": "high",
        "default_params": {"min_methods": 10, "min_cohesion_gap": 0.3},
        "source": "advanced"
    },
    "find_inline_class_opportunities": {
        "description": "Detect small, trivial classes that should be inlined",
        "category": "refactoring",
        "severity": "low",
        "default_params": {"max_methods": 3, "limit": 50, "offset": 0},
        "source": "advanced"
    },
    "find_primitive_obsession": {
        "description": "Detect primitives that should be value objects",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_usages": 5, "limit": 50, "offset": 0},
        "source": "advanced"
    },
    "find_encapsulate_collection_opportunities": {
        "description": "Detect methods returning mutable collections without encapsulation",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"limit": 50, "offset": 0},
        "source": "advanced"
    },
    "find_encapsulate_field_opportunities": {
        "description": "Find public attributes that should be private with getters/setters",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"limit": 50, "offset": 0},
        "source": "advanced"
    },
    "find_hide_delegate_opportunities": {
        "description": "Detect classes that should add delegating methods to hide dependencies",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_client_calls": 3, "limit": 50, "offset": 0},
        "source": "advanced"
    },
    "find_encapsulate_record_opportunities": {
        "description": "Find dict/record usage that should be encapsulated in a class",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_accesses": 5, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_split_variable_opportunities": {
        "description": "Find variables assigned multiple times for different purposes",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_assignments": 2, "include_loop_vars": False, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_split_loop_opportunities": {
        "description": "Find loops doing multiple things",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_operations": 2, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_pipeline_conversion_opportunities": {
        "description": "Find loops replaceable with collection pipelines",
        "category": "refactoring",
        "severity": "low",
        "default_params": {"limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_move_function_opportunities": {
        "description": "Find functions that should move to a different class",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"coupling_threshold": 0.5, "min_external_refs": 5, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_move_field_opportunities": {
        "description": "Find fields that should move to a different class",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"access_ratio_threshold": 0.6, "min_external_accesses": 3, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_move_statements_into_function_opportunities": {
        "description": "Find repeated statement sequences before/after function calls",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_duplicate_stmts": 2, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_slide_statements_opportunities": {
        "description": "Find statements that access same data but are separated",
        "category": "refactoring",
        "severity": "low",
        "default_params": {"min_gap": 2, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_replace_inline_code_opportunities": {
        "description": "Find duplicate statement sequences replaceable with function calls",
        "category": "refactoring",
        "severity": "medium",
        "default_params": {"min_sequence_length": 3, "limit": 100, "offset": 0},
        "source": "advanced"
    },

    # Inheritance Refactoring - all in AdvancedPythonTools
    "find_pull_up_method_candidates": {
        "description": "Find duplicate methods in sibling classes",
        "category": "inheritance",
        "severity": "medium",
        "default_params": {"limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_push_down_method_candidates": {
        "description": "Find superclass methods only used by some subclasses",
        "category": "inheritance",
        "severity": "medium",
        "default_params": {"limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_remove_subclass_candidates": {
        "description": "Find trivial subclasses that should be removed",
        "category": "inheritance",
        "severity": "low",
        "default_params": {"max_methods": 2, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_extract_superclass_candidates": {
        "description": "Find classes with similar methods that should share superclass",
        "category": "inheritance",
        "severity": "medium",
        "default_params": {"min_shared_methods": 2, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_collapse_hierarchy_candidates": {
        "description": "Find nearly identical parent-child pairs that should be merged",
        "category": "inheritance",
        "severity": "low",
        "default_params": {"max_additional_methods": 2, "limit": 100, "offset": 0},
        "source": "advanced"
    },
    "find_replace_with_delegate_candidates": {
        "description": "Find inheritance with low coupling (refused bequest)",
        "category": "inheritance",
        "severity": "medium",
        "default_params": {"max_coupling_ratio": 0.3, "limit": 100, "offset": 0},
        "source": "advanced"
    },

    # Dependency Analysis
    "find_circular_imports": {
        "description": "Find circular import dependencies",
        "category": "dependency",
        "severity": "high",
        "default_params": {},
        "source": "advanced"
    },
    "find_unused_code": {
        "description": "Find potentially unused code in the codebase",
        "category": "dependency",
        "severity": "medium",
        "default_params": {"limit": 50, "offset": 0},
        "source": "advanced"
    },

    # Documentation
    "find_undocumented_code": {
        "description": "Find undocumented classes and functions",
        "category": "documentation",
        "severity": "low",
        "default_params": {},
        "source": "advanced"
    },

    # Type Safety
    "find_untyped_functions": {
        "description": "Find functions/methods without return type hints",
        "category": "type_safety",
        "severity": "low",
        "default_params": {},
        "source": "advanced"
    },

    # Testing
    "find_test_files": {
        "description": "Find test files based on naming conventions",
        "category": "testing",
        "severity": "low",
        "default_params": {},
        "source": "advanced"
    },
    "find_test_fixtures": {
        "description": "Find pytest fixtures",
        "category": "testing",
        "severity": "low",
        "default_params": {},
        "source": "advanced"
    },

    # Architecture
    "find_duplicate_names": {
        "description": "Find entities with duplicate names across modules",
        "category": "architecture",
        "severity": "medium",
        "default_params": {},
        "source": "advanced"
    },

    # Exception Handling
    "detect_silent_exception_swallowing": {
        "description": "Find except blocks that silently swallow exceptions (empty or pass)",
        "category": "exception_handling",
        "severity": "critical",
        "default_params": {},
        "source": "advanced"
    },
    "detect_too_general_exceptions": {
        "description": "Find except blocks catching overly broad exceptions (Exception, BaseException, bare except)",
        "category": "exception_handling",
        "severity": "high",
        "default_params": {},
        "source": "advanced"
    },
    "detect_general_exception_raising": {
        "description": "Find raise statements using generic Exception/BaseException instead of specific types",
        "category": "exception_handling",
        "severity": "medium",
        "default_params": {},
        "source": "advanced"
    },
    "detect_error_codes_over_exceptions": {
        "description": "Find functions returning error codes (-1, None, False) instead of raising exceptions",
        "category": "exception_handling",
        "severity": "medium",
        "default_params": {},
        "source": "advanced"
    },
    "detect_finally_without_context_manager": {
        "description": "Find try/finally with cleanup (close/release/unlock) that should use 'with' statement",
        "category": "exception_handling",
        "severity": "medium",
        "default_params": {},
        "source": "advanced"
    },
    "analyze_exception_handling": {
        "description": "Comprehensive exception handling analysis - runs all exception detectors",
        "category": "exception_handling",
        "severity": "high",
        "default_params": {},
        "source": "advanced"
    },

    # RAG-based Semantic Analysis
    "detect_duplicate_code": {
        "description": "Find semantically similar code (potential duplicates) using RAG embeddings",
        "category": "duplication",
        "severity": "high",
        "default_params": {
            "similarity_threshold": 0.85,
            "max_results": 50,
            "exclude_same_file": True,
            "exclude_same_class": True,
            "entity_types": ["method", "function"]
        },
        "source": "rag"
    },
    "find_similar_clusters": {
        "description": "Find clusters of semantically similar code using K-means clustering",
        "category": "duplication",
        "severity": "medium",
        "default_params": {
            "n_clusters": 50,
            "min_cluster_size": 2,
            "exclude_same_file": True,
            "exclude_same_class": True,
            "entity_types": ["method", "function"]
        },
        "source": "rag"
    },
    "detect_extraction_opportunities": {
        "description": "Find similar code blocks that can be extracted into shared private functions/methods",
        "category": "refactoring",
        "severity": "high",
        "default_params": {
            "similarity_threshold": 0.80,
            "scope": "same_class",  # "same_class", "same_module", "any"
            "min_lines": 3,
            "max_results": 20,
            "entity_types": ["method", "function"]
        },
        "source": "rag"
    },
}


class RefactoringTool(RefactoringToolBase):
    """
    Refactoring & Improving Plugin.

    Provides two high-level tools for code analysis:
    1. refactoring_improving_prepare - Creates recommendations for all detectors
    2. refactoring_improving_detector - Runs a specific detector and stores findings

    Inherits common functionality from RefactoringToolBase.
    """

    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="refactoring_improving",
            version="1.0.0",
            description="High-level refactoring and code improvement tools with recommendation tracking",
            author="RETER Team",
            requires_reter=True,
            dependencies=["recommendations"],
            categories=["refactoring", "code-quality", "analysis"]
        )

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="prepare",
                description="Generate recommendations for running all available code detectors. Creates a work plan stored in the unified session.",
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
                            "description": "Filter detectors by category (code_smell, refactoring, inheritance, dependency, documentation, type_safety, testing, architecture). Default: all"
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
                description="Run a specific code detector and store findings as recommendations. Use 'prepare' first to see available detectors.",
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
                            "description": "Name of the detector to run (e.g., 'find_large_classes', 'detect_feature_envy')"
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
    # Tool Implementations
    # =========================================================================

    def prepare(
        self,
        instance_name: str = "default",
        categories: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        session_instance: str = "default"
    ) -> Dict[str, Any]:
        """
        Generate recommendations for running all detectors.

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

            # Import tool classes to check for method existence
            from ..python_advanced.advanced_python_tools import AdvancedPythonTools
            from ..python_advanced.refactoring_detector import RefactoringOpportunityDetector

            for name, info in DETECTORS.items():
                if categories and info["category"] not in categories:
                    continue
                if severities and info["severity"] not in severities:
                    continue

                # Check if the method actually exists on the tool class
                source = info.get("source", "unknown")
                if source == "rag":
                    # RAG detectors are always available (handled by _run_rag_detector)
                    pass
                elif source == "advanced":
                    if not hasattr(AdvancedPythonTools, name):
                        continue
                else:
                    if not hasattr(RefactoringOpportunityDetector, name):
                        continue

                filtered_detectors[name] = info

            # Generate timestamp for this prepare run
            run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Create recommendations for each detector
            created = []
            for detector_name, detector_info in filtered_detectors.items():
                # Create recommendation text
                text = f"Run detector: {detector_name}"
                description = (
                    f"{detector_info['description']}\n\n"
                    f"Category: {detector_info['category']}\n"
                    f"Default params: {detector_info['default_params']}\n\n"
                    f"Execute with: refactoring_detector(detector_name='{detector_name}')"
                )

                # Create task item in unified store (Design Docs approach)
                item_id = store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=text,
                    description=description,
                    category="refactor",  # Use TaskCategory
                    priority=self._severity_to_priority(detector_info["severity"]),
                    status="pending",
                    source_tool="refactoring_improving:prepare",
                    metadata={
                        "detector_name": detector_name,
                        "detector_category": detector_info["category"],
                        "run_timestamp": run_timestamp,
                        "default_params": detector_info["default_params"]
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
                "message": f"Created {len(created)} detector tasks",
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
        Run a specific detector and store findings as recommendations.
        """
        try:
            # Validate detector exists (with fuzzy matching for find_/detect_ prefix)
            resolved_name = self._resolve_detector_name(detector_name)
            if resolved_name is None:
                available = list(DETECTORS.keys())
                return {
                    "success": False,
                    "error": f"Unknown detector: {detector_name!r}",
                    "available_detectors": available
                }
            detector_name = resolved_name

            detector_info = DETECTORS[detector_name]

            # Get RETER instance
            reter = self.instance_manager.get_or_create_instance(instance_name)

            # Merge default params with provided params (only accept keys defined in default_params)
            effective_params = dict(detector_info["default_params"])
            if params:
                # Only update with params that are valid for this detector
                valid_keys = set(detector_info["default_params"].keys())
                for key, value in params.items():
                    if key in valid_keys:
                        effective_params[key] = value

            # Run the detector
            if detector_info["source"] == "rag":
                # RAG-based detectors use the RAG index manager
                result = self._run_rag_detector(detector_name, instance_name, effective_params)
            elif detector_info["source"] == "advanced":
                from ..python_advanced.advanced_python_tools import AdvancedPythonTools
                # Extract language from params if provided (default: "oo" for language-independent)
                language = params.get("language", "oo") if params else "oo"
                tools = AdvancedPythonTools(reter, language=language)
                method = getattr(tools, detector_name, None)
                if not method:
                    return {"success": False, "error": f"Method {detector_name} not found on AdvancedPythonTools"}
                result = method(instance_name, **effective_params)
            else:  # refactoring
                from ..python_advanced.refactoring_detector import RefactoringOpportunityDetector
                # Extract language from params if provided (default: "oo" for language-independent)
                language = params.get("language", "oo") if params else "oo"
                tools = RefactoringOpportunityDetector(reter, language=language)
                method = getattr(tools, detector_name, None)
                if not method:
                    return {"success": False, "error": f"Method {detector_name} not found on RefactoringOpportunityDetector"}
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
                link_to_thought=link_to_thought,
                create_tasks=create_tasks
            )

            # Mark any pending "Run detector: X" tasks as completed
            try:
                pending_items = store.get_items(
                    session_id=session_id,
                    item_type="task",
                    status="pending"
                )
                for item in pending_items:
                    content = item.get("content", "")
                    if content == f"Run detector: {detector_name}":
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
                "time_ms": result.get("time_ms"),
                "note": "Call items(item_type='task', category='refactor') to view details"
            }

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    # Helper methods are inherited from RefactoringToolBase:
    # _get_recommendations_tool, _count_findings, _findings_to_recommendations,
    # _extract_findings, _finding_to_text, _extract_files, _extract_entities

    def _resolve_detector_name(self, detector_name: str) -> Optional[str]:
        """
        Resolve detector name with fuzzy matching for find_/detect_ prefix variations.

        Since some detectors use 'find_' prefix and others use 'detect_' prefix,
        this allows users to use either prefix and get matched to the correct detector.

        Returns:
            Resolved detector name if found, None otherwise.
        """
        # Exact match
        if detector_name in DETECTORS:
            return detector_name

        # Try swapping find_ <-> detect_ prefix
        if detector_name.startswith("find_"):
            alt_name = "detect_" + detector_name[5:]
            if alt_name in DETECTORS:
                return alt_name
        elif detector_name.startswith("detect_"):
            alt_name = "find_" + detector_name[7:]
            if alt_name in DETECTORS:
                return alt_name

        # Try partial match (suffix match)
        # e.g., "feature_envy" -> "detect_feature_envy"
        if not detector_name.startswith(("find_", "detect_")):
            for prefix in ("find_", "detect_"):
                candidate = prefix + detector_name
                if candidate in DETECTORS:
                    return candidate

        return None

    def _run_rag_detector(
        self,
        detector_name: str,
        instance_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a RAG-based detector using the RAG index manager.

        Args:
            detector_name: Name of the detector (detect_duplicate_code, find_similar_clusters)
            instance_name: RETER instance name (used for RAG manager lookup)
            params: Parameters for the detector

        Returns:
            Detection results in standard format
        """
        import time
        start_time = time.time()

        # Get the RAG manager from the default instance manager
        default_manager = self.instance_manager.get_default_instance_manager()
        if not default_manager:
            return {
                "success": False,
                "error": "Default instance manager not available"
            }

        rag_manager = default_manager.get_rag_manager()
        if not rag_manager:
            return {
                "success": False,
                "error": "RAG index not available. Ensure RAG is enabled and initialized."
            }

        try:
            if detector_name == "detect_duplicate_code":
                # Call find_duplicate_candidates on RAG manager
                result = rag_manager.find_duplicate_candidates(
                    similarity_threshold=params.get("similarity_threshold", 0.85),
                    max_results=params.get("max_results", 50),
                    exclude_same_file=params.get("exclude_same_file", True),
                    exclude_same_class=params.get("exclude_same_class", True),
                    entity_types=params.get("entity_types", ["method", "function"])
                )

                # Transform pairs to findings format for recommendation creation
                if result.get("success") and result.get("pairs"):
                    findings = []
                    for pair in result["pairs"]:
                        findings.append({
                            "similarity": pair["similarity"],
                            "entity1_name": pair["entity1"]["name"],
                            "entity1_file": pair["entity1"]["file"],
                            "entity1_line": pair["entity1"]["line"],
                            "entity1_class": pair["entity1"].get("class_name", ""),
                            "entity2_name": pair["entity2"]["name"],
                            "entity2_file": pair["entity2"]["file"],
                            "entity2_line": pair["entity2"]["line"],
                            "entity2_class": pair["entity2"].get("class_name", ""),
                            "recommendation": f"Consider extracting common logic from '{pair['entity1']['name']}' and '{pair['entity2']['name']}' into a shared function or using delegation pattern",
                            "refactoring": "Extract Function (106) or Replace Superclass with Delegate (399)"
                        })
                    result["findings"] = findings

            elif detector_name == "find_similar_clusters":
                # Call find_similar_clusters on RAG manager
                result = rag_manager.find_similar_clusters(
                    n_clusters=params.get("n_clusters", 50),
                    min_cluster_size=params.get("min_cluster_size", 2),
                    exclude_same_file=params.get("exclude_same_file", True),
                    exclude_same_class=params.get("exclude_same_class", True),
                    entity_types=params.get("entity_types", ["method", "function"])
                )

                # Transform clusters to findings format
                if result.get("success") and result.get("clusters"):
                    findings = []
                    for cluster in result["clusters"]:
                        member_names = [m["name"] for m in cluster["members"]]
                        member_files = list(set(m["file"] for m in cluster["members"]))
                        findings.append({
                            "cluster_id": cluster["cluster_id"],
                            "member_count": cluster["member_count"],
                            "unique_files": cluster["unique_files"],
                            "members": member_names,
                            "files": member_files,
                            "avg_distance": cluster.get("avg_distance", 0),
                            "recommendation": f"Review cluster of {cluster['member_count']} similar methods across {cluster['unique_files']} files: {', '.join(member_names[:3])}{'...' if len(member_names) > 3 else ''}",
                            "refactoring": "Extract Function (106) or Pull Up Method (350)"
                        })
                    result["findings"] = findings

            elif detector_name == "detect_extraction_opportunities":
                result = self._run_extraction_opportunities_detector(
                    rag_manager=rag_manager,
                    instance_name=instance_name,
                    params=params
                )

            else:
                return {
                    "success": False,
                    "error": f"Unknown RAG detector: {detector_name}"
                }

            time_ms = int((time.time() - start_time) * 1000)
            result["time_ms"] = time_ms
            return result

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _run_extraction_opportunities_detector(
        self,
        rag_manager,
        instance_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find similar code blocks that can be extracted into shared functions/methods.

        This detector:
        1. Uses RAG to find semantically similar code pairs
        2. Filters by scope (same_class, same_module, any)
        3. Analyzes differences to determine if extraction is possible
        4. Generates extraction suggestions with parameter recommendations

        Args:
            rag_manager: The RAG index manager
            instance_name: RETER instance name
            params: Detector parameters

        Returns:
            Detection results with extraction opportunities
        """
        # Using local dataclasses (moved from code_diff.py)

        similarity_threshold = params.get("similarity_threshold", 0.80)
        scope = params.get("scope", "same_class")
        min_lines = params.get("min_lines", 3)
        max_results = params.get("max_results", 20)
        entity_types = params.get("entity_types", ["method", "function"])

        # Step 1: Get similar code pairs from RAG
        # Use a slightly lower threshold to catch more candidates
        rag_threshold = max(0.70, similarity_threshold - 0.10)

        rag_result = rag_manager.find_duplicate_candidates(
            similarity_threshold=rag_threshold,
            max_results=max_results * 3,  # Get more, will filter
            exclude_same_file=False,  # We want same-file matches for extraction
            exclude_same_class=False,  # We want same-class matches
            entity_types=entity_types
        )

        if not rag_result.get("success"):
            return rag_result

        pairs = rag_result.get("pairs", [])
        if not pairs:
            return {
                "success": True,
                "opportunities": [],
                "findings": [],
                "total_candidates": 0,
                "analyzed": 0,
                "extractable": 0
            }

        # Step 2: Filter by scope
        filtered_pairs = []
        for pair in pairs:
            e1 = pair["entity1"]
            e2 = pair["entity2"]

            # Skip if same entity
            if (e1["file"] == e2["file"] and
                e1["line"] == e2["line"]):
                continue

            if scope == "same_class":
                # Both must be in same class
                c1 = e1.get("class_name", "")
                c2 = e2.get("class_name", "")
                if not c1 or not c2 or c1 != c2:
                    continue
                # Must be in same file for same class
                if e1["file"] != e2["file"]:
                    continue

            elif scope == "same_module":
                # Both must be in same file
                if e1["file"] != e2["file"]:
                    continue

            # scope == "any" - no filtering

            filtered_pairs.append(pair)

        if not filtered_pairs:
            return {
                "success": True,
                "opportunities": [],
                "findings": [],
                "total_candidates": len(pairs),
                "analyzed": 0,
                "extractable": 0,
                "note": f"No pairs matched scope '{scope}'"
            }

        # Step 3: Analyze each pair for extraction potential
        # Get RETER instance for querying method properties
        reter = self.instance_manager.get_or_create_instance(instance_name)

        opportunities = []
        findings = []
        analyzed = 0

        for pair in filtered_pairs[:max_results * 2]:  # Limit analysis
            e1 = pair["entity1"]
            e2 = pair["entity2"]

            # Check if both have qualified names (needed for RETER queries)
            qname1 = e1.get("qualified_name", "")
            qname2 = e2.get("qualified_name", "")

            if not qname1 or not qname2:
                continue

            # Check minimum lines (from RAG entity info)
            lines1 = (e1.get("end_line", 0) or 0) - (e1.get("line", 0) or 0) + 1
            lines2 = (e2.get("end_line", 0) or 0) - (e2.get("line", 0) or 0) + 1
            if lines1 < min_lines or lines2 < min_lines:
                continue

            analyzed += 1

            # Analyze using RETER's parsed data (no tokenization needed)
            analysis = self._analyze_methods_with_reter(reter, e1, e2)

            if not analysis.extractable:
                continue

            # Filter by similarity threshold
            if analysis.similarity_score < similarity_threshold:
                continue

            # Determine placement
            if scope == "same_class" or (e1.get("class_name") and e1.get("class_name") == e2.get("class_name")):
                placement = "private_method"
                suggested_name = f"_{analysis.suggested_name.lstrip('_')}"
            elif e1["file"] == e2["file"]:
                placement = "module_function"
                suggested_name = f"_{analysis.suggested_name.lstrip('_')}"
            else:
                placement = "shared_utils"
                suggested_name = analysis.suggested_name.lstrip('_')

            # Build opportunity record
            opportunity = {
                "entity1": {
                    "name": e1["name"],
                    "file": e1["file"],
                    "line": e1.get("line"),
                    "end_line": e1.get("end_line"),
                    "class": e1.get("class_name", "")
                },
                "entity2": {
                    "name": e2["name"],
                    "file": e2["file"],
                    "line": e2.get("line"),
                    "end_line": e2.get("end_line"),
                    "class": e2.get("class_name", "")
                },
                "semantic_similarity": pair["similarity"],
                "token_similarity": analysis.similarity_score,
                "extractable": True,
                "suggested_name": suggested_name,
                "placement": placement,
                "parameters": [
                    {
                        "name": p.name,
                        "inferred_type": p.inferred_type,
                        "values": p.values,
                        "diff_type": p.diff_type
                    }
                    for p in analysis.parameters
                ],
                "diff_count": analysis.diff_count,
                "total_tokens": analysis.total_tokens,
                "common_pattern_preview": analysis.common_pattern[:200] + "..." if len(analysis.common_pattern) > 200 else analysis.common_pattern
            }

            opportunities.append(opportunity)

            # Also create a finding for recommendation system
            param_desc = ", ".join(f"{p.name}: {p.inferred_type}" for p in analysis.parameters[:3])
            if len(analysis.parameters) > 3:
                param_desc += f", ... (+{len(analysis.parameters) - 3} more)"

            findings.append({
                "entity1_name": e1["name"],
                "entity1_file": e1["file"],
                "entity1_line": e1.get("line"),
                "entity1_class": e1.get("class_name", ""),
                "entity2_name": e2["name"],
                "entity2_file": e2["file"],
                "entity2_line": e2.get("line"),
                "entity2_class": e2.get("class_name", ""),
                "similarity": pair["similarity"],
                "suggested_name": suggested_name,
                "placement": placement,
                "parameters": param_desc,
                "recommendation": f"Extract common logic from '{e1['name']}' and '{e2['name']}' into {placement.replace('_', ' ')} '{suggested_name}({param_desc})'",
                "refactoring": "Extract Function (106)"
            })

            if len(opportunities) >= max_results:
                break

        return {
            "success": True,
            "opportunities": opportunities,
            "findings": findings,
            "total_candidates": len(pairs),
            "filtered_by_scope": len(filtered_pairs),
            "analyzed": analyzed,
            "extractable": len(opportunities)
        }

    def _get_method_properties(
        self,
        reter,
        qualified_name: str,
        language_prefix: str = "py"
    ) -> Dict[str, Any]:
        """
        Query RETER for method/function properties.

        Args:
            reter: RETER wrapper instance
            qualified_name: Qualified name of the method/function
            language_prefix: Language prefix (py, js, oo)

        Returns:
            Dictionary with string_literals, number_literals, calls
        """
        props = {
            "string_literals": [],
            "number_literals": [],
            "calls": []
        }

        # Query for string literals
        query = f"""
            SELECT ?literal WHERE {{
                "{qualified_name}" hasStringLiteral ?literal
            }}
        """
        try:
            result = reter.reql(query)
            if result.num_rows > 0:
                props["string_literals"] = result.column(0).to_pylist()
        except Exception:
            pass

        # Query for number literals
        query = f"""
            SELECT ?literal WHERE {{
                "{qualified_name}" hasNumberLiteral ?literal
            }}
        """
        try:
            result = reter.reql(query)
            if result.num_rows > 0:
                props["number_literals"] = result.column(0).to_pylist()
        except Exception:
            pass

        # Query for calls
        query = f"""
            SELECT ?callee WHERE {{
                "{qualified_name}" calls ?callee
            }}
        """
        try:
            result = reter.reql(query)
            if result.num_rows > 0:
                props["calls"] = result.column(0).to_pylist()
        except Exception:
            pass

        return props

    def _analyze_methods_with_reter(
        self,
        reter,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze two methods using RETER data to determine extraction potential.

        Uses RETER's already-parsed data instead of re-tokenizing source code.
        This is more robust than using Python's tokenize module.

        Args:
            reter: RETER wrapper instance
            entity1: First entity info (name, file, line, qualified_name, etc.)
            entity2: Second entity info

        Returns:
            Analysis result with extractable, parameters, similarity, etc.
        """
        # Using local dataclasses (moved from code_diff.py)

        qname1 = entity1.get("qualified_name", "")
        qname2 = entity2.get("qualified_name", "")

        if not qname1 or not qname2:
            return ExtractionAnalysis(
                extractable=False,
                reason="Missing qualified names",
                similarity_score=0.0,
                parameters=[],
                common_pattern="",
                suggested_name="_extracted",
                diff_count=0,
                total_tokens=0
            )

        # Get properties for both methods from RETER
        props1 = self._get_method_properties(reter, qname1)
        props2 = self._get_method_properties(reter, qname2)

        # Compare call patterns (structural similarity)
        calls1 = set(props1["calls"])
        calls2 = set(props2["calls"])

        # If call patterns are very different, not extractable
        if calls1 and calls2:
            call_overlap = len(calls1 & calls2) / max(len(calls1 | calls2), 1)
        elif not calls1 and not calls2:
            call_overlap = 1.0  # Both have no calls
        else:
            call_overlap = 0.0

        if call_overlap < 0.5:
            return ExtractionAnalysis(
                extractable=False,
                reason=f"Call patterns too different ({call_overlap:.2f})",
                similarity_score=call_overlap,
                parameters=[],
                common_pattern="",
                suggested_name="_extracted",
                diff_count=0,
                total_tokens=0
            )

        # Find differing literals
        parameters = []
        param_counter = 0

        # String literal differences
        str1 = set(props1["string_literals"])
        str2 = set(props2["string_literals"])
        str_only1 = str1 - str2
        str_only2 = str2 - str1

        # Try to pair up string literals by position-like heuristic
        # If same count of unique strings, pair them
        if str_only1 and str_only2 and len(str_only1) == len(str_only2):
            for s1, s2 in zip(sorted(str_only1), sorted(str_only2)):
                param_counter += 1
                parameters.append(ParameterSuggestion(
                    name=f"text_{param_counter}",
                    inferred_type="str",
                    values=[s1, s2],
                    diff_type="string"
                ))

        # Number literal differences
        num1 = set(props1["number_literals"])
        num2 = set(props2["number_literals"])
        num_only1 = num1 - num2
        num_only2 = num2 - num1

        if num_only1 and num_only2 and len(num_only1) == len(num_only2):
            for n1, n2 in zip(sorted(num_only1), sorted(num_only2)):
                param_counter += 1
                # Determine if float or int
                inferred_type = "float" if ("." in str(n1) or "." in str(n2)) else "int"
                parameters.append(ParameterSuggestion(
                    name=f"value_{param_counter}",
                    inferred_type=inferred_type,
                    values=[str(n1), str(n2)],
                    diff_type="literal"
                ))

        # Calculate overall similarity
        total_items = (len(calls1 | calls2) + len(str1 | str2) + len(num1 | num2)) or 1
        common_items = len(calls1 & calls2) + len(str1 & str2) + len(num1 & num2)
        similarity = common_items / total_items

        # Determine extractability
        extractable = similarity >= 0.5 and call_overlap >= 0.5

        # Suggest function name based on method names
        name1 = entity1.get("name", "")
        name2 = entity2.get("name", "")

        # Find common prefix in method names
        common_prefix = ""
        for c1, c2 in zip(name1, name2):
            if c1 == c2:
                common_prefix += c1
            else:
                break

        if common_prefix and len(common_prefix) >= 3:
            suggested_name = f"_{common_prefix.rstrip('_')}_impl"
        else:
            # Default based on common calls
            if calls1 & calls2:
                first_call = sorted(calls1 & calls2)[0]
                # Extract just the method name from qualified name
                call_name = first_call.split(".")[-1] if "." in first_call else first_call
                suggested_name = f"_{call_name}"
            else:
                suggested_name = "_extracted_logic"

        diff_count = len(parameters)

        return ExtractionAnalysis(
            extractable=extractable,
            reason="Code blocks can be merged with parameterization" if extractable else "Insufficient similarity",
            similarity_score=similarity,
            parameters=parameters,
            common_pattern="",  # Not generating pattern without source code
            suggested_name=suggested_name,
            diff_count=diff_count,
            total_tokens=total_items
        )
