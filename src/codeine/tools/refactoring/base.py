"""
Base class for Refactoring Tools

Provides shared functionality for RefactoringTool and RefactoringToPatternsTool,
including recommendation creation, finding extraction, and common helper methods.

Updated for Phase 6: Uses UnifiedStore instead of RecommendationsTool.
"""

import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from codeine.tools.base import BaseTool
from codeine.services.language_support import LanguageSupport, LanguageType


class RefactoringToolBase(BaseTool):
    """
    Base class for refactoring analysis tools.

    Provides common functionality:
    - Getting unified store for item creation
    - Converting findings to recommendation items
    - Extracting findings from results
    - Creating recommendation text from findings
    """

    # Override in subclasses with detector-specific keys
    FINDING_KEYS = [
        "classes", "functions", "methods", "clumps", "groups",
        "opportunities", "candidates", "smells", "results",
        "variables", "chains", "hierarchies", "fields",
        "issues", "items", "files", "fixtures", "cycles",
        "singletons", "pairs", "clusters", "findings"
    ]

    # Language support for multi-language analysis
    language: LanguageType = "oo"
    _lang = LanguageSupport

    def _concept(self, entity: str) -> str:
        """Build concept string for current language (e.g., 'py:Class' or 'oo:Class')."""
        return self._lang.concept(entity, self.language)

    def _relation(self, rel: str) -> str:
        """Build relation string for current language (e.g., 'py:inheritsFrom')."""
        return self._lang.relation(rel, self.language)

    def _get_unified_store(self):
        """Get the UnifiedStore instance for creating items."""
        try:
            from ..unified.store import UnifiedStore
            return UnifiedStore()
        except (ImportError, OSError) as e:
            # ImportError: Module not available
            # OSError: Database file issues
            return None

    def _get_or_create_session(self, store, session_instance: str) -> Optional[str]:
        """Get or create a session and return its ID."""
        try:
            # get_or_create_session returns session_id string directly
            return store.get_or_create_session(session_instance)
        except (sqlite3.Error, OSError) as e:
            # sqlite3.Error: Database operation failed
            # OSError: File system issues
            return None

    def _severity_to_priority(self, severity: str) -> str:
        """Map detector severity to item priority."""
        mapping = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
            "info": "info"
        }
        return mapping.get(severity, "medium")

    def _get_recommendations_tool(self):
        """Get the recommendations tool instance (deprecated - use _get_unified_store)."""
        try:
            from ..recommendations.tool import RecommendationsTool
            return RecommendationsTool(self.instance_manager)
        except (ImportError, AttributeError) as e:
            # ImportError: Module not available
            # AttributeError: instance_manager not set
            return None

    def _count_findings(self, result: Dict[str, Any]) -> int:
        """Count findings in a detector result."""
        for key in self.FINDING_KEYS:
            if key in result and isinstance(result[key], list):
                return len(result[key])
        if "count" in result:
            return result["count"]
        return 0

    def _extract_findings(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract list of findings from detector result."""
        for key in self.FINDING_KEYS:
            if key in result and isinstance(result[key], list):
                return result[key]
        return []

    def _findings_to_items(
        self,
        detector_name: str,
        detector_info: Dict[str, Any],
        result: Dict[str, Any],
        store,
        session_id: str,
        category_prefix: Optional[str] = None,
        link_to_thought: Optional[str] = None,
        create_tasks: bool = False
    ) -> Dict[str, int]:
        """
        Convert detector findings to unified items (recommendations).

        Each recommendation includes a test-first refactoring workflow:
        1. Write/verify unit tests first
        2. Apply the refactoring
        3. Run tests to verify behavior is preserved

        Args:
            detector_name: Name of the detector
            detector_info: Detector metadata (category, severity)
            result: Detector result dictionary
            store: UnifiedStore instance
            session_id: Session ID for item creation
            category_prefix: Optional prefix for category (e.g., "pattern:")
            link_to_thought: Optional thought ID to link recommendations to
            create_tasks: If True, auto-create tasks for test-first workflow

        Returns:
            Dict with items_created, tasks_created, relations_created counts
        """
        findings = self._extract_findings(result)

        if not findings:
            return {"items_created": 0, "tasks_created": 0, "relations_created": 0}

        items_created = 0
        tasks_created = 0
        relations_created = 0

        cat_prefix = category_prefix or ""
        source_tool = f"{self.get_metadata().name}:{detector_name}"
        priority = self._severity_to_priority(detector_info.get("severity", "medium"))

        for i, finding in enumerate(findings):
            # Use full recommendation with test-first workflow
            text = self._finding_to_full_recommendation(detector_name, finding)
            short_text = self._finding_to_text(detector_name, finding)
            affected_files = self._extract_files(finding)
            affected_entities = self._extract_entities(finding)
            workflow_steps = self._build_refactoring_workflow(detector_name, finding)

            # Create recommendation item with full workflow
            rec_id = store.add_item(
                session_id=session_id,
                item_type="recommendation",
                content=text,
                category=f"{cat_prefix}{detector_info['category']}",
                priority=priority,
                source_tool=source_tool,
                metadata={
                    "finding": finding,
                    "detector": detector_name,
                    "workflow": workflow_steps
                }
            )
            items_created += 1

            # Add file relations (affects)
            for file_path in affected_files:
                store.add_relation(rec_id, file_path, "file", "affects")
                relations_created += 1

            # Add entity relations (affects_entity)
            for entity in affected_entities:
                store.add_relation(rec_id, entity, "entity", "affects")
                relations_created += 1

            # Link to thought if specified
            if link_to_thought:
                store.add_relation(rec_id, link_to_thought, "item", "traces")
                relations_created += 1

            # Auto-create tasks for test-first workflow
            if create_tasks and priority in ("critical", "high"):
                # Task 1: Find existing tests using code_inspection
                find_tests_task_id = store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=workflow_steps[0],  # Find existing tests
                    priority=priority,
                    status="pending",
                    category=f"{cat_prefix}{detector_info['category']}",
                    source_tool=source_tool,
                    metadata={"step": 1, "phase": "find_tests"}
                )
                store.add_relation(find_tests_task_id, rec_id, "item", "traces")
                tasks_created += 1
                relations_created += 1

                # Task 2: Ensure test coverage (depends on find tests task)
                ensure_tests_task_id = store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=workflow_steps[1],  # Ensure test coverage
                    priority=priority,
                    status="pending",
                    category=f"{cat_prefix}{detector_info['category']}",
                    source_tool=source_tool,
                    metadata={"step": 2, "phase": "ensure_coverage"}
                )
                store.add_relation(ensure_tests_task_id, rec_id, "item", "traces")
                store.add_relation(ensure_tests_task_id, find_tests_task_id, "item", "depends_on")
                tasks_created += 1
                relations_created += 2

                # Task 3: Apply refactoring (depends on ensure tests task)
                refactor_task_id = store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=workflow_steps[2],  # Apply refactoring
                    priority=priority,
                    status="pending",
                    category=f"{cat_prefix}{detector_info['category']}",
                    source_tool=source_tool,
                    metadata={"step": 3, "phase": "refactor"}
                )
                store.add_relation(refactor_task_id, rec_id, "item", "traces")
                store.add_relation(refactor_task_id, ensure_tests_task_id, "item", "depends_on")
                tasks_created += 1
                relations_created += 2

                # Task 4: Verify tests pass (depends on refactor task)
                verify_task_id = store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=workflow_steps[3],  # Verify tests pass
                    priority=priority,
                    status="pending",
                    category=f"{cat_prefix}{detector_info['category']}",
                    source_tool=source_tool,
                    metadata={"step": 4, "phase": "verify"}
                )
                store.add_relation(verify_task_id, rec_id, "item", "traces")
                store.add_relation(verify_task_id, refactor_task_id, "item", "depends_on")
                tasks_created += 1
                relations_created += 2

        return {
            "items_created": items_created,
            "tasks_created": tasks_created,
            "relations_created": relations_created
        }

    def _findings_to_recommendations(
        self,
        detector_name: str,
        detector_info: Dict[str, Any],
        result: Dict[str, Any],
        rec_tool,
        recommendations_instance: str,
        rec_id_prefix: Optional[str] = None,
        category_prefix: Optional[str] = None,
        source_tool_prefix: Optional[str] = None
    ) -> int:
        """
        Convert detector findings to recommendations using batch creation.
        DEPRECATED: Use _findings_to_items() with UnifiedStore instead.

        Args:
            detector_name: Name of the detector
            detector_info: Detector metadata (category, severity)
            result: Detector result dictionary
            rec_tool: Recommendations tool instance
            recommendations_instance: Recommendations instance name
            rec_id_prefix: Optional prefix for recommendation IDs
            category_prefix: Optional prefix for category (e.g., "pattern:")
            source_tool_prefix: Optional prefix for source tool name

        Returns:
            Number of recommendations created
        """
        findings = self._extract_findings(result)

        if not findings:
            return 0

        # Build list of recommendation dicts for batch creation
        recommendations = []
        id_prefix = rec_id_prefix or detector_name.upper()
        cat_prefix = category_prefix or ""
        src_prefix = source_tool_prefix or self.get_metadata().name

        for i, finding in enumerate(findings):
            rec_id = f"{id_prefix}_{i+1:04d}"
            text = self._finding_to_text(detector_name, finding)
            affected_files = self._extract_files(finding)
            affected_entities = self._extract_entities(finding)

            recommendations.append({
                "text": text,
                "rec_id": rec_id,
                "category": f"{cat_prefix}{detector_info['category']}",
                "severity": detector_info["severity"],
                "description": str(finding),
                "source_tool": f"{src_prefix}:{detector_name}",
                "affected_files": affected_files if affected_files else [],
                "affected_entities": affected_entities if affected_entities else []
            })

        # Create all recommendations in a single batch
        batch_result = rec_tool.rec_create_batch(
            recommendations=recommendations,
            instance_name=recommendations_instance
        )

        return batch_result.get("created", 0)

    def _finding_to_text(self, detector_name: str, finding: Dict[str, Any]) -> str:
        """Convert a single finding to recommendation text."""
        # Extract key information based on detector type
        if "name" in finding and "module" in finding:
            target = f"{finding['name']} in {finding['module']}"
        elif "qualified_name" in finding:
            target = finding['qualified_name']
        elif "class_name" in finding:
            target = finding['class_name']
        elif "method_name" in finding:
            target = finding['method_name']
        elif "function_name" in finding:
            target = finding['function_name']
        elif "file" in finding:
            target = finding['file']
        else:
            target = str(finding)[:100]

        return f"{detector_name}: {target}"

    def _build_refactoring_workflow(
        self, detector_name: str, finding: Dict[str, Any]
    ) -> List[str]:
        """
        Build the test-first refactoring workflow steps for a finding.

        Returns a list of workflow steps following the pattern:
        1. Search for existing tests using code_inspection
        2. Write new tests if needed (or verify existing tests cover behavior)
        3. Apply the refactoring
        4. Run tests to verify behavior preserved
        """
        # Extract target entity for the workflow
        if "name" in finding and "module" in finding:
            target = f"{finding['name']} in {finding['module']}"
            entity = finding['name']
            module = finding['module']
        elif "qualified_name" in finding:
            target = finding['qualified_name']
            entity = finding['qualified_name'].split('.')[-1]
            module = '.'.join(finding['qualified_name'].split('.')[:-1])
        elif "class_name" in finding:
            target = finding['class_name']
            entity = finding['class_name']
            module = finding.get('module', 'unknown')
        elif "method_name" in finding:
            target = finding['method_name']
            entity = finding['method_name']
            module = finding.get('module', finding.get('class_name', 'unknown'))
        elif "function_name" in finding:
            target = finding['function_name']
            entity = finding['function_name']
            module = finding.get('module', 'unknown')
        elif "file" in finding:
            target = finding['file']
            entity = finding['file']
            module = finding['file']
        else:
            target = str(finding)[:50]
            entity = "target"
            module = "unknown"

        return [
            f"1. FIND EXISTING TESTS: Use `code_inspection(action=\"find_tests\", target=\"{entity}\")` to search for existing tests",
            f"2. ENSURE TEST COVERAGE: If tests exist, verify they cover current behavior. If not, create unit tests for `{entity}` to capture all public methods and edge cases",
            f"3. APPLY REFACTORING: Apply {detector_name} refactoring to `{target}`. Keep changes minimal and focused. If new code is added, ensure it is also covered by unit tests",
            f"4. VERIFY TESTS PASS: Run all tests to confirm behavior is preserved. If any test fails, revert and investigate. Consider adding regression tests for any bugs found"
        ]

    def _finding_to_full_recommendation(
        self, detector_name: str, finding: Dict[str, Any]
    ) -> str:
        """
        Convert a finding to a full recommendation with test-first workflow.

        Returns a comprehensive recommendation including:
        - The refactoring description
        - Step-by-step workflow (test first, refactor, verify)
        """
        description = self._finding_to_text(detector_name, finding)
        workflow = self._build_refactoring_workflow(detector_name, finding)

        return f"{description}\n\nWorkflow:\n" + "\n".join(workflow)

    def _extract_files(self, finding: Dict[str, Any]) -> List[str]:
        """Extract file paths from a finding."""
        files = []
        for key in ["file", "files", "module", "source_file"]:
            if key in finding:
                val = finding[key]
                if isinstance(val, list):
                    files.extend(val)
                elif isinstance(val, str):
                    files.append(val)
        return files

    def _extract_entities(self, finding: Dict[str, Any]) -> List[str]:
        """Extract entity names from a finding."""
        entities = []
        for key in ["name", "class_name", "method_name", "function_name",
                    "qualified_name", "entity"]:
            if key in finding:
                val = finding[key]
                if isinstance(val, str):
                    entities.append(val)
        return entities

    def _run_detector_common(
        self,
        detector_name: str,
        detectors_registry: Dict[str, Any],
        instance_name: str,
        params: Optional[Dict[str, Any]],
        recommendations_instance: str,
        rec_id_prefix: str,
        category_prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Common logic for running a detector.

        Override to customize detector execution.
        """
        try:
            # Validate detector exists
            if detector_name not in detectors_registry:
                available = list(detectors_registry.keys())
                return {
                    "success": False,
                    "error": f"Unknown detector: {detector_name}",
                    "available_detectors": available
                }

            detector_info = detectors_registry[detector_name]

            # Merge default params with provided params
            effective_params = dict(detector_info.get("default_params", {}))
            if params:
                effective_params.update(params)

            # Subclass should override _execute_detector to run the actual detector
            result = self._execute_detector(
                detector_name, detector_info, instance_name, effective_params
            )

            if not result.get("success", True):
                return result

            # Get recommendations tool
            rec_tool = self._get_recommendations_tool()
            if not rec_tool:
                return {
                    "success": True,
                    "detector": detector_name,
                    "raw_result": result,
                    "recommendations_created": 0,
                    "warning": "Recommendations plugin not available"
                }

            # Convert findings to recommendations
            recommendations_created = self._findings_to_recommendations(
                detector_name=detector_name,
                detector_info=detector_info,
                result=result,
                rec_tool=rec_tool,
                recommendations_instance=recommendations_instance,
                rec_id_prefix=detector_name.upper(),
                category_prefix=category_prefix,
                source_tool_prefix=self.get_metadata().name
            )

            # Mark the RUN recommendation as completed if it exists
            run_rec_id = f"{rec_id_prefix}_{detector_name.upper()}"
            rec_tool.rec_update_status(
                rec_id=run_rec_id,
                status="completed",
                instance_name=recommendations_instance
            )

            return {
                "success": True,
                "detector": detector_name,
                "params_used": effective_params,
                "findings_count": self._count_findings(result),
                "recommendations_created": recommendations_created,
                "recommendations_instance": recommendations_instance,
                "time_ms": result.get("time_ms")
            }

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _execute_detector(
        self,
        detector_name: str,
        detector_info: Dict[str, Any],
        instance_name: str,
        effective_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a detector. Override in subclasses.

        Args:
            detector_name: Name of the detector to run
            detector_info: Detector metadata
            instance_name: RETER instance name
            effective_params: Parameters to pass to detector

        Returns:
            Detector result dictionary
        """
        raise NotImplementedError("Subclasses must implement _execute_detector")
