"""
Redundancy Reduction Tool

Provides detectors for finding duplicate and similar code using RAG-based
semantic similarity analysis. Helps identify code that could be consolidated.
"""

from typing import Dict, Any, Optional, List
from codeine.tools.base import ToolMetadata, ToolDefinition, BaseTool
from codeine.reter_wrapper import is_initialization_complete


# =============================================================================
# DETECTOR REGISTRY
# =============================================================================

DETECTORS = {
    # Similarity-based detection
    "similar_clusters": {
        "description": "Find clusters of semantically similar code across files",
        "category": "similarity",
        "severity": "medium",
        "default_params": {
            "n_clusters": 50,
            "min_cluster_size": 2,
            "exclude_same_file": True,
            "exclude_same_class": True,
            "entity_types": None,
            "source_type": None,
        },
        "source": "rag",
    },
    "duplicate_candidates": {
        "description": "Find pairs of highly similar code (potential duplicates)",
        "category": "duplicates",
        "severity": "high",
        "default_params": {
            "similarity_threshold": 0.85,
            "max_results": 50,
            "exclude_same_file": True,
            "exclude_same_class": True,
            "entity_types": None,
        },
        "source": "rag",
    },
}


class RedundancyReductionTool(BaseTool):
    """
    Redundancy Reduction analysis tool.

    Uses RAG-based semantic similarity to find:
    - Clusters of similar code that could be unified
    - Pairs of near-duplicate code that should be consolidated

    Provides:
    - prepare(): List available detectors
    - detector(): Run a specific detector and store findings
    """

    def __init__(self, instance_manager, default_manager=None):
        """Initialize with RETER instance manager and default manager."""
        self.instance_manager = instance_manager
        self._default_manager = default_manager

    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="redundancy_reduction",
            description="Find duplicate and similar code for consolidation",
            version="1.0.0",
        )

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="prepare",
                description="List available redundancy reduction detectors",
                parameters={},
            ),
            ToolDefinition(
                name="detector",
                description="Run a specific redundancy reduction detector",
                parameters={
                    "detector_name": "Name of detector to run",
                    "params": "Override default parameters",
                },
            ),
        ]

    def _get_unified_store(self):
        """Get the UnifiedStore instance for creating items."""
        try:
            from ..unified.store import UnifiedStore
            return UnifiedStore()
        except (ImportError, OSError):
            return None

    def _get_or_create_session(self, store, session_instance: str) -> Optional[str]:
        """Get or create a session and return its ID."""
        try:
            return store.get_or_create_session(session_instance)
        except Exception:
            return None

    def _get_rag_manager(self):
        """Get the RAG manager if available."""
        try:
            if self._default_manager:
                return self._default_manager.get_rag_manager()
            return None
        except Exception:
            return None

    def _get_reter(self, instance_name: str):
        """Get RETER instance."""
        return self.instance_manager[instance_name]

    def prepare(
        self,
        instance_name: str = "default",
        categories: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        session_instance: str = "default"
    ) -> Dict[str, Any]:
        """
        List available redundancy reduction detectors.

        Args:
            instance_name: RETER instance name
            categories: Filter by category
            severities: Filter by severity
            session_instance: Session for storing recommendations

        Returns:
            Dict with available detectors grouped by category
        """
        filtered = {}
        for name, info in DETECTORS.items():
            if categories and info["category"] not in categories:
                continue
            if severities and info["severity"] not in severities:
                continue
            filtered[name] = info

        by_category = {}
        for name, info in filtered.items():
            cat = info["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                "name": name,
                "description": info["description"],
                "severity": info["severity"],
            })

        return {
            "success": True,
            "detectors": filtered,
            "by_category": by_category,
            "total": len(filtered),
        }

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
        Run a specific redundancy reduction detector.

        Args:
            detector_name: Name of the detector to run
            instance_name: RETER instance name
            params: Override default parameters
            session_instance: Session for storing findings
            create_tasks: Create tasks from findings
            link_to_thought: Link tasks to a thought ID

        Returns:
            Dict with findings and metadata
        """
        if detector_name not in DETECTORS:
            return {
                "success": False,
                "error": f"Unknown detector: {detector_name}",
                "available": list(DETECTORS.keys()),
            }

        detector_info = DETECTORS[detector_name]

        # Merge params with defaults
        final_params = dict(detector_info.get("default_params", {}))
        if params:
            final_params.update(params)

        # Get RAG manager
        rag_manager = self._get_rag_manager()
        if rag_manager is None:
            return {
                "success": False,
                "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable.",
                "findings": [],
            }

        # Check sync status and auto-sync if stale
        synced = False
        try:
            reter = self._get_reter(instance_name)
            sync_status = rag_manager.get_sync_status(reter)
            if not sync_status.get("is_synced", True):
                project_root = self._default_manager.project_root
                if project_root:
                    rag_manager.sync_sources(reter, project_root)
                    synced = True
        except Exception:
            pass

        # Run the appropriate detector
        try:
            if detector_name == "similar_clusters":
                result = self._run_similar_clusters(rag_manager, final_params)
            elif detector_name == "duplicate_candidates":
                result = self._run_duplicate_candidates(rag_manager, final_params)
            else:
                return {"success": False, "error": f"Detector not implemented: {detector_name}"}

            if synced:
                result["auto_synced"] = True

            # Create tasks if requested
            if create_tasks and result.get("success"):
                tasks_created = self._create_tasks_from_findings(
                    detector_name=detector_name,
                    detector_info=detector_info,
                    result=result,
                    session_instance=session_instance,
                    link_to_thought=link_to_thought,
                )
                result["tasks_created"] = tasks_created

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "findings": [],
            }

    def _run_similar_clusters(
        self,
        rag_manager,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the similar_clusters detector."""
        result = rag_manager.find_similar_clusters(
            n_clusters=params.get("n_clusters", 50),
            min_cluster_size=params.get("min_cluster_size", 2),
            exclude_same_file=params.get("exclude_same_file", True),
            exclude_same_class=params.get("exclude_same_class", True),
            entity_types=params.get("entity_types"),
            source_type=params.get("source_type"),
        )

        # Transform to standard format
        findings = []
        for cluster in result.get("clusters", []):
            findings.append({
                "type": "similar_cluster",
                "cluster_id": cluster.get("cluster_id"),
                "member_count": cluster.get("member_count", 0),
                "unique_files": cluster.get("unique_files", 0),
                "avg_distance": cluster.get("avg_distance", 0),
                "members": cluster.get("members", []),
                "severity_score": cluster.get("member_count", 0) * 10,
            })

        return {
            "success": result.get("success", True),
            "detector": "similar_clusters",
            "findings": findings,
            "count": len(findings),
            "total_vectors_analyzed": result.get("total_vectors_analyzed", 0),
            "time_ms": result.get("time_ms", 0),
        }

    def _run_duplicate_candidates(
        self,
        rag_manager,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the duplicate_candidates detector."""
        result = rag_manager.find_duplicate_candidates(
            similarity_threshold=params.get("similarity_threshold", 0.85),
            max_results=params.get("max_results", 50),
            exclude_same_file=params.get("exclude_same_file", True),
            exclude_same_class=params.get("exclude_same_class", True),
            entity_types=params.get("entity_types"),
        )

        # Transform to standard format
        findings = []
        for pair in result.get("pairs", []):
            similarity = pair.get("similarity", 0)
            findings.append({
                "type": "duplicate_pair",
                "similarity": similarity,
                "entity1": pair.get("entity1", {}),
                "entity2": pair.get("entity2", {}),
                "severity_score": int(similarity * 100),
            })

        return {
            "success": result.get("success", True),
            "detector": "duplicate_candidates",
            "findings": findings,
            "count": len(findings),
            "time_ms": result.get("time_ms", 0),
        }

    def _create_tasks_from_findings(
        self,
        detector_name: str,
        detector_info: Dict[str, Any],
        result: Dict[str, Any],
        session_instance: str,
        link_to_thought: Optional[str] = None
    ) -> int:
        """Create tasks from detector findings."""
        store = self._get_unified_store()
        if not store:
            return 0

        session_id = self._get_or_create_session(store, session_instance)
        if not session_id:
            return 0

        findings = result.get("findings", [])
        tasks_created = 0
        source_tool = f"redundancy_reduction:{detector_name}"

        for finding in findings:
            # Build task content based on finding type
            if finding.get("type") == "similar_cluster":
                members = finding.get("members", [])
                if len(members) >= 2:
                    names = [m.get("name", "?") for m in members[:3]]
                    content = f"Consolidate similar code: {', '.join(names)} [{finding.get('member_count', 0)} instances]"
                else:
                    continue
            elif finding.get("type") == "duplicate_pair":
                e1 = finding.get("entity1", {})
                e2 = finding.get("entity2", {})
                similarity = finding.get("similarity", 0)
                content = f"Merge duplicates: {e1.get('name', '?')} and {e2.get('name', '?')} [{int(similarity * 100)}% similar]"
            else:
                continue

            severity_score = finding.get("severity_score", 0)
            priority = "high" if severity_score >= 80 else "medium" if severity_score >= 50 else "low"

            try:
                store.add_item(
                    session_id=session_id,
                    item_type="task",
                    content=content,
                    category="refactor",
                    priority=priority,
                    status="pending",
                    source_tool=source_tool,
                    severity_score=severity_score,
                    metadata=finding,
                )
                tasks_created += 1
            except Exception:
                continue

        return tasks_created
