"""
Documentation Maintenance Tool

Provides detectors for analyzing documentation quality and relevance to code.
Uses RAG-based semantic similarity to match documentation to actual code.
"""

from typing import Dict, Any, Optional, List
from codeine.tools.base import ToolMetadata, ToolDefinition, BaseTool
from codeine.reter_wrapper import is_initialization_complete
from codeine.services.language_support import LanguageSupport, LanguageType


# =============================================================================
# DETECTOR REGISTRY
# =============================================================================

DETECTORS = {
    # Orphaned Documentation - Docs not matching any code
    "orphaned_sections": {
        "description": "Find documentation sections with no matching code (potentially outdated)",
        "category": "orphaned",
        "severity": "medium",
        "default_params": {
            "min_relevance": 0.4,
            "max_results": 100,
            "doc_types": ["section"],
        },
        "source": "rag",
    },
    "orphaned_code_blocks": {
        "description": "Find code examples in docs that don't match actual code",
        "category": "orphaned",
        "severity": "high",
        "default_params": {
            "min_relevance": 0.5,
            "max_results": 100,
            "doc_types": ["code_block"],
        },
        "source": "rag",
    },
    "all_orphaned_docs": {
        "description": "Find all documentation chunks with low code relevance",
        "category": "orphaned",
        "severity": "medium",
        "default_params": {
            "min_relevance": 0.45,
            "max_results": 200,
            "doc_types": ["section", "code_block", "document"],
        },
        "source": "rag",
    },

    # Outdated Documentation - Docs that may need updates
    "low_relevance_docs": {
        "description": "Find documentation with relevance below threshold (may need update)",
        "category": "outdated",
        "severity": "low",
        "default_params": {
            "min_relevance": 0.3,
            "max_relevance": 0.5,
            "max_results": 100,
        },
        "source": "rag",
    },
    "stale_api_docs": {
        "description": "Find API documentation sections that may be stale",
        "category": "outdated",
        "severity": "high",
        "default_params": {
            "min_relevance": 0.4,
            "max_results": 50,
            "doc_types": ["section", "code_block"],
            "heading_patterns": ["api", "endpoint", "method", "function", "class"],
        },
        "source": "rag",
    },

    # Documentation Quality - Overall doc health
    "documentation_coverage": {
        "description": "Analyze overall documentation coverage and quality",
        "category": "quality",
        "severity": "medium",
        "default_params": {
            "min_relevance": 0.5,
            "max_results": 200,
        },
        "source": "rag",
    },
    "undocumented_code": {
        "description": "Find code entities with no corresponding documentation",
        "category": "quality",
        "severity": "medium",
        "default_params": {
            "min_doc_similarity": 0.6,
            "code_types": ["class", "function"],
        },
        "source": "rag",
    },

    # Docstring Maintenance - Stale or missing docstrings
    "missing_docstrings": {
        "description": "Find public classes/methods/functions without docstrings",
        "category": "docstring",
        "severity": "medium",
        "default_params": {
            "include_private": False,
            "entity_types": ["class", "method", "function"],
        },
        "source": "reter",
    },
    "stale_docstrings": {
        "description": "Find docstrings that don't match their code implementation",
        "category": "docstring",
        "severity": "high",
        "default_params": {
            "min_similarity": 0.4,
            "max_results": 100,
        },
        "source": "rag",
    },
    "docstring_param_mismatch": {
        "description": "Find docstrings with parameter documentation that doesn't match actual parameters",
        "category": "docstring",
        "severity": "high",
        "default_params": {
            "check_args": True,
            "check_returns": True,
        },
        "source": "reter",
    },

    # Comment Maintenance - TODO/FIXME markers
    "stale_todo_comments": {
        "description": "Find TODO/FIXME/HACK comments that may be outdated",
        "category": "comment",
        "severity": "low",
        "default_params": {
            "comment_types": ["todo", "fixme", "hack", "xxx", "bug"],
            "max_results": 100,
        },
        "source": "rag",
    },
}


class DocumentationMaintenanceTool(BaseTool):
    """
    Documentation Maintenance analysis tool.

    Provides:
    - prepare(): List available detectors and create recommendations
    - detector(): Run a specific detector and store findings

    Uses RAG-based semantic similarity to analyze documentation relevance.
    """

    def __init__(self, instance_manager, default_manager=None, language: LanguageType = "oo"):
        """Initialize with RETER instance manager."""
        self.instance_manager = instance_manager
        self._default_manager = default_manager
        self.language = language
        self._lang = LanguageSupport

    def _concept(self, entity: str) -> str:
        """Build concept string for current language (e.g., 'py:Class' or 'oo:Class')."""
        return self._lang.concept(entity, self.language)

    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="documentation_maintenance",
            description="Analyze documentation quality and relevance to code",
            version="1.0.0",
            author="reter",
            categories=["documentation", "quality", "maintenance"]
        )

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="prepare",
                description="Generate recommendations for running documentation detectors",
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
                            "description": "Filter by category (orphaned, outdated, quality)"
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
                description="Run a specific documentation detector",
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
                            "description": "Auto-create tasks for findings",
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

    def _get_rag_manager(self):
        """Get RAG manager from the default instance manager."""
        if self._default_manager is None:
            return None
        return self._default_manager.get_rag_manager()

    def _get_unified_store(self):
        """Get UnifiedStore for creating items."""
        try:
            from ..unified.store import UnifiedStore
            return UnifiedStore()
        except (ImportError, OSError):
            return None

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
        List available documentation maintenance detectors.

        Args:
            instance_name: RETER instance to analyze
            categories: Filter by category (orphaned, outdated, quality)
            severities: Filter by severity (critical, high, medium, low)
            session_instance: Session instance for recommendations

        Returns:
            Dict with available detectors and recommendations created
        """
        # Check initialization state first
        if not is_initialization_complete():
            return {
                "success": False,
                "error": "Server is still initializing. The embedding model and code index are being loaded in the background. Please wait a few seconds and retry.",
                "status": "initializing",
                "detectors": [],
                "detector_count": 0,
            }

        # Check RAG availability
        rag_manager = self._get_rag_manager()
        if rag_manager is None:
            return {
                "success": False,
                "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable documentation analysis.",
                "detectors": [],
                "detector_count": 0,
            }

        if not rag_manager.is_initialized:
            return {
                "success": False,
                "error": "RAG index not initialized. Please wait for initialization to complete.",
                "status": "not_initialized",
                "detectors": [],
                "detector_count": 0,
            }

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
                        content=f"Run documentation detector: {det['name']} - {det['description']}",
                        category=f"documentation:{det['category']}",
                        priority=self._severity_to_priority(det['severity']),
                        status="pending",
                        source_tool="documentation_maintenance:prepare",
                        metadata={"detector": det['name'], "action": "run_detector"}
                    )
                    recommendations_created += 1
            except Exception:
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
        Run a specific documentation maintenance detector.

        Args:
            detector_name: Name of detector to run
            instance_name: RETER instance to analyze
            params: Override default parameters
            session_instance: Session for storing recommendations
            create_tasks: Auto-create tasks for findings
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

        # Check initialization state first
        if not is_initialization_complete():
            return {
                "success": False,
                "error": "Server is still initializing. The embedding model and code index are being loaded in the background. Please wait a few seconds and retry.",
                "status": "initializing",
            }

        # Check RAG availability
        rag_manager = self._get_rag_manager()
        if rag_manager is None:
            return {
                "success": False,
                "error": "RAG is not configured. Set RETER_PROJECT_ROOT to enable.",
            }

        if not rag_manager.is_initialized:
            return {
                "success": False,
                "error": "RAG index not initialized. Please wait for initialization to complete.",
                "status": "not_initialized",
            }

        detector_info = DETECTORS[detector_name]
        effective_params = dict(detector_info.get("default_params", {}))
        if params:
            effective_params.update(params)

        # Run the detector
        try:
            if detector_name in ("orphaned_sections", "orphaned_code_blocks", "all_orphaned_docs"):
                result = self._detect_orphaned_docs(
                    rag_manager, detector_name, **effective_params
                )
            elif detector_name == "low_relevance_docs":
                result = self._detect_low_relevance_docs(
                    rag_manager, **effective_params
                )
            elif detector_name == "stale_api_docs":
                result = self._detect_stale_api_docs(
                    rag_manager, **effective_params
                )
            elif detector_name == "documentation_coverage":
                result = self._detect_documentation_coverage(
                    rag_manager, **effective_params
                )
            elif detector_name == "undocumented_code":
                result = self._detect_undocumented_code(
                    rag_manager, instance_name, **effective_params
                )
            # Docstring detectors
            elif detector_name == "missing_docstrings":
                result = self._detect_missing_docstrings(
                    instance_name, **effective_params
                )
            elif detector_name == "stale_docstrings":
                result = self._detect_stale_docstrings(
                    rag_manager, **effective_params
                )
            elif detector_name == "docstring_param_mismatch":
                result = self._detect_docstring_param_mismatch(
                    instance_name, **effective_params
                )
            # Comment detectors
            elif detector_name == "stale_todo_comments":
                result = self._detect_stale_todo_comments(
                    rag_manager, **effective_params
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
            "stats": result.get("stats", {}),
            "recommendations_created": items_created.get("items_created", 0),
            "tasks_created": items_created.get("tasks_created", 0),
            "session_instance": session_instance
        }

    # =========================================================================
    # DETECTOR IMPLEMENTATIONS
    # =========================================================================

    def _detect_orphaned_docs(
        self,
        rag_manager,
        detector_name: str,
        min_relevance: float = 0.4,
        max_results: int = 100,
        doc_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find documentation that doesn't match any code.

        Uses analyze_documentation_relevance and filters for orphaned docs.
        """
        result = rag_manager.analyze_documentation_relevance(
            min_relevance=min_relevance,
            max_results=max_results,
            doc_entity_types=doc_types,
        )

        if not result.get("success"):
            return {"findings": [], "stats": {}, "error": result.get("error")}

        # Orphaned docs are those with low/zero similarity to code
        orphaned = result.get("orphaned_docs", [])

        findings = []
        for doc in orphaned:
            finding = {
                "type": "orphaned_documentation",
                "name": doc.get("doc_name", ""),
                "file": doc.get("doc_file", ""),
                "line": doc.get("doc_line", 0),
                "doc_type": doc.get("doc_type", ""),
                "heading": doc.get("doc_heading", ""),
                "similarity": doc.get("best_code_similarity", 0),
                "content_preview": doc.get("content_preview", ""),
                "recommendation": self._get_orphaned_recommendation(doc),
            }
            if doc.get("best_code_match"):
                finding["closest_code"] = doc["best_code_match"]
            findings.append(finding)

        return {
            "findings": findings,
            "stats": result.get("stats", {}),
        }

    def _detect_low_relevance_docs(
        self,
        rag_manager,
        min_relevance: float = 0.3,
        max_relevance: float = 0.5,
        max_results: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find documentation with relevance in a middle range - may need updates.
        """
        # Get all docs analyzed
        result = rag_manager.analyze_documentation_relevance(
            min_relevance=min_relevance,
            max_results=max_results,
        )

        if not result.get("success"):
            return {"findings": [], "stats": {}, "error": result.get("error")}

        # Get relevant docs but filter for those in low-medium range
        relevant = result.get("relevant_docs", [])
        orphaned = result.get("orphaned_docs", [])

        # Combine and filter for docs in the "needs review" range
        all_docs = relevant + orphaned
        findings = []

        for doc in all_docs:
            similarity = doc.get("best_code_similarity", 0)
            if min_relevance <= similarity < max_relevance:
                findings.append({
                    "type": "low_relevance_documentation",
                    "name": doc.get("doc_name", ""),
                    "file": doc.get("doc_file", ""),
                    "line": doc.get("doc_line", 0),
                    "doc_type": doc.get("doc_type", ""),
                    "similarity": similarity,
                    "content_preview": doc.get("content_preview", ""),
                    "recommendation": "Review and update this documentation - it may be partially outdated",
                    "closest_code": doc.get("best_code_match"),
                })

        # Sort by similarity (lowest first - most likely to need update)
        findings.sort(key=lambda x: x["similarity"])

        return {
            "findings": findings,
            "stats": {
                "docs_in_low_range": len(findings),
                "relevance_range": f"{min_relevance}-{max_relevance}",
            },
        }

    def _detect_stale_api_docs(
        self,
        rag_manager,
        min_relevance: float = 0.4,
        max_results: int = 50,
        doc_types: Optional[List[str]] = None,
        heading_patterns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find API documentation that may be stale.
        """
        result = rag_manager.analyze_documentation_relevance(
            min_relevance=min_relevance,
            max_results=max_results,
            doc_entity_types=doc_types,
        )

        if not result.get("success"):
            return {"findings": [], "stats": {}, "error": result.get("error")}

        patterns = heading_patterns or ["api", "endpoint", "method", "function", "class"]
        orphaned = result.get("orphaned_docs", [])

        findings = []
        for doc in orphaned:
            heading = (doc.get("doc_heading", "") or "").lower()
            name = (doc.get("doc_name", "") or "").lower()

            # Check if this looks like API documentation
            is_api_doc = any(p in heading or p in name for p in patterns)

            if is_api_doc:
                findings.append({
                    "type": "stale_api_documentation",
                    "name": doc.get("doc_name", ""),
                    "file": doc.get("doc_file", ""),
                    "line": doc.get("doc_line", 0),
                    "doc_type": doc.get("doc_type", ""),
                    "heading": doc.get("doc_heading", ""),
                    "similarity": doc.get("best_code_similarity", 0),
                    "content_preview": doc.get("content_preview", ""),
                    "recommendation": "This API documentation may reference removed or renamed code - verify and update",
                    "closest_code": doc.get("best_code_match"),
                })

        return {
            "findings": findings,
            "stats": {
                "stale_api_docs_found": len(findings),
                "patterns_checked": patterns,
            },
        }

    def _detect_documentation_coverage(
        self,
        rag_manager,
        min_relevance: float = 0.5,
        max_results: int = 200,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze overall documentation coverage.
        """
        result = rag_manager.analyze_documentation_relevance(
            min_relevance=min_relevance,
            max_results=max_results,
        )

        if not result.get("success"):
            return {"findings": [], "stats": {}, "error": result.get("error")}

        stats = result.get("stats", {})
        relevant_count = stats.get("relevant_count", 0)
        orphaned_count = stats.get("orphaned_count", 0)
        total = relevant_count + orphaned_count
        relevance_rate = stats.get("relevance_rate", 0)

        # Create summary findings
        findings = []

        # Overall health finding
        if relevance_rate < 0.5:
            findings.append({
                "type": "documentation_health",
                "severity": "high",
                "name": "Low Documentation Relevance",
                "message": f"Only {relevance_rate*100:.1f}% of documentation matches code",
                "recommendation": "Many documentation sections don't match actual code. Consider a documentation review.",
                "stats": {
                    "relevant_docs": relevant_count,
                    "orphaned_docs": orphaned_count,
                    "relevance_rate": relevance_rate,
                }
            })
        elif relevance_rate < 0.7:
            findings.append({
                "type": "documentation_health",
                "severity": "medium",
                "name": "Moderate Documentation Relevance",
                "message": f"{relevance_rate*100:.1f}% of documentation matches code",
                "recommendation": "Some documentation may need updates. Review orphaned sections.",
                "stats": {
                    "relevant_docs": relevant_count,
                    "orphaned_docs": orphaned_count,
                    "relevance_rate": relevance_rate,
                }
            })
        else:
            findings.append({
                "type": "documentation_health",
                "severity": "low",
                "name": "Good Documentation Relevance",
                "message": f"{relevance_rate*100:.1f}% of documentation matches code",
                "recommendation": "Documentation is well-aligned with code. Continue maintaining.",
                "stats": {
                    "relevant_docs": relevant_count,
                    "orphaned_docs": orphaned_count,
                    "relevance_rate": relevance_rate,
                }
            })

        return {
            "findings": findings,
            "stats": stats,
        }

    def _detect_undocumented_code(
        self,
        rag_manager,
        instance_name: str,
        min_doc_similarity: float = 0.6,
        code_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find code entities with no corresponding documentation.

        Searches from code -> docs perspective.
        """
        import numpy as np

        code_types = code_types or ["class", "function"]

        # Get all code vectors
        code_vectors = []
        for vid_str, meta in rag_manager._metadata.get("vectors", {}).items():
            vid = int(vid_str)
            if meta.get("source_type") == "python" and meta.get("entity_type") in code_types:
                code_vectors.append((vid, meta))

        if not code_vectors:
            return {"findings": [], "stats": {"error": "No code entities found"}}

        # For each code entity, find best matching documentation
        findings = []
        well_documented = 0

        for code_id, code_meta in code_vectors[:200]:  # Limit for performance
            code_embedding = rag_manager._faiss_wrapper.get_vector(code_id)
            if code_embedding is None:
                continue

            # Search for similar docs
            distances, indices = rag_manager._faiss_wrapper.search(
                np.array([code_embedding]),
                top_k=10
            )

            # Find best documentation match
            best_doc_similarity = 0.0
            best_doc = None

            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                match_meta = rag_manager._metadata.get("vectors", {}).get(str(idx))
                if not match_meta or match_meta.get("source_type") != "markdown":
                    continue

                # For normalized vectors with IP metric: distance is in [-1, 1], convert to [0, 1]
                similarity = max(0.0, min(1.0, (distances[0][i] + 1.0) / 2.0))
                if similarity > best_doc_similarity:
                    best_doc_similarity = similarity
                    best_doc = match_meta

            if best_doc_similarity < min_doc_similarity:
                findings.append({
                    "type": "undocumented_code",
                    "name": code_meta.get("name", ""),
                    "qualified_name": code_meta.get("qualified_name", ""),
                    "file": code_meta.get("file", ""),
                    "line": code_meta.get("line", 0),
                    "entity_type": code_meta.get("entity_type", ""),
                    "best_doc_similarity": float(round(best_doc_similarity, 4)),
                    "recommendation": f"Consider adding documentation for this {code_meta.get('entity_type', 'code')}",
                    "closest_doc": {
                        "name": best_doc.get("name", "") if best_doc else None,
                        "file": best_doc.get("file", "") if best_doc else None,
                    } if best_doc else None,
                })
            else:
                well_documented += 1

        # Sort by similarity (lowest first)
        findings.sort(key=lambda x: x["best_doc_similarity"])

        return {
            "findings": findings,
            "stats": {
                "code_entities_checked": len(code_vectors[:200]),
                "undocumented_count": len(findings),
                "well_documented_count": well_documented,
                "documentation_rate": float(round(
                    well_documented / len(code_vectors[:200]), 4
                )) if code_vectors else 0,
            },
        }

    def _get_orphaned_recommendation(self, doc: Dict[str, Any]) -> str:
        """Generate recommendation for orphaned documentation."""
        doc_type = doc.get("doc_type", "")
        similarity = doc.get("best_code_similarity", 0)

        if similarity == 0:
            if doc_type == "code_block":
                return "This code example doesn't match any current code - consider removing or updating"
            elif doc_type == "section":
                return "This section has no matching code - may be obsolete or need restructuring"
            else:
                return "This documentation has no code match - review for removal or update"
        else:
            return f"Low code relevance ({similarity:.2f}) - verify content is still accurate"

    # =========================================================================
    # FINDINGS TO ITEMS
    # =========================================================================

    def _findings_to_items(
        self,
        detector_name: str,
        detector_info: Dict[str, Any],
        result: Dict[str, Any],
        session_instance: str,
        create_tasks: bool,
        link_to_thought: Optional[str]
    ) -> Dict[str, Any]:
        """Convert findings to recommendations and optionally tasks."""
        store = self._get_unified_store()
        if not store:
            return {"items_created": 0, "tasks_created": 0}

        items_created = 0
        tasks_created = 0
        findings = result.get("findings", [])

        try:
            session_id = store.get_or_create_session(session_instance)
            severity = detector_info.get("severity", "medium")
            priority = self._severity_to_priority(severity)

            for finding in findings:
                # Create recommendation
                content = self._format_finding_content(finding)
                metadata = {
                    "detector": detector_name,
                    "finding_type": finding.get("type", ""),
                    "file": finding.get("file", ""),
                    "line": finding.get("line", 0),
                }

                if link_to_thought:
                    metadata["linked_thought"] = link_to_thought

                item_id = store.add_item(
                    session_id=session_id,
                    item_type="recommendation",
                    content=content,
                    category=f"documentation:{detector_info['category']}",
                    priority=priority,
                    status="pending",
                    source_tool=f"documentation_maintenance:{detector_name}",
                    metadata=metadata
                )
                items_created += 1

                # Create task if requested and high severity
                if create_tasks and severity in ("critical", "high"):
                    task_content = f"Fix: {finding.get('name', 'documentation issue')}"
                    store.add_item(
                        session_id=session_id,
                        item_type="task",
                        content=task_content,
                        category="documentation:fix",
                        priority=priority,
                        status="pending",
                        source_tool=f"documentation_maintenance:{detector_name}",
                        metadata={"recommendation_id": item_id, **metadata}
                    )
                    tasks_created += 1

        except Exception:
            pass

        return {"items_created": items_created, "tasks_created": tasks_created}

    def _format_finding_content(self, finding: Dict[str, Any]) -> str:
        """Format a finding into a readable recommendation."""
        parts = []

        finding_type = finding.get("type", "issue")
        name = finding.get("name", "")
        file = finding.get("file", "")
        line = finding.get("line", 0)

        if name:
            parts.append(f"**{finding_type}**: {name}")
        else:
            parts.append(f"**{finding_type}**")

        if file:
            location = f"{file}"
            if line:
                location += f":{line}"
            parts.append(f"Location: {location}")

        if finding.get("recommendation"):
            parts.append(f"Action: {finding['recommendation']}")

        if finding.get("similarity") is not None:
            parts.append(f"Code similarity: {finding['similarity']:.2f}")

        return " | ".join(parts)

    # =========================================================================
    # DOCSTRING DETECTORS
    # =========================================================================

    def _detect_missing_docstrings(
        self,
        instance_name: str,
        include_private: bool = False,
        entity_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find public classes/methods/functions without docstrings.

        Uses RETER's py:undocumented inference.
        """
        entity_types = entity_types or ["class", "method", "function"]
        findings = []

        try:
            reter = self.instance_manager.get_or_create_instance(instance_name)
        except Exception as e:
            return {"findings": [], "stats": {"error": str(e)}}

        for entity_type in entity_types:
            if entity_type == "method":
                concept = self._concept('Method')
            elif entity_type == "function":
                concept = self._concept('Function')
            elif entity_type == "class":
                concept = self._concept('Class')
            else:
                concept = self._concept(entity_type.capitalize())

            query = f'''
            SELECT ?entity ?name ?file ?line
            WHERE {{
                ?entity type {concept} .
                ?entity name ?name .
                ?entity inFile ?file .
                ?entity atLine ?line .
                FILTER(NOT EXISTS {{ ?entity hasDocstring ?doc }})
            }}
            '''

            try:
                result = reter.reql(query)
                if result is not None and result.num_rows > 0:
                    for row in result.to_pylist():
                        name = row.get("?name", "")

                        # Skip private entities unless include_private
                        if not include_private and name.startswith("_"):
                            continue

                        findings.append({
                            "type": "missing_docstring",
                            "name": name,
                            "entity_type": entity_type,
                            "file": row.get("?file", ""),
                            "line": int(row.get("?line", 0)),
                            "recommendation": f"Add docstring to {entity_type} '{name}'",
                        })
            except Exception:
                continue

        return {
            "findings": findings,
            "stats": {
                "missing_docstrings": len(findings),
                "entity_types_checked": entity_types,
            },
        }

    def _detect_stale_docstrings(
        self,
        rag_manager,
        min_similarity: float = 0.4,
        max_results: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find docstrings that don't semantically match their code implementation.

        Compares docstring embeddings to code body embeddings.
        """
        import numpy as np

        findings = []
        checked = 0

        # Get all code vectors with docstrings
        for vid_str, meta in list(rag_manager._metadata.get("vectors", {}).items())[:max_results * 2]:
            if meta.get("source_type") != "python":
                continue
            if not meta.get("docstring_preview"):
                continue

            checked += 1
            vid = int(vid_str)

            # Get the embedding for this entity (includes docstring + code)
            entity_embedding = rag_manager._faiss_wrapper.get_vector(vid)
            if entity_embedding is None:
                continue

            # Generate embedding for just the docstring
            docstring = meta.get("docstring_preview", "")
            if len(docstring) < 20:  # Skip very short docstrings
                continue

            try:
                docstring_embedding = rag_manager._embedding_service.generate_embedding(
                    f"docstring: {docstring}"
                )

                # Calculate similarity between docstring and full entity
                # Low similarity suggests docstring doesn't match implementation
                similarity = float(np.dot(docstring_embedding, entity_embedding) /
                                 (np.linalg.norm(docstring_embedding) * np.linalg.norm(entity_embedding)))

                if similarity < min_similarity:
                    findings.append({
                        "type": "stale_docstring",
                        "name": meta.get("name", ""),
                        "qualified_name": meta.get("qualified_name", ""),
                        "file": meta.get("file", ""),
                        "line": meta.get("line", 0),
                        "entity_type": meta.get("entity_type", ""),
                        "similarity": round(similarity, 4),
                        "docstring_preview": docstring,
                        "recommendation": f"Docstring may not match implementation (similarity: {similarity:.2f})",
                    })

            except Exception:
                continue

            if len(findings) >= max_results:
                break

        # Sort by similarity (lowest first)
        findings.sort(key=lambda x: x.get("similarity", 1))

        return {
            "findings": findings,
            "stats": {
                "entities_checked": checked,
                "stale_docstrings_found": len(findings),
                "min_similarity_threshold": min_similarity,
            },
        }

    def _detect_docstring_param_mismatch(
        self,
        instance_name: str,
        check_args: bool = True,
        check_returns: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find docstrings with parameter documentation that doesn't match actual parameters.

        Parses docstrings and compares to actual function signatures.
        """
        import re

        findings = []

        try:
            reter = self.instance_manager.get_or_create_instance(instance_name)
        except Exception as e:
            return {"findings": [], "stats": {"error": str(e)}}

        # Query methods and functions with docstrings and parameters
        method_concept = self._concept('Method')
        query = f'''
        SELECT ?entity ?name ?file ?line ?docstring ?paramName
        WHERE {{
            ?entity type {method_concept} .
            ?entity name ?name .
            ?entity inFile ?file .
            ?entity atLine ?line .
            ?entity hasDocstring ?docstring .
            OPTIONAL {{ ?entity hasParameter ?param . ?param name ?paramName }}
        }}
        '''

        try:
            result = reter.reql(query)
            if result is None or result.num_rows == 0:
                return {"findings": [], "stats": {"checked": 0}}

            # Group by entity
            entities = {}
            for row in result.to_pylist():
                entity_id = row.get("?entity", "")
                if entity_id not in entities:
                    entities[entity_id] = {
                        "name": row.get("?name", ""),
                        "file": row.get("?file", ""),
                        "line": int(row.get("?line", 0)),
                        "docstring": row.get("?docstring", ""),
                        "params": []
                    }
                param = row.get("?paramName")
                if param and param not in entities[entity_id]["params"]:
                    entities[entity_id]["params"].append(param)

            # Check each entity
            for entity_id, info in entities.items():
                docstring = info["docstring"]
                actual_params = set(info["params"]) - {"self", "cls"}

                if not docstring or not actual_params:
                    continue

                # Parse documented parameters from docstring
                # Handle both :param name: and Args:\n  name: formats
                doc_params = set()

                # Google-style: Args:\n    name (type): description
                args_match = re.search(r'Args?:\s*\n((?:\s+\w+.*\n?)+)', docstring)
                if args_match:
                    for line in args_match.group(1).split('\n'):
                        param_match = re.match(r'\s+(\w+)', line)
                        if param_match:
                            doc_params.add(param_match.group(1))

                # RST-style: :param name: description
                for match in re.finditer(r':param\s+(\w+):', docstring):
                    doc_params.add(match.group(1))

                # NumPy-style: Parameters\n----------\n  name : type
                params_match = re.search(r'Parameters?\s*\n\s*-+\s*\n((?:.*\n?)+?)(?:\n\s*\n|\Z)', docstring)
                if params_match:
                    for line in params_match.group(1).split('\n'):
                        param_match = re.match(r'\s*(\w+)\s*:', line)
                        if param_match:
                            doc_params.add(param_match.group(1))

                # Compare
                missing_in_doc = actual_params - doc_params
                extra_in_doc = doc_params - actual_params

                if check_args and (missing_in_doc or extra_in_doc):
                    finding = {
                        "type": "docstring_param_mismatch",
                        "name": info["name"],
                        "file": info["file"],
                        "line": info["line"],
                        "actual_params": list(actual_params),
                        "documented_params": list(doc_params),
                    }

                    if missing_in_doc:
                        finding["missing_in_docstring"] = list(missing_in_doc)
                        finding["recommendation"] = f"Document parameters: {', '.join(missing_in_doc)}"

                    if extra_in_doc:
                        finding["extra_in_docstring"] = list(extra_in_doc)
                        if "recommendation" in finding:
                            finding["recommendation"] += f"; Remove: {', '.join(extra_in_doc)}"
                        else:
                            finding["recommendation"] = f"Remove stale param docs: {', '.join(extra_in_doc)}"

                    findings.append(finding)

        except Exception as e:
            return {"findings": [], "stats": {"error": str(e)}}

        return {
            "findings": findings,
            "stats": {
                "entities_checked": len(entities) if 'entities' in dir() else 0,
                "mismatches_found": len(findings),
            },
        }

    # =========================================================================
    # COMMENT DETECTORS
    # =========================================================================

    def _detect_stale_todo_comments(
        self,
        rag_manager,
        comment_types: Optional[List[str]] = None,
        max_results: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find TODO/FIXME/HACK comments - these often become stale.
        """
        comment_types = comment_types or ["todo", "fixme", "hack", "xxx", "bug"]

        findings = []

        # Get all TODO-type comments
        for vid_str, meta in rag_manager._metadata.get("vectors", {}).items():
            if meta.get("source_type") != "python_comment":
                continue

            comment_type = meta.get("comment_type", "")
            if comment_type not in comment_types:
                continue

            findings.append({
                "type": "stale_todo_comment",
                "file": meta.get("file", ""),
                "line": meta.get("line", 0),
                "comment_type": comment_type.upper(),
                "content_preview": meta.get("content_preview", ""),
                "context_entity": meta.get("context_entity"),
                "recommendation": f"Review this {comment_type.upper()} comment - may be completed or obsolete",
            })

            if len(findings) >= max_results:
                break

        return {
            "findings": findings,
            "stats": {
                "todo_comments_found": len(findings),
                "types_checked": comment_types,
            },
        }
