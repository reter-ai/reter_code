"""
Shared Data Classes for Tool Parameters

These dataclasses are used to reduce long parameter lists in internal methods.
MCP tool handlers maintain their individual parameter signatures for API compatibility.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SequenceDiagramOptions:
    """Options for sequence diagram generation."""
    entry_point: Optional[str] = None
    max_depth: int = 10
    exclude_patterns: Optional[List[str]] = None
    include_only_classes: Optional[List[str]] = None


@dataclass
class DependencyGraphOptions:
    """Options for dependency graph generation."""
    show_external: bool = False
    group_by_package: bool = True
    highlight_circular: bool = True
    limit: int = 100
    offset: int = 0
    module_filter: Optional[str] = None
    summary_only: bool = False
    circular_deps_limit: int = 10  # Limit circular deps in response (0 = unlimited)


@dataclass
class CallGraphOptions:
    """Options for call graph generation."""
    direction: str = "both"
    max_depth: int = 3
    exclude_patterns: Optional[List[str]] = None


@dataclass
class CouplingMatrixOptions:
    """Options for coupling matrix generation."""
    threshold: int = 0
    include_inheritance: bool = True
    max_classes: int = 20


@dataclass
class ClassDiagramOptions:
    """Options for class diagram generation."""
    include_methods: bool = True
    include_attributes: bool = True


@dataclass
class TaskParams:
    """Parameters for creating a Gantt task."""
    name: str
    description: Optional[str] = None
    start_date: Optional[str] = None
    duration_days: Optional[int] = None
    end_date: Optional[str] = None
    assigned_to: Optional[List[str]] = None
    depends_on: Optional[List[str]] = None
    phase: Optional[str] = None
    task_id: Optional[str] = None


@dataclass
class TaskUpdateParams:
    """Parameters for updating a Gantt task."""
    task_id: str
    status: Optional[str] = None
    progress_percent: Optional[int] = None
    actual_start_date: Optional[str] = None
    actual_end_date: Optional[str] = None


@dataclass
class RecommendationParams:
    """Parameters for creating a recommendation."""
    text: str
    description: Optional[str] = None
    severity: str = "medium"
    category: str = "general"
    source_tool: Optional[str] = None
    source_activity_id: Optional[str] = None
    affected_files: Optional[List[str]] = None
    affected_entities: Optional[List[str]] = None
    rec_id: Optional[str] = None


@dataclass
class ActivityParams:
    """Parameters for recording an activity."""
    recorded_tool_name: str
    reter_instance: str = "default"
    params_summary: Optional[str] = None
    result_summary: Optional[str] = None
    duration_ms: Optional[float] = None
    issues_found: int = 0
    files_analyzed: Optional[List[str]] = None


@dataclass
class ArtifactParams:
    """Parameters for recording an artifact."""
    file_path: str
    artifact_type: str
    tool_name: str
    source_files: Optional[List[str]] = None


@dataclass
class ItemsQueryFilters:
    """Filter parameters for querying items.

    Groups related filter parameters to reduce the items() function signature.
    """
    # Type and status filters
    item_type: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    phase: Optional[str] = None
    category: Optional[str] = None

    # Relation filters
    traces_to: Optional[str] = None
    traced_by: Optional[str] = None
    depends_on: Optional[str] = None
    blocks: Optional[str] = None
    affects: Optional[str] = None

    # Date filters
    start_after: Optional[str] = None
    end_before: Optional[str] = None

    # Pagination
    limit: int = 100
    offset: int = 0

    # Options
    include_relations: bool = False


@dataclass
class ThoughtInput:
    """Input parameters for creating a thought.

    Groups thought-related parameters to reduce the think() function signature.
    """
    # Core thought content (required)
    thought: str
    thought_number: int
    total_thoughts: int

    # Thought metadata
    thought_type: str = "reasoning"
    next_thought_needed: bool = True
    needs_more_thoughts: bool = False

    # Branching options
    branch_id: Optional[str] = None
    branch_from: Optional[int] = None
    is_revision: bool = False
    revises_thought: Optional[int] = None

    # Operations to execute with the thought
    operations: Optional[Dict[str, Any]] = None
