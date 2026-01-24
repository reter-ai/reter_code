"""
Data models for RETER Logical Thinking MCP Server

Pydantic models for reasoning sessions, thoughts, and knowledge structures.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from uuid import uuid4, UUID
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class ThoughtType(str, Enum):
    """Type of logical thinking step"""
    REASONING = "reasoning"          # General reasoning (default)
    ASSERTION = "assertion"          # Adding facts or axioms
    QUERY = "query"                  # Querying knowledge
    INFERENCE = "inference"          # Deriving new facts
    HYPOTHESIS = "hypothesis"        # Proposing hypothesis
    VERIFICATION = "verification"    # Verifying hypothesis
    ANALYSIS = "analysis"            # Analyzing code


class LogicType(str, Enum):
    """Type of logic system"""
    DL = "dl"                # Description Logic
    REQL = "reql"            # RETER Query Language
    PYTHON = "python"        # Python semantics


# ============================================================================
# Core Data Models
# ============================================================================

class WME(BaseModel):
    """Working Memory Element - Basic fact unit"""
    subject: str
    predicate: str
    object: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Inference(BaseModel):
    """Represents a derived fact with justification"""
    fact: WME
    rule: str
    premises: List[WME]
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LogicalThought(BaseModel):
    """Single thought in logical reasoning chain (mirrors Sequential Thinking)"""
    thought_id: UUID = Field(default_factory=uuid4)
    thought: str  # The reasoning step description
    thought_number: int = Field(..., ge=1)
    total_thoughts: int = Field(..., ge=1)
    next_thought_needed: bool = True

    # Thought type and logic operations
    thought_type: ThoughtType = ThoughtType.REASONING
    logic_operation: Optional[Dict[str, Any]] = None

    # Results from logic operations
    query_results: Optional[Dict[str, Any]] = None
    inferences: Optional[List[Inference]] = None
    contradictions: Optional[List[str]] = None

    # Metadata
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    justification: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)

    # Navigation (identical to Sequential Thinking)
    is_revision: bool = False
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: bool = False

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LogicalSession(BaseModel):
    """Logical reasoning session with thought history"""
    session_id: UUID = Field(default_factory=uuid4)
    goal: Optional[str] = None
    context: Optional[str] = None

    thought_history: List[LogicalThought] = Field(default_factory=list)
    branches: Dict[str, List[LogicalThought]] = Field(default_factory=dict)

    # RETER knowledge state
    knowledge_stats: Dict[str, int] = Field(default_factory=dict)
    loaded_sources: List[str] = Field(default_factory=list)

    status: Literal["active", "completed", "paused", "failed"] = "active"

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class OntologyInfo(BaseModel):
    """Metadata about loaded ontology"""
    namespace: str
    source: str
    triple_count: int = 0
    class_count: int = 0
    property_count: int = 0
    loaded_at: datetime = Field(default_factory=datetime.utcnow)


class PythonAnalysis(BaseModel):
    """Results from Python code analysis"""
    module_name: str
    file_path: Optional[str] = None
    classes: List[Dict[str, Any]] = Field(default_factory=list)
    functions: List[Dict[str, Any]] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    call_graph: List[Dict[str, str]] = Field(default_factory=list)
    complexity_metrics: Optional[Dict[str, int]] = None


# ============================================================================
# Tool Input Models
# ============================================================================

class LogicalThinkingInput(BaseModel):
    """Input for logical_thinking tool - mirrors Sequential Thinking"""
    thought: str = Field(..., description="Your current reasoning step")
    next_thought_needed: bool
    thought_number: int = Field(..., ge=1)
    total_thoughts: int = Field(..., ge=1)

    # Extensions for logic
    thought_type: ThoughtType = ThoughtType.REASONING
    logic_operation: Optional[Dict[str, Any]] = None

    # Branching/Revision (identical to Sequential Thinking)
    is_revision: bool = False
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: bool = False

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "thought": "Loading domain ontology to establish facts",
            "thought_type": "assertion",
            "logic_operation": {
                "assert": "Person is_a Animal",
                "source": "domain"
            },
            "thought_number": 1,
            "total_thoughts": 5,
            "next_thought_needed": True
        }
    })


class ThinkingRequest(BaseModel):
    """
    Parameter object for logical_thinking internal implementation.

    Combines instance_name with all LogicalThinkingInput fields to reduce
    parameter passing complexity (Fowler's "Introduce Parameter Object" pattern).
    """
    instance_name: str
    thought: str = Field(..., description="Your current reasoning step")
    next_thought_needed: bool
    thought_number: int = Field(..., ge=1)
    total_thoughts: int = Field(..., ge=1)

    # Extensions for logic
    thought_type: ThoughtType = ThoughtType.REASONING
    logic_operation: Optional[Dict[str, Any]] = None

    # Branching/Revision (identical to Sequential Thinking)
    is_revision: bool = False
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: bool = False


class AddKnowledgeInput(BaseModel):
    """Auxiliary tool for incrementally adding knowledge to RETER

    RETER is an incremental reasoner - knowledge accumulates, not replaces.
    Each call adds facts/rules to the existing knowledge base.
    """
    source: str = Field(..., description="File path or ontology content to add")
    type: Literal["ontology", "python", "facts"] = "ontology"
    source_id: Optional[str] = Field(None, description="Optional source identifier for later selective forgetting")


class QuickQueryInput(BaseModel):
    """Auxiliary tool for quick queries"""
    query: str
    type: Literal["reql", "dl", "pattern"] = "reql"


class ForgetSourceInput(BaseModel):
    """Input for forget_source tool"""
    source: str = Field(..., description="Source identifier to forget")


class SaveStateInput(BaseModel):
    """Input for save_state tool"""
    filename: str = Field(..., description="Path to save file")


class LoadStateInput(BaseModel):
    """Input for load_state tool"""
    filename: str = Field(..., description="Path to load file")


# ============================================================================
# Tool Output Models
# ============================================================================

class LogicalThinkingOutput(BaseModel):
    """Output for logical_thinking tool"""
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool

    # Logic results (optional)
    query_results: Optional[Dict[str, Any]] = None
    inferences: Optional[List[Inference]] = None
    contradictions: Optional[List[str]] = None

    # Session info
    branches: List[str] = Field(default_factory=list)
    session_status: str = "active"
    thought_history_length: int = 0


class AddKnowledgeOutput(BaseModel):
    """Output for add_knowledge tool

    Reports how many facts/rules were incrementally added to RETER.
    items_added accumulates with each call - knowledge persists!
    """
    success: bool
    items_added: int = 0  # Renamed from items_loaded - emphasizes incremental nature
    execution_time_ms: float = 0  # Renamed from load_time_ms
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class QueryOutput(BaseModel):
    """Output for query tools"""
    success: bool
    results: Optional[Dict[str, Any]] = None
    count: int = 0
    execution_time_ms: float = 0
    error: Optional[str] = None


class ForgetSourceOutput(BaseModel):
    """Output for forget_source tool"""
    success: bool
    message: str
    execution_time_ms: float = 0


class StateOperationOutput(BaseModel):
    """Output for save/load state operations"""
    success: bool
    filename: str
    message: str
    execution_time_ms: float = 0


class ConsistencyCheckOutput(BaseModel):
    """Output for consistency check"""
    consistent: bool
    contradictions: List[str] = Field(default_factory=list)
    unsatisfiable: List[str] = Field(default_factory=list)
    check_time_ms: float = 0
