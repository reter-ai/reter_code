"""
Data classes for unified tools.

Contains type-safe containers for passing structured data between components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ThoughtInput:
    """
    Input parameters for creating a thought in the thinking session.

    @reter-cnl: This is-in-layer Utility-Layer.
    @reter-cnl: This is a value-object.

    Groups all parameters needed to create a thought item,
    reducing parameter passing overhead and improving type safety.
    """

    thought: str
    thought_number: int
    total_thoughts: int
    thought_type: str = "reasoning"
    section: Optional[str] = None
    next_thought_needed: bool = True
    needs_more_thoughts: bool = False
    branch_id: Optional[str] = None
    branch_from: Optional[int] = None
    is_revision: bool = False
    revises_thought: Optional[int] = None
    operations: Optional[Dict[str, Any]] = None


# ============================================================
# THOUGHT STORE DATA CLASSES
# ============================================================

@dataclass
class LogicData:
    """
    Logic operation results for a thought.

    @reter-cnl: This is-in-layer Utility-Layer.
    @reter-cnl: This is a value-object.

    Groups data from logical reasoning operations performed during thinking.
    """
    logic_operation: Optional[Dict] = None
    query_results: Optional[Dict] = None
    inferences: Optional[List] = None
    contradictions: Optional[List[str]] = None


@dataclass
class ReasoningMeta:
    """
    Reasoning metadata for a thought.

    @reter-cnl: This is-in-layer Utility-Layer.
    @reter-cnl: This is a value-object.

    Groups confidence and justification data for a thought.
    """
    confidence: float = 1.0
    justification: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None


@dataclass
class BranchingInfo:
    """
    Branching and revision information for a thought.

    @reter-cnl: This is-in-layer Utility-Layer.
    @reter-cnl: This is a value-object.

    Groups all branching-related parameters for thought chains.
    """
    is_revision: bool = False
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: bool = False


@dataclass
class ThoughtStoreData:
    """
    Complete data for storing a thought in LogicalThinkingStore.

    @reter-cnl: This is-in-layer Utility-Layer.
    @reter-cnl: This is a value-object.

    Consolidates 19 individual parameters into a structured object with
    logical groupings:
    - Core: thought content and numbering
    - Logic: operation results from reasoning
    - Reasoning: confidence and justification metadata
    - Branching: revision and branch information
    """
    # Core thought data
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    thought_type: str = "reasoning"

    # Grouped sub-objects
    logic: LogicData = field(default_factory=LogicData)
    reasoning: ReasoningMeta = field(default_factory=ReasoningMeta)
    branching: BranchingInfo = field(default_factory=BranchingInfo)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for storage."""
        inferences = self.logic.inferences
        if inferences:
            inferences = [i.dict() if hasattr(i, 'dict') else i for i in inferences]

        return {
            "thought_number": self.thought_number,
            "total_thoughts": self.total_thoughts,
            "next_thought_needed": self.next_thought_needed,
            "thought_type": self.thought_type,
            "logic_operation": self.logic.logic_operation,
            "query_results": self.logic.query_results,
            "inferences": inferences,
            "contradictions": self.logic.contradictions,
            "confidence": self.reasoning.confidence,
            "justification": self.reasoning.justification,
            "assumptions": self.reasoning.assumptions,
            "is_revision": self.branching.is_revision,
            "revises_thought": self.branching.revises_thought,
            "branch_from_thought": self.branching.branch_from_thought,
            "branch_id": self.branching.branch_id,
            "needs_more_thoughts": self.branching.needs_more_thoughts,
        }

    @classmethod
    def from_params(
        cls,
        thought: str,
        thought_number: int,
        total_thoughts: int,
        next_thought_needed: bool,
        thought_type: str = "reasoning",
        logic_operation: Optional[Dict] = None,
        query_results: Optional[Dict] = None,
        inferences: Optional[List] = None,
        contradictions: Optional[List[str]] = None,
        confidence: float = 1.0,
        justification: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        is_revision: bool = False,
        revises_thought: Optional[int] = None,
        branch_from_thought: Optional[int] = None,
        branch_id: Optional[str] = None,
        needs_more_thoughts: bool = False,
    ) -> "ThoughtStoreData":
        """Create ThoughtStoreData from individual parameters.

        Provides backward compatibility with the original 19-parameter signature.
        """
        return cls(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_thought_needed=next_thought_needed,
            thought_type=thought_type,
            logic=LogicData(
                logic_operation=logic_operation,
                query_results=query_results,
                inferences=inferences,
                contradictions=contradictions,
            ),
            reasoning=ReasoningMeta(
                confidence=confidence,
                justification=justification,
                assumptions=assumptions,
            ),
            branching=BranchingInfo(
                is_revision=is_revision,
                revises_thought=revises_thought,
                branch_from_thought=branch_from_thought,
                branch_id=branch_id,
                needs_more_thoughts=needs_more_thoughts,
            ),
        )
