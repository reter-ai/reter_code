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
