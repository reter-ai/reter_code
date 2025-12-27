"""
Refactoring Detectors - Identify refactoring opportunities.

All tools in this package use @detector decorator and produce recommendations.

Includes:
- extract_method: Long methods that should be split
- extract_class: Classes with multiple responsibilities
- inline_method: Trivial methods that should be inlined
- inline_class: Trivial classes that should be inlined
- move_method: Methods in the wrong class
- move_field: Fields in the wrong class
- rename_method: Poorly named methods
- push_down_method: Methods to push to subclasses
- pull_up_method: Methods to pull to superclass
- introduce_parameter_object: Parameter groups to extract
- attribute_data_clumps: Attribute groups to extract
- duplicate_parameter_lists: Identical parameter signatures
- encapsulate_collection: Mutable collections needing encapsulation
- hide_delegate: Delegate hiding opportunities
- remove_middle_man: Unnecessary delegation
- replace_conditional_with_polymorphism: Type-based conditionals
- replace_inheritance_with_delegation: Inheritance to composition
- split_loop: Loops doing multiple things
- pipeline_conversion: Loops to collection pipelines
- move_statements: Statements to move into functions
- slide_statements: Related statements to group
- replace_inline_code: Duplicate code to extract
"""

from .extract_method import extract_method
from .extract_class import extract_class
from .inline_method import inline_method
from .inline_class import inline_class
from .move_method import move_method
from .move_field import move_field
from .rename_method import rename_method
from .push_down_method import push_down_method
from .pull_up_method import pull_up_method
from .introduce_parameter_object import introduce_parameter_object
from .attribute_data_clumps import attribute_data_clumps
from .duplicate_parameter_lists import duplicate_parameter_lists
from .replace_conditional_with_polymorphism import replace_conditional_with_polymorphism
from .replace_inheritance_with_delegation import replace_inheritance_with_delegation
from .encapsulate_collection import encapsulate_collection
from .hide_delegate import hide_delegate
from .remove_middle_man import remove_middle_man
from .split_loop import split_loop
from .pipeline_conversion import pipeline_conversion
from .move_statements import move_statements
from .slide_statements import slide_statements
from .replace_inline_code import replace_inline_code

# Re-export from rag module (has category="refactoring" but lives in rag/)
from ..rag.extraction_opportunities import detect_extraction_opportunities

__all__ = [
    "extract_method",
    "extract_class",
    "inline_method",
    "inline_class",
    "move_method",
    "move_field",
    "rename_method",
    "push_down_method",
    "pull_up_method",
    "introduce_parameter_object",
    "attribute_data_clumps",
    "duplicate_parameter_lists",
    "replace_conditional_with_polymorphism",
    "replace_inheritance_with_delegation",
    "encapsulate_collection",
    "hide_delegate",
    "remove_middle_man",
    "split_loop",
    "pipeline_conversion",
    "move_statements",
    "slide_statements",
    "replace_inline_code",
    "detect_extraction_opportunities",
]
