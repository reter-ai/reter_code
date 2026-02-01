"""
Language Support Module for Multi-Language Code Analysis.

This module provides utilities for language-independent code analysis.

NOTE: REQL queries use CNL (Controlled Natural Language) naming conventions:
- Types use plain lowercase names: class, method, function
- Predicates use hyphenated format: has-name, is-in-file, is-defined-in, inherits-from

Usage:
    from reter_code.services.language_support import LanguageSupport, lang

    # Get prefix for a language (legacy support)
    prefix = LanguageSupport.get_prefix("python")  # Returns "py"
    prefix = LanguageSupport.get_prefix("oo")      # Returns "oo" (default)

    # Build relation string (legacy support)
    relation = lang.relation("inherits-from", "py")  # Returns 'py:inherits-from'
"""

from typing import Literal, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """
    Supported programming languages for code analysis.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    OO = "oo"           # Object-Oriented meta-ontology (language-independent)
    PYTHON = "py"       # Python
    JAVASCRIPT = "js"   # JavaScript
    HTML = "html"       # HTML documents
    CSHARP = "cs"       # C#
    CPP = "cpp"         # C++

    @classmethod
    def from_string(cls, value) -> "Language":
        """Convert string to Language enum, with flexible matching."""
        # If already a Language, return it
        if isinstance(value, Language):
            return value
        value_lower = value.lower().strip()
        mapping = {
            "oo": cls.OO,
            "object-oriented": cls.OO,
            "generic": cls.OO,
            "all": cls.OO,
            "py": cls.PYTHON,
            "python": cls.PYTHON,
            "js": cls.JAVASCRIPT,
            "javascript": cls.JAVASCRIPT,
            "html": cls.HTML,
            "htm": cls.HTML,
            "cs": cls.CSHARP,
            "csharp": cls.CSHARP,
            "c#": cls.CSHARP,
            "cpp": cls.CPP,
            "c++": cls.CPP,
            "cxx": cls.CPP,
        }
        if value_lower in mapping:
            return mapping[value_lower]
        raise ValueError(f"Unsupported language: {value}. Supported: oo, python, javascript, html, csharp, cpp")


# Type alias for language parameter
LanguageType = Literal["oo", "py", "python", "js", "javascript", "html", "htm", "cs", "csharp", "c#", "cpp", "c++", "cxx", "all", "generic"]


@dataclass
class EntityMapping:
    """
    Maps generic OO concepts to language-specific concepts.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    oo: str           # Generic OO concept
    py: Optional[str] = None  # Python-specific (defaults to oo)
    js: Optional[str] = None  # JavaScript-specific (defaults to oo)
    html: Optional[str] = None  # HTML-specific (defaults to oo)
    cs: Optional[str] = None  # C#-specific (defaults to oo)
    cpp: Optional[str] = None  # C++-specific (defaults to oo)

    def get(self, language: Language) -> str:
        """Get the concept name for a specific language."""
        if language == Language.PYTHON and self.py:
            return self.py
        if language == Language.JAVASCRIPT and self.js:
            return self.js
        if language == Language.HTML and self.html:
            return self.html
        if language == Language.CSHARP and self.cs:
            return self.cs
        if language == Language.CPP and self.cpp:
            return self.cpp
        return self.oo


class LanguageSupport:
    """
    Central class for managing language-specific ontology prefixes.

    ::: This is-in-layer Utility-Layer.
    ::: This is a utility.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    This enables language-independent code analysis tools by abstracting
    the ontology prefix (oo:, py:, js:) from the query logic.
    """

    # Mapping of language to ontology prefix
    PREFIXES: Dict[Language, str] = {
        Language.OO: "oo",
        Language.PYTHON: "py",
        Language.JAVASCRIPT: "js",
        Language.HTML: "html",
        Language.CSHARP: "cs",
        Language.CPP: "cpp",
    }

    # Entity concept mappings (some entities have different names per language)
    ENTITY_MAPPINGS: Dict[str, EntityMapping] = {
        # Core entities (same across languages)
        "CodeEntity": EntityMapping("CodeEntity"),
        "Module": EntityMapping("Module"),
        "Class": EntityMapping("Class"),
        "Function": EntityMapping("Function"),
        "Method": EntityMapping("Method"),
        "Constructor": EntityMapping("Constructor"),
        "Parameter": EntityMapping("Parameter"),
        "Import": EntityMapping("Import"),
        "Export": EntityMapping("Export"),
        "Assignment": EntityMapping("Assignment"),

        # Field/Attribute mapping (Python uses Attribute, JS uses Field)
        "Field": EntityMapping("Field", py="Attribute"),
        "Attribute": EntityMapping("Field", py="Attribute"),

        # Error handling
        "TryBlock": EntityMapping("TryBlock"),
        "CatchClause": EntityMapping("CatchClause"),
        "ThrowStatement": EntityMapping("ThrowStatement"),
        "ReturnStatement": EntityMapping("ReturnStatement"),
        "Call": EntityMapping("Call"),

        # Python-specific (map to generic when possible)
        "ExceptHandler": EntityMapping("CatchClause", py="ExceptHandler"),
        "RaiseStatement": EntityMapping("ThrowStatement", py="RaiseStatement"),
        "FinallyBlock": EntityMapping("TryBlock", py="FinallyBlock"),  # Simplified

        # JavaScript-specific
        "ArrowFunction": EntityMapping("Function", js="ArrowFunction"),
        "Variable": EntityMapping("Field", js="Variable"),
        "FinallyClause": EntityMapping("TryBlock", js="FinallyClause"),

        # C++-specific entities
        "Namespace": EntityMapping("Module", cpp="Namespace"),
        "TranslationUnit": EntityMapping("Module", cpp="TranslationUnit"),
        "Struct": EntityMapping("Class", cpp="Struct"),
        "Destructor": EntityMapping("Method", cpp="Destructor"),
        "Operator": EntityMapping("Method", cpp="Operator"),
        "Enum": EntityMapping("CodeEntity", cpp="Enum"),
        "EnumClass": EntityMapping("CodeEntity", cpp="EnumClass"),
        "Enumerator": EntityMapping("CodeEntity", cpp="Enumerator"),
        "UsingDirective": EntityMapping("Import", cpp="UsingDirective"),
        "UsingDeclaration": EntityMapping("Import", cpp="UsingDeclaration"),
        "Inheritance": EntityMapping("CodeEntity", cpp="Inheritance"),
        "IntegerLiteral": EntityMapping("CodeEntity", cpp="IntegerLiteral"),
        "FloatingLiteral": EntityMapping("CodeEntity", cpp="FloatingLiteral"),
        "StringLiteral": EntityMapping("CodeEntity", cpp="StringLiteral"),
        "CharacterLiteral": EntityMapping("CodeEntity", cpp="CharacterLiteral"),
        "BreakStatement": EntityMapping("CodeEntity", cpp="BreakStatement"),
        "ContinueStatement": EntityMapping("CodeEntity", cpp="ContinueStatement"),
        "GotoStatement": EntityMapping("CodeEntity", cpp="GotoStatement"),

        # HTML-specific entities
        "Document": EntityMapping("Module", html="Document"),
        "Element": EntityMapping("CodeEntity", html="Element"),
        "Script": EntityMapping("CodeEntity", html="Script"),
        "ScriptReference": EntityMapping("Import", html="ScriptReference"),
        "StyleSheet": EntityMapping("CodeEntity", html="StyleSheet"),
        "Form": EntityMapping("CodeEntity", html="Form"),
        "FormInput": EntityMapping("CodeEntity", html="FormInput"),
        "Link": EntityMapping("CodeEntity", html="Link"),
        "EventHandler": EntityMapping("CodeEntity", html="EventHandler"),
        "Meta": EntityMapping("CodeEntity", html="Meta"),
        "Image": EntityMapping("CodeEntity", html="Image"),
        "Iframe": EntityMapping("CodeEntity", html="Iframe"),

        # HTML Framework directives
        "VueDirective": EntityMapping("CodeEntity", html="VueDirective"),
        "AngularDirective": EntityMapping("CodeEntity", html="AngularDirective"),
        "HtmxAttribute": EntityMapping("CodeEntity", html="HtmxAttribute"),
        "AlpineDirective": EntityMapping("CodeEntity", html="AlpineDirective"),
        "DataAttribute": EntityMapping("CodeEntity", html="DataAttribute"),
    }

    # Common relationships and attributes (CNL hyphenated format)
    RELATIONSHIPS = [
        "inherits-from", "calls", "calls-transitive", "imports", "imports-transitive",
        "is-defined-in", "has-method", "inherits-method", "has-parameter", "is-of-function",
        "has-field", "has-attribute", "is-in-module", "contains-class", "contains-function",
        "has-docstring", "is-undocumented", "is-static", "is-async", "is-private",
        "has-name", "is-at-line", "has-line-count", "has-decorator", "has-type-annotation",
        "has-access-modifier", "has-parameter-count", "has-method-count"
    ]

    @classmethod
    def get_language(cls, language: LanguageType = "oo") -> Language:
        """Convert language string to Language enum."""
        return Language.from_string(language)

    @classmethod
    def get_prefix(cls, language: LanguageType = "oo") -> str:
        """Get the ontology prefix for a language."""
        lang = cls.get_language(language)
        return cls.PREFIXES[lang]

    @classmethod
    def relation(cls, rel: str, language: LanguageType = "oo") -> str:
        """
        Build a relation/property string.

        Args:
            rel: Relationship name (e.g., "inherits-from", "has-method")
            language: Target language

        Returns:
            Fully qualified relation string (e.g., "py:inherits-from")
        """
        prefix = cls.get_prefix(language)
        return f"{prefix}:{rel}"

    @classmethod
    def query_relation(cls, subj: str, rel: str, obj: str, language: LanguageType = "oo") -> str:
        """
        Build a REQL relation clause.

        Args:
            subj: Subject variable (e.g., "?method")
            rel: Relationship name (e.g., "is-defined-in")
            obj: Object variable or value (e.g., "?class")
            language: Target language

        Returns:
            REQL clause (e.g., '?method py:is-defined-in ?class')
        """
        relation = cls.relation(rel, language)
        return f'{subj} {relation} {obj}'

    @classmethod
    def supported_languages(cls) -> List[str]:
        """Return list of supported language identifiers."""
        return ["oo", "python", "javascript", "html", "csharp", "cpp"]

    @classmethod
    def is_language_specific(cls, entity: str) -> bool:
        """Check if an entity has language-specific mappings."""
        if entity not in cls.ENTITY_MAPPINGS:
            return False
        mapping = cls.ENTITY_MAPPINGS[entity]
        return mapping.py is not None or mapping.js is not None or mapping.html is not None or mapping.cs is not None or mapping.cpp is not None


# Convenience alias for shorter access
lang = LanguageSupport


def build_reql_query(
    select_vars: List[str],
    where_clauses: List[str],
    language: LanguageType = "oo",
    filter_clause: Optional[str] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """
    Build a complete REQL query with language-aware prefixes.

    Args:
        select_vars: Variables to select (e.g., ["?class", "?name"])
        where_clauses: WHERE clause conditions
        language: Target language for ontology prefixes
        filter_clause: Optional FILTER clause
        order_by: Optional ORDER BY clause
        limit: Optional LIMIT value

    Returns:
        Complete REQL query string
    """
    query_parts = [
        f"SELECT {' '.join(select_vars)}",
        "WHERE {",
        "  " + " .\n  ".join(where_clauses)
    ]

    if filter_clause:
        query_parts.append(f"  {filter_clause}")

    query_parts.append("}")

    if order_by:
        query_parts.append(f"ORDER BY {order_by}")

    if limit:
        query_parts.append(f"LIMIT {limit}")

    return "\n".join(query_parts)
