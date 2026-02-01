"""
Hybrid NL Query Engine - Routes queries to REQL, CADSL, or RAG.

This module implements a smart routing engine that:
1. Uses LLM to classify the NL query type
2. Routes to the appropriate execution path:
   - REQL: Simple structural queries
   - CADSL: Complex queries with graph algorithms, joins, visualizations
   - RAG: Semantic/similarity-based queries

The query-generating LLM has access to tools for fetching grammars and examples.
"""

import re
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from ..logging_config import nlq_debug_logger as debug_log
from .agent_sdk_client import (
    is_agent_sdk_available,
    generate_reql_query,
    generate_cadsl_query,
    classify_query as classify_query_sdk,
)


# ============================================================
# CADSL TOOL INDEX FOR CASE-BASED REASONING
# ============================================================

@dataclass
class CADSLToolMetadata:
    """
    Metadata extracted from a CADSL tool file.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    name: str
    file_path: Path
    category: str
    tool_type: str  # detector, query, diagram
    description: str
    docstring: str
    severity: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    embedding_text: str = ""  # Text used for semantic embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "tool_type": self.tool_type,
            "description": self.description,
            "severity": self.severity,
            "keywords": self.keywords,
        }


class CADSLToolIndex:
    """
    Index of CADSL tools for case-based reasoning.

    Uses sentence-transformers embeddings for semantic similarity matching.
    Scans CADSL tool files and creates embeddings for efficient search.

    ::: This is-in-layer Service-Layer.
    ::: This is a query-engine.
    ::: This depends-on `reter_code.services.EmbeddingService`.
    ::: This is-in-process Main-Process.
    ::: This is stateful.
    """

    def __init__(self, tools_dir: Optional[Path] = None):
        self.tools_dir = tools_dir or _CADSL_TOOLS_DIR
        self._tools: Dict[str, CADSLToolMetadata] = {}
        self._tool_names: List[str] = []  # Ordered list for embedding index mapping
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_service = None
        self._indexed = False

    def _get_embedding_service(self):
        """Lazy-load the embedding service."""
        if self._embedding_service is None:
            from .embedding_service import get_embedding_service
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def index(self) -> None:
        """Scan and index all CADSL tool files with semantic embeddings."""
        if self._indexed:
            return

        if not self.tools_dir.exists():
            debug_log.warning(f"CADSL tools directory not found: {self.tools_dir}")
            return

        # First pass: extract metadata
        for cadsl_file in self.tools_dir.rglob("*.cadsl"):
            try:
                metadata = self._extract_metadata(cadsl_file)
                if metadata:
                    self._tools[metadata.name] = metadata
                    self._tool_names.append(metadata.name)
            except Exception as e:
                debug_log.debug(f"Failed to index {cadsl_file}: {e}")

        if not self._tools:
            self._indexed = True
            return

        # Second pass: generate embeddings
        try:
            embedding_service = self._get_embedding_service()
            texts = [self._tools[name].embedding_text for name in self._tool_names]
            self._embeddings = embedding_service.generate_embeddings_batch(texts)
            debug_log.debug(f"Generated embeddings for {len(self._tools)} CADSL tools")
        except Exception as e:
            debug_log.warning(f"Failed to generate embeddings, falling back to keyword matching: {e}")
            self._embeddings = None

        self._indexed = True
        debug_log.debug(f"Indexed {len(self._tools)} CADSL tools with semantic embeddings")

    def _extract_metadata(self, file_path: Path) -> Optional[CADSLToolMetadata]:
        """Extract metadata from a CADSL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract header comments (description)
        header_lines = []
        for line in content.split('\n'):
            if line.startswith('#'):
                header_lines.append(line.lstrip('#').strip())
            elif line.strip() and not line.startswith('#'):
                break
        description = ' '.join(header_lines)

        # Extract tool definition: detector/query/diagram name(...)
        tool_match = re.search(
            r'(detector|query|diagram)\s+(\w+)\s*\(([^)]*)\)',
            content
        )
        if not tool_match:
            return None

        tool_type = tool_match.group(1)
        tool_name = tool_match.group(2)
        params_str = tool_match.group(3)

        # Extract category and severity from params
        category = file_path.parent.name  # Default to directory name
        severity = None

        cat_match = re.search(r'category\s*=\s*"([^"]+)"', params_str)
        if cat_match:
            category = cat_match.group(1)

        sev_match = re.search(r'severity\s*=\s*"([^"]+)"', params_str)
        if sev_match:
            severity = sev_match.group(1)

        # Extract docstring
        docstring_match = re.search(r'"""([^"]+)"""', content)
        docstring = docstring_match.group(1).strip() if docstring_match else ""

        # Build rich embedding text for semantic search
        # Include name (with underscores replaced), description, docstring, and category
        name_readable = tool_name.replace('_', ' ')
        embedding_text = f"{name_readable}. {description} {docstring} Category: {category}"

        # Generate keywords for fallback matching
        text = f"{tool_name} {description} {docstring}".lower()
        keywords = self._extract_keywords(text)

        return CADSLToolMetadata(
            name=tool_name,
            file_path=file_path,
            category=category,
            tool_type=tool_type,
            description=description,
            docstring=docstring,
            severity=severity,
            keywords=keywords,
            embedding_text=embedding_text,
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text (used as fallback)."""
        keywords = set()
        keyword_patterns = [
            r'\b(god\s*class|large\s*class)\b', r'\b(long\s*method)\b',
            r'\b(dead\s*code|unused)\b', r'\b(duplicate|clone)\b',
            r'\b(magic\s*number)\b', r'\b(exception|error)\b',
            r'\b(silent|swallow)\b', r'\b(import|circular)\b',
            r'\b(test|untested)\b', r'\b(singleton|factory)\b',
            r'\b(class|method|function)\b', r'\b(diagram|graph)\b',
        ]
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keywords.add(match.group(1).lower().replace(' ', '_'))
        return list(keywords)

    def find_similar_tools(
        self,
        question: str,
        max_results: int = 3,
        min_score: float = 0.3
    ) -> List[Tuple[CADSLToolMetadata, float]]:
        """
        Find CADSL tools similar to the given question using semantic embeddings.

        Uses sentence-transformers for semantic similarity matching.
        Falls back to keyword matching if embeddings are unavailable.

        Args:
            question: Natural language question
            max_results: Maximum number of similar tools to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of (metadata, score) tuples sorted by score descending
        """
        self.index()

        if not self._tools:
            return []

        # Use semantic similarity if embeddings are available
        if self._embeddings is not None:
            return self._find_similar_semantic(question, max_results, min_score)
        else:
            return self._find_similar_keyword(question, max_results, min_score)

    def _find_similar_semantic(
        self,
        question: str,
        max_results: int,
        min_score: float
    ) -> List[Tuple[CADSLToolMetadata, float]]:
        """Find similar tools using semantic embeddings."""
        try:
            embedding_service = self._get_embedding_service()

            # Generate embedding for query
            query_embedding = embedding_service.generate_embedding(question)

            # Compute similarities
            similarities = embedding_service.compute_similarities_batch(
                query_embedding, self._embeddings
            )

            # Get top results above threshold
            results = []
            for idx, score in enumerate(similarities):
                if score >= min_score:
                    tool_name = self._tool_names[idx]
                    results.append((self._tools[tool_name], float(score)))

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            debug_log.debug(
                f"Semantic search for '{question[:50]}...': "
                f"found {len(results)} matches, top={results[0][0].name if results else 'none'}"
            )

            return results[:max_results]

        except Exception as e:
            debug_log.warning(f"Semantic search failed, falling back to keywords: {e}")
            return self._find_similar_keyword(question, max_results, min_score)

    def _find_similar_keyword(
        self,
        question: str,
        max_results: int,
        min_score: float
    ) -> List[Tuple[CADSLToolMetadata, float]]:
        """Fallback: find similar tools using keyword matching."""
        question_lower = question.lower()
        question_keywords = set(self._extract_keywords(question_lower))
        question_words = set(re.findall(r'\b[a-z]{3,}\b', question_lower))

        scores = []
        for name, metadata in self._tools.items():
            score = 0.0

            # Keyword overlap
            tool_keywords = set(metadata.keywords)
            if tool_keywords:
                overlap = len(question_keywords & tool_keywords)
                score += (overlap / len(tool_keywords)) * 0.5

            # Name matching
            name_lower = name.lower().replace('_', ' ')
            if any(word in name_lower for word in question_words if len(word) > 3):
                score += 0.3
            if name_lower in question_lower:
                score += 0.5

            if score >= min_score:
                scores.append((metadata, min(score, 1.0)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:max_results]

    def get_tool_content(self, name: str) -> Optional[str]:
        """Get the full content of a CADSL tool file."""
        self.index()

        if name not in self._tools:
            return None

        metadata = self._tools[name]
        with open(metadata.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_all_tools(self) -> Dict[str, CADSLToolMetadata]:
        """Get all indexed tools."""
        self.index()
        return self._tools.copy()


# Global tool index singleton
_cadsl_tool_index: Optional[CADSLToolIndex] = None


def get_cadsl_tool_index() -> CADSLToolIndex:
    """Get the global CADSL tool index."""
    global _cadsl_tool_index
    if _cadsl_tool_index is None:
        _cadsl_tool_index = CADSLToolIndex()
    return _cadsl_tool_index


# ============================================================
# RESOURCE LOADING
# ============================================================

_RESOURCES_DIR = Path(__file__).parent.parent / "resources"
_CADSL_TOOLS_DIR = Path(__file__).parent.parent / "cadsl" / "tools"


def _load_resource(filename: str) -> str:
    """Load a resource file from the resources directory."""
    resource_path = _RESOURCES_DIR / filename
    if resource_path.exists():
        with open(resource_path, 'r', encoding='utf-8') as f:
            return f.read()
    return f"# Resource file not found: {filename}"


def _load_classification_prompt() -> str:
    """Load the classification prompt."""
    return _load_resource("HYBRID_CLASSIFICATION.prompt")


# Load classification prompt at module import
CLASSIFICATION_SYSTEM_PROMPT = _load_classification_prompt()


# ============================================================
# QUERY GENERATION TOOLS
# ============================================================

# Tool definitions for the query-generating LLM
QUERY_TOOLS = [
    {
        "name": "get_reql_grammar",
        "description": "Get the formal REQL grammar specification. Use for syntax reference when generating structural queries.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_cadsl_grammar",
        "description": "Get the formal CADSL grammar specification. Use for syntax reference when generating pipeline queries.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "list_examples",
        "description": "List all CADSL example files by category. Categories: smells, rag, diagrams, dependencies, inspection, testing, refactoring, patterns, exceptions, inheritance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (optional). Leave empty for all categories."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_example",
        "description": "IMPORTANT: Get full working CADSL code for a specific example. Use 'category/name' format (e.g., 'smells/god_class', 'rag/duplicate_code'). These are production-ready templates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Example name in 'category/name' format (e.g., 'smells/god_class', 'rag/duplicate_code')"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "search_examples",
        "description": "Search for CADSL examples similar to your question using semantic similarity. Returns ranked list with scores. Use get_example to fetch full code. Helpful if unsure about syntax.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you're looking for (e.g., 'find duplicate code', 'class diagram', 'long methods')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)"
                }
            },
            "required": ["query"]
        }
    }
]


def handle_get_reql_grammar() -> str:
    """Return the REQL grammar."""
    grammar = _load_resource("REQL_GRAMMAR.lark")
    return f"""# REQL Grammar (Lark format)

{grammar}

## Key Points:
- Use `type` predicate for entity types: `?x type class`
- Use `oo:` prefix ONLY for types (class, method, function)
- Predicates have NO prefix: `has-name`, `is-in-file`, `is-defined-in`, `calls`, `inherits-from`
- FILTER requires parentheses: `FILTER(?count > 5)`
- Patterns separated by dots: `?x type class . ?x has-name ?n`

## Type vs Concept:
- `?x type class` - Filter with subsumption (matches class, class, etc.)
- `?x type ?t` - Returns ALL types (asserted + inferred) - MULTIPLE rows per entity
- `?x concept ?t` - Returns ONLY asserted type - ONE row per entity
- Use `concept` when you need the concrete type string (e.g., "method")
- CRITICAL: When using `concept` with FILTER in UNION queries, include the variable in SELECT!
"""


def handle_get_cadsl_grammar() -> str:
    """Return the CADSL grammar."""
    grammar = _load_resource("CADSL_GRAMMAR.lark")
    return f"""# CADSL Grammar (Lark format)

{grammar}

## Key Points:
- Tool types: `query`, `detector`, `diagram`
- Sources: `reql {{ ... }}`, `rag {{ search, query: "..." }}`
- Pipeline steps: `| filter {{ }}`, `| select {{ }}`, `| map {{ }}`, `| emit {{ }}`
- Graph ops: `| graph_cycles {{ }}`, `| graph_traverse {{ }}`, `| render_mermaid {{ }}`
- REQL inside uses same syntax as standalone REQL (see get_reql_grammar)
"""


def handle_list_examples(category: Optional[str] = None) -> str:
    """List available CADSL examples by category."""
    if not _CADSL_TOOLS_DIR.exists():
        return "# No examples directory found"

    examples = {}
    for subdir in _CADSL_TOOLS_DIR.iterdir():
        if subdir.is_dir():
            cat_name = subdir.name
            if category and cat_name != category:
                continue
            files = list(subdir.glob("*.cadsl"))
            if files:
                examples[cat_name] = [f.stem for f in files]

    if not examples:
        return f"# No examples found" + (f" for category '{category}'" if category else "")

    result = "# Available CADSL Examples\n\n"
    for cat, files in sorted(examples.items()):
        result += f"## {cat}\n"
        for f in sorted(files):
            result += f"- {cat}/{f}\n"
        result += "\n"

    result += "\nUse get_example(name) with 'category/name' format (e.g., 'smells/god_class')."
    return result


def handle_get_example(name: str) -> str:
    """Get content of a specific CADSL example.

    Args:
        name: Example name in 'category/name' format (e.g., 'smells/god_class')
              or just 'name' for backwards compatibility (searches all categories)
    """
    # Check if category/name format is used
    if "/" in name:
        category, example_name = name.split("/", 1)
        cadsl_file = _CADSL_TOOLS_DIR / category / f"{example_name}.cadsl"
        if cadsl_file.exists():
            with open(cadsl_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"# Example: {category}/{example_name}.cadsl\n\n{content}"
        return f"# Example '{name}' not found. Check category and name are correct."

    # Fallback: search in all subdirectories (backwards compatibility)
    found_in = []
    for subdir in sorted(_CADSL_TOOLS_DIR.iterdir()):  # Sort for deterministic order
        if subdir.is_dir():
            cadsl_file = subdir / f"{name}.cadsl"
            if cadsl_file.exists():
                found_in.append(subdir.name)

    if not found_in:
        return f"# Example '{name}' not found. Use list_examples() to see available examples."

    if len(found_in) > 1:
        locations = ", ".join(f"'{cat}/{name}'" for cat in found_in)
        return f"# Ambiguous: '{name}' exists in multiple categories: {locations}\nPlease use 'category/name' format."

    # Exactly one match found
    category = found_in[0]
    cadsl_file = _CADSL_TOOLS_DIR / category / f"{name}.cadsl"
    with open(cadsl_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return f"# Example: {category}/{name}.cadsl\n\n{content}"


def handle_search_examples(query: str, max_results: int = 10, include_code_count: int = 3) -> str:
    """Search for similar CADSL examples using semantic similarity.

    Args:
        query: Natural language search query
        max_results: Maximum number of examples to return
        include_code_count: Number of top matches to include full CADSL code for

    Returns:
        Formatted string with examples (top N with full code, rest as metadata)
    """
    tool_index = get_cadsl_tool_index()

    if not tool_index._tools:
        return "# No examples indexed. Tool index not initialized."

    # Use existing similarity search
    similar = tool_index.find_similar_tools(query, max_results=max_results, min_score=0.1)

    if not similar:
        return f"# No examples found matching '{query}'. Try list_examples() to browse all."

    result = f"# Examples matching '{query}'\n\n"
    result += f"Found {len(similar)} similar examples (sorted by relevance).\n\n"

    # Include full CADSL code for top matches (case-based reasoning)
    top_matches = similar[:include_code_count]
    remaining_matches = similar[include_code_count:]

    if top_matches:
        result += "## Top Matches (with full code)\n\n"
        result += "Use these as templates - adapt the patterns to your specific query:\n\n"

        for tool_meta, score in top_matches:
            content = tool_index.get_tool_content(tool_meta.name)
            result += f"### {tool_meta.category}/{tool_meta.name} (score: {score:.2f})\n"
            if tool_meta.description:
                result += f"{tool_meta.description}\n"
            result += "```cadsl\n"
            result += content.strip() if content else "# Content not available"
            result += "\n```\n\n"

    # List remaining matches as metadata only
    if remaining_matches:
        result += "## Other Matches\n\n"

        # Group by category for cleaner display
        by_category: Dict[str, List[tuple]] = {}
        for tool, score in remaining_matches:
            cat = tool.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((tool.name, score, tool.description[:80] if tool.description else ""))

        for cat, tools in sorted(by_category.items(), key=lambda x: -max(t[1] for t in x[1])):
            result += f"**{cat}:**\n"
            for name, score, desc in sorted(tools, key=lambda x: -x[1]):
                result += f"- {cat}/{name} (score: {score:.2f})"
                if desc:
                    result += f" - {desc}..."
                result += "\n"
            result += "\n"

        result += "Use get_example('category/name') to view full code for these examples."

    return result


def handle_tool_call(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Handle a tool call and return the result."""
    if tool_name == "get_reql_grammar":
        return handle_get_reql_grammar()
    elif tool_name == "get_cadsl_grammar":
        return handle_get_cadsl_grammar()
    elif tool_name == "list_examples":
        return handle_list_examples(tool_input.get("category"))
    elif tool_name == "get_example":
        return handle_get_example(tool_input.get("name", ""))
    elif tool_name == "search_examples":
        return handle_search_examples(
            tool_input.get("query", ""),
            tool_input.get("max_results", 10)
        )
    else:
        return f"Unknown tool: {tool_name}"


# ============================================================
# QUERY TYPE CLASSIFICATION
# ============================================================

class QueryType(Enum):
    """
    Types of queries the hybrid engine can handle.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    REQL = "reql"           # Simple structural queries
    CADSL = "cadsl"         # Complex pipelines, graph algorithms
    RAG = "rag"             # Semantic/similarity search


@dataclass
class SimilarTool:
    """
    A similar CADSL tool found via case-based reasoning.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    name: str
    score: float
    category: str
    description: str
    content: str  # Full CADSL code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "category": self.category,
            "description": self.description,
        }


@dataclass
class QueryClassification:
    """
    Result of classifying a natural language query.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    query_type: QueryType
    confidence: float
    reasoning: str
    suggested_cadsl_tool: Optional[str] = None
    similar_tools: List[SimilarTool] = field(default_factory=list)


def find_similar_cadsl_tools(question: str, max_results: int = 5) -> List[SimilarTool]:
    """
    Find CADSL tools similar to the question using case-based reasoning.

    Args:
        question: Natural language question
        max_results: Maximum number of similar tools to return

    Returns:
        List of SimilarTool instances with content
    """
    tool_index = get_cadsl_tool_index()
    similar = tool_index.find_similar_tools(question, max_results=max_results)

    result = []
    for metadata, score in similar:
        content = tool_index.get_tool_content(metadata.name)
        if content:
            result.append(SimilarTool(
                name=metadata.name,
                score=score,
                category=metadata.category,
                description=metadata.description,
                content=content,
            ))

    return result


def build_classification_context(similar_tools: List[SimilarTool]) -> str:
    """
    Build context about similar tools to help the classifier.

    Args:
        similar_tools: List of similar CADSL tools found

    Returns:
        Context string to append to classification prompt
    """
    if not similar_tools:
        return ""

    lines = ["\n\n## Available Similar Tools (from case-based reasoning)"]
    lines.append("These existing CADSL tools match your query semantically:")

    for tool in similar_tools:
        tool_type = "REQL-based" if "reql" in tool.content.lower() else "CADSL pipeline"
        lines.append(f"- **{tool.name}** ({tool.category}, score: {tool.score:.2f}): {tool.description[:100]}...")
        lines.append(f"  Type: {tool_type}")

    lines.append("\nIf a similar tool exists, prefer REQL or CADSL over RAG for structural queries.")

    return "\n".join(lines)


def _check_keyword_rag_routing(question: str) -> Optional[QueryClassification]:
    """
    Fast-path keyword check for queries that MUST use RAG.

    Some query types (duplicate detection, similarity search) always require
    semantic embeddings and should never use REQL structural queries.

    Returns:
        QueryClassification if keyword match, None to proceed with LLM classification
    """
    question_lower = question.lower()

    # Keywords that REQUIRE RAG (semantic comparison needed)
    rag_keywords = [
        "duplicate", "duplicated", "duplication",
        "similar code", "similar methods", "similar functions", "similar classes",
        "clone", "cloned", "code clone",
        "copy", "copied",
        "redundant", "repeated code", "code repetition",
        "same code", "identical code",
        "find similar", "detect similar",
    ]

    for keyword in rag_keywords:
        if keyword in question_lower:
            debug_log.debug(f"KEYWORD RAG ROUTING: '{keyword}' detected, forcing RAG")
            return QueryClassification(
                query_type=QueryType.RAG,
                confidence=1.0,
                reasoning=f"Keyword '{keyword}' requires semantic similarity analysis (RAG)",
                similar_tools=[],
            )

    return None


async def classify_query_with_llm(question: str, ctx) -> QueryClassification:
    """
    Use LLM to classify a natural language query.

    First checks for keywords that MUST use RAG (duplicate/similar detection),
    then performs case-based reasoning to find similar CADSL tools,
    then uses Agent SDK (or sampling fallback) to classify.

    Args:
        question: The natural language question
        ctx: MCP context for LLM sampling (used as fallback)

    Returns:
        QueryClassification with type, reasoning, and similar tools
    """
    # Fast-path: check for keywords that require RAG
    keyword_classification = _check_keyword_rag_routing(question)
    if keyword_classification:
        return keyword_classification

    # Case-based reasoning: find similar CADSL tools FIRST
    similar_tools = find_similar_cadsl_tools(question, max_results=5)
    if similar_tools:
        tool_names = [t.name for t in similar_tools]
        debug_log.debug(f"CASE-BASED REASONING: Found similar tools: {tool_names}")

    # Use Agent SDK for classification
    if not is_agent_sdk_available():
        debug_log.debug("Agent SDK not available, defaulting to REQL")
        return QueryClassification(
            query_type=QueryType.REQL,
            confidence=0.5,
            reasoning="Agent SDK not available, defaulting to REQL",
            similar_tools=similar_tools,
        )

    try:
        debug_log.debug("Using Agent SDK for classification")
        result = await classify_query_sdk(question)
        query_type = QueryType(result.get("type", "reql"))
        return QueryClassification(
            query_type=query_type,
            confidence=float(result.get("confidence", 0.8)),
            reasoning=result.get("reasoning", "Agent SDK classification"),
            suggested_cadsl_tool=result.get("suggested_tool"),
            similar_tools=similar_tools,
        )
    except Exception as e:
        debug_log.debug(f"Agent SDK classification failed: {e}, defaulting to REQL")
        return QueryClassification(
            query_type=QueryType.REQL,
            confidence=0.5,
            reasoning=f"Fallback to REQL (classification error: {e})",
            similar_tools=similar_tools,
        )


# ============================================================
# TOOL-AUGMENTED QUERY GENERATION
# ============================================================

REQL_GENERATION_PROMPT = """You are a REQL query generator. Generate valid REQL queries for code analysis.

You have tools available to help you:
- get_reql_grammar: Get the formal REQL grammar
- list_examples: See available query examples
- search_examples: Search for similar examples by description
- get_example: View a specific example

## CRITICAL: UNDERSTAND THE INTENT

Before generating a query, understand what the user is ACTUALLY asking for:
- "entry points" = functions named main, run, start, serve, app, execute, handle, etc.
- "server classes" = classes named Server, Service, Handler, Controller, API, App, etc.
- "utility classes" = classes named Utils, Helper, Common, Shared, etc.
- "test classes" = classes named Test*, *Test, *Spec, etc.
- "data classes" = classes named *Model, *Entity, *DTO, *Schema, etc.
- "configuration" = classes/functions with Config, Settings, Options in name

Use REGEX with case-insensitive flag "i" to match semantic concepts:
  FILTER(REGEX(?name, "main|run|start|serve|execute", "i"))

## ENTITY TYPES (oo: prefix for cross-language)
- class, method, function, module
- constructor, field, parameter
- import, catch-clause, attribute

## COMMON PREDICATES (NO prefix, CNL hyphenated format)
- `has-name` - Entity name
- `is-in-file` - Source file path
- `is-at-line` - Line number
- `is-defined-in` - Parent class/module
- `inherits-from` - Class inheritance
- `calls` - Function calls another
- `has-docstring` - Documentation string

## SYNTAX RULES
1. Type patterns: `?x type class` (NOT `?x class`)
2. Predicates have NO prefix: `has-name`, `is-in-file`, `is-defined-in`
3. FILTER needs parentheses: `FILTER(?count > 5)`
4. Separate patterns with dots: `?x type class . ?x has-name ?n`

## SEMANTIC QUERY PATTERNS

Find entry points and main functions:
```
SELECT ?func ?name ?file ?line WHERE {
    ?func type function .
    ?func has-name ?name .
    ?func is-in-file ?file .
    ?func is-at-line ?line .
    FILTER(REGEX(?name, "main|run|start|entry|serve|execute|handle|app", "i"))
}
```

Find server/service classes:
```
SELECT ?class ?name ?file WHERE {
    ?class type class .
    ?class has-name ?name .
    ?class is-in-file ?file .
    FILTER(REGEX(?name, "server|service|handler|controller|api|app|endpoint|router", "i"))
}
```

Find test classes:
```
SELECT ?class ?name ?file WHERE {
    ?class type class .
    ?class has-name ?name .
    ?class is-in-file ?file .
    FILTER(REGEX(?name, "^Test|Test$|Spec$|_test", "i"))
}
```

## AGGREGATION PATTERNS
- Use (COUNT(?var) AS ?alias) in SELECT clause
- Use GROUP BY + HAVING for filtering aggregates (NOT FILTER!)

Example - Find large classes:
```
SELECT ?class ?name (COUNT(?method) AS ?method_count) WHERE {
    ?class type class . ?class has-name ?name .
    ?method type method . ?method is-defined-in ?class
}
GROUP BY ?class ?name
HAVING (?method_count > 10)
ORDER BY DESC(?method_count)
```

## UNION RULES (CRITICAL)
- UNION must be INSIDE WHERE clause
- ALL UNION arms MUST bind the SAME variables!
  WRONG: `{ ?x has-name ?n } UNION { ?x parent ?p }` <- different vars!
  RIGHT: `{ ?x has-name ?n } UNION { ?x has-name ?n }` <- same vars

## NOT SUPPORTED
- BIND(...AS ?var) - NOT available
- VALUES ?var { ... } - NOT available

When ready, return ONLY the REQL query (no explanation).
"""

CADSL_GENERATION_PROMPT = """You are a CADSL query generator. Generate valid CADSL pipelines for complex code analysis.

You have tools available to help you:
- get_cadsl_grammar: Get the formal CADSL grammar
- get_reql_grammar: Get the REQL grammar (for embedded reql blocks)
- list_examples: See available CADSL tool examples by category
- search_examples: Search for similar examples by description
- get_example: View a specific example to understand patterns

CRITICAL RULES:
1. CADSL uses `query`, `detector`, or `diagram` as tool types
2. Inside reql {} blocks, use `type` predicate with oo: prefix: `?x type class`
3. Entity types use `oo:` prefix: `class`, `method`, `module`, `function`
4. Predicates have NO prefix (CNL hyphenated): `has-name`, `is-in-file`, `is-defined-in`, `calls`, `imports`
5. Pipeline steps use `|` operator: `| filter { }`, `| emit { }`

COMMON MISTAKE - NEVER do this in REQL blocks:
  WRONG: `?m1 module .`           <- missing `type` predicate!
  RIGHT: `?m1 type module .`      <- always include `type`

TYPE vs CONCEPT:
- `?x type method` - Filter with subsumption (matches method, etc.)
- `?x type ?t` - Returns ALL types (asserted + inferred) - MULTIPLE rows per entity
- `?x concept ?t` - Returns ONLY asserted type - ONE row per entity (e.g., "method")
- CRITICAL: When using `concept` with FILTER in UNION queries, include the variable in SELECT!

FILE_SCAN SOURCE (grep-like file search):
Use `file_scan` for file-level analysis, text search, and stats from loaded sources.

Basic syntax:
```
file_scan { glob: "*.py" }
```

Full parameters:
- `glob: "pattern"` or `glob: {param}` - file pattern (e.g., "*.py", "**/*.ts")
- `exclude: ["pattern1", "pattern2"]` - patterns to exclude
- `contains: "regex"` - grep-like search for text/regex in files
- `not_contains: "regex"` - exclude files containing text
- `case_sensitive: true/false` - case sensitivity for search (default: false)
- `include_matches: true` - include matching lines in output
- `context_lines: N` - lines of context around matches
- `max_matches_per_file: N` - limit matches per file
- `include_stats: true` - include line_count, file_size, last_modified

Returns:
- `file` - normalized file path (always)
- `line_count`, `file_size`, `last_modified` - when include_stats: true
- `match_count` - when contains is used
- `matches` - array of {line_number, content, context_before, context_after} when include_matches: true

Example - find TODOs with context:
```
file_scan { glob: "*.py", contains: "TODO|FIXME", include_matches: true, context_lines: 2 }
| filter { match_count > 0 }
| emit { file, matches }
```

Example - large files with stats:
```
file_scan { glob: "*.py", include_stats: true }
| filter { line_count > 500 }
| join { on: file, right: reql { SELECT ?file (COUNT(?c) AS ?class_count) WHERE { ?c type class . ?c is-in-file ?file } GROUP BY ?file } }
| emit { file, line_count, class_count }
```

RECOMMENDED: Call get_example() to see the EXACT syntax used in working examples.

When ready, return ONLY the CADSL query (no explanation).
"""


def build_similar_tools_section(similar_tools: List[SimilarTool]) -> str:
    """
    Build a prompt section with similar tools from case-based reasoning.

    Args:
        similar_tools: List of similar CADSL tools found

    Returns:
        Formatted string to include in the prompt
    """
    if not similar_tools:
        return ""

    sections = []
    sections.append("\n## SIMILAR EXISTING TOOLS (Case-Based Reasoning)")
    sections.append("I found these existing tools that are similar to your question.")
    sections.append("You can use them as templates and modify as needed:\n")

    for i, tool in enumerate(similar_tools, 1):
        sections.append(f"### Similar Tool {i}: {tool.name} (score: {tool.score:.2f})")
        sections.append(f"Category: {tool.category}")
        sections.append(f"Description: {tool.description}")
        sections.append("```cadsl")
        sections.append(tool.content.strip())
        sections.append("```\n")

    sections.append("**TIP**: If one of these tools closely matches what you need,")
    sections.append("you can adapt it with minor modifications instead of starting from scratch.")
    sections.append("For REQL queries, extract the reql {{ ... }} block and simplify as needed.\n")

    return "\n".join(sections)


async def generate_query_with_tools(
    question: str,
    query_type: QueryType,
    ctx,  # Kept for backwards compatibility but not used
    max_iterations: int = 5,
    similar_tools: Optional[List[SimilarTool]] = None,
    reter_instance=None,
    schema_info: str = "",
    rag_manager=None,
    project_root: Optional[str] = None
) -> str:
    """
    Generate a query using Agent SDK with case-based reasoning.

    Args:
        question: Natural language question
        query_type: REQL or CADSL
        ctx: Unused (kept for backwards compatibility)
        max_iterations: Maximum iterations for Agent SDK
        similar_tools: Similar CADSL tools from case-based reasoning
        reter_instance: Optional Reter instance for query validation
        schema_info: Schema information for REQL queries
        project_root: Project root path for file verification tools

    Returns:
        Generated query string
    """
    if not is_agent_sdk_available():
        raise RuntimeError("Claude Agent SDK not available. Install with: pip install claude-agent-sdk")

    # Build similar tools context
    similar_tools_context = None
    if similar_tools:
        similar_tools_context = build_similar_tools_section(similar_tools)
        debug_log.debug(f"Including {len(similar_tools)} similar tools as templates")

    debug_log.debug(f"Using Agent SDK for {query_type.value} generation")

    if query_type == QueryType.REQL:
        result = await generate_reql_query(
            question=question,
            schema_info=schema_info,
            reter_instance=reter_instance,
            max_iterations=max_iterations,
            similar_tools_context=similar_tools_context,
            rag_manager=rag_manager,
            project_root=project_root
        )
    else:  # CADSL
        result = await generate_cadsl_query(
            question=question,
            schema_info=schema_info,
            max_iterations=max_iterations,
            similar_tools_context=similar_tools_context,
            reter_instance=reter_instance,
            rag_manager=rag_manager,
            project_root=project_root
        )

    if result.success and result.query:
        debug_log.debug(f"Agent SDK generated query: {result.query[:200]}...")
        return result.query
    else:
        raise RuntimeError(f"Query generation failed: {result.error}")


def extract_query_from_response(response_text: str, query_type: QueryType) -> str:
    """Extract query from LLM response."""
    lang = "cadsl" if query_type == QueryType.CADSL else "reql|sparql|sql"

    # Try to find query in code blocks
    code_block_match = re.search(
        rf'```(?:{lang})?\s*\n?(.*?)\n?```',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    if code_block_match:
        query = code_block_match.group(1).strip()
    else:
        # Assume entire response is the query
        query = response_text.strip()

    # Remove any markdown artifacts
    query = re.sub(r'^```\w*\s*', '', query)
    query = re.sub(r'\s*```$', '', query)

    return query


# ============================================================
# LEGACY HELPERS (kept for compatibility)
# ============================================================

def extract_cadsl_from_response(response_text: str) -> str:
    """Extract CADSL query from LLM response."""
    return extract_query_from_response(response_text, QueryType.CADSL)


def build_cadsl_prompt(
    question: str,
    schema_info: str,
    attempt: int = 1,
    generated_query: Optional[str] = None,
    last_error: Optional[str] = None
) -> str:
    """Build the prompt for CADSL generation (legacy)."""
    if attempt == 1:
        return f"{schema_info}\nQuestion: {question}\n\nGenerate a CADSL query:"
    else:
        return f"""Question: {question}

Previous CADSL query attempt:
```cadsl
{generated_query}
```

Error received:
{last_error}

Please fix the CADSL query to correct the error. Return ONLY the corrected query:"""


# ============================================================
# RAG QUERY BUILDER
# ============================================================

def build_rag_query_params(question: str) -> Dict[str, Any]:
    """Extract RAG query parameters from natural language question.

    Detects three RAG modes:
    - "search": Semantic similarity search (default)
    - "duplicates": Find duplicate/similar code pairs
    - "clusters": Find clusters of semantically similar code
    """
    params = {
        "top_k": 20,
        "search_scope": "code",
        "include_content": False,
        "analysis_type": "search",  # search, duplicates, or clusters
    }

    question_lower = question.lower()
    entity_types = []

    # Detect duplicate/cluster analysis requests
    # Check cluster keywords FIRST (more specific) before duplicates
    cluster_keywords = ["cluster", "clusters", "group similar", "semantic groups", "code groups",
                        "similar patterns", "related code blocks"]
    duplicate_keywords = ["duplicate", "duplicated", "copy", "copied", "clone", "cloned",
                          "same code", "repeated code", "code repetition"]

    if any(kw in question_lower for kw in cluster_keywords):
        params["analysis_type"] = "clusters"
        params["n_clusters"] = 50
        params["min_size"] = 2
        params["exclude_same_file"] = True
        params["exclude_same_class"] = True
    elif any(kw in question_lower for kw in duplicate_keywords):
        params["analysis_type"] = "duplicates"
        params["similarity_threshold"] = 0.85  # Default similarity threshold
        params["max_results"] = 50
        params["exclude_same_file"] = True
        params["exclude_same_class"] = True

    # Entity type detection
    if "class" in question_lower and "method" not in question_lower:
        entity_types.append("class")
    if "method" in question_lower or "function" in question_lower:
        entity_types.extend(["method", "function"])
    if "document" in question_lower or "docs" in question_lower:
        params["search_scope"] = "docs"
        entity_types.append("document")

    if entity_types:
        params["entity_types"] = entity_types

    query = question
    for word in ["find", "search", "look for", "show", "get", "list"]:
        query = re.sub(rf'\b{word}\b', '', query, flags=re.IGNORECASE)
    query = query.strip()

    params["query"] = query if query else question

    return params


# ============================================================
# HYBRID QUERY EXECUTION
# ============================================================

@dataclass
class HybridQueryResult:
    """
    Result of executing a hybrid query.

    ::: This is-in-layer Utility-Layer.
    ::: This is a value-object.
    """
    success: bool
    results: List[Dict[str, Any]]
    count: int
    query_type: QueryType
    generated_query: Optional[str] = None
    rag_params: Optional[Dict[str, Any]] = None
    classification: Optional[QueryClassification] = None
    execution_time_ms: float = 0
    error: Optional[str] = None
    tools_used: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "results": self.results,
            "count": self.count,
            "query_type": self.query_type.value,
            "generated_query": self.generated_query,
            "rag_params": self.rag_params,
            "classification": {
                "type": self.classification.query_type.value,
                "confidence": self.classification.confidence,
                "reasoning": self.classification.reasoning,
            } if self.classification else None,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
            "tools_used": self.tools_used,
        }
