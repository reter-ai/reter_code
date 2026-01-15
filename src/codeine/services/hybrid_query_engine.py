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


# ============================================================
# CADSL TOOL INDEX FOR CASE-BASED REASONING
# ============================================================

@dataclass
class CADSLToolMetadata:
    """Metadata extracted from a CADSL tool file."""
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
        "description": "Get the complete REQL (RETE Query Language) formal grammar in Lark format. Call this when you need to generate a REQL query and want to ensure correct syntax.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_cadsl_grammar",
        "description": "Get the complete CADSL (Code Analysis DSL) formal grammar in Lark format. Call this when you need to generate a CADSL query with pipelines, graph operations, or diagrams.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "list_examples",
        "description": "List available CADSL example files organized by category. Use this to find relevant examples before generating a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category: 'smells', 'diagrams', 'rag', 'testing', 'inspection', 'refactoring', 'patterns', 'exceptions', 'dependencies', 'inheritance'. Leave empty for all."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_example",
        "description": "Get the content of a specific CADSL example file. Use this to see how similar queries are structured.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The example name (e.g., 'god_class', 'call_graph', 'duplicate_code')"
                }
            },
            "required": ["name"]
        }
    }
]


def handle_get_reql_grammar() -> str:
    """Return the REQL grammar."""
    grammar = _load_resource("REQL_GRAMMAR.lark")
    return f"""# REQL Grammar (Lark format)

{grammar}

## Key Points:
- Use `type` predicate for entity types: `?x type oo:Class`
- Use `oo:` prefix ONLY for types (oo:Class, oo:Method, oo:Function)
- Predicates have NO prefix: `name`, `inFile`, `definedIn`, `calls`, `inheritsFrom`
- FILTER requires parentheses: `FILTER(?count > 5)`
- Patterns separated by dots: `?x type oo:Class . ?x name ?n`
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
            result += f"- {f}\n"
        result += "\n"

    result += "\nUse get_example(name) to view a specific example."
    return result


def handle_get_example(name: str) -> str:
    """Get content of a specific CADSL example."""
    # Search in all subdirectories
    for subdir in _CADSL_TOOLS_DIR.iterdir():
        if subdir.is_dir():
            cadsl_file = subdir / f"{name}.cadsl"
            if cadsl_file.exists():
                with open(cadsl_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"# Example: {name}.cadsl (from {subdir.name})\n\n{content}"

    return f"# Example '{name}' not found. Use list_examples() to see available examples."


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
    else:
        return f"Unknown tool: {tool_name}"


# ============================================================
# QUERY TYPE CLASSIFICATION
# ============================================================

class QueryType(Enum):
    """Types of queries the hybrid engine can handle."""
    REQL = "reql"           # Simple structural queries
    CADSL = "cadsl"         # Complex pipelines, graph algorithms
    RAG = "rag"             # Semantic/similarity search


@dataclass
class SimilarTool:
    """A similar CADSL tool found via case-based reasoning."""
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
    """Result of classifying a natural language query."""
    query_type: QueryType
    confidence: float
    reasoning: str
    suggested_cadsl_tool: Optional[str] = None
    similar_tools: List[SimilarTool] = field(default_factory=list)


def find_similar_cadsl_tools(question: str, max_results: int = 3) -> List[SimilarTool]:
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
    then passes them to the classifier as context to improve routing.

    Args:
        question: The natural language question
        ctx: MCP context for LLM sampling

    Returns:
        QueryClassification with type, reasoning, and similar tools
    """
    # Fast-path: check for keywords that require RAG
    keyword_classification = _check_keyword_rag_routing(question)
    if keyword_classification:
        return keyword_classification

    # Case-based reasoning: find similar CADSL tools FIRST
    # This helps the classifier make better routing decisions
    similar_tools = find_similar_cadsl_tools(question, max_results=3)
    if similar_tools:
        tool_names = [t.name for t in similar_tools]
        debug_log.debug(f"CASE-BASED REASONING: Found similar tools: {tool_names}")

    try:
        # Build prompt with similar tools context
        prompt = f"Classify this code analysis question:\n\n{question}"
        classification_context = build_classification_context(similar_tools)
        if classification_context:
            prompt = prompt + classification_context

        response = await ctx.sample(prompt, system_prompt=CLASSIFICATION_SYSTEM_PROMPT)
        response_text = response.text if hasattr(response, 'text') else str(response)

        debug_log.debug(f"CLASSIFICATION RESPONSE: {response_text}")

        # Parse JSON response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response_text)

        query_type = QueryType(result.get("type", "reql"))

        return QueryClassification(
            query_type=query_type,
            confidence=float(result.get("confidence", 0.8)),
            reasoning=result.get("reasoning", "LLM classification"),
            suggested_cadsl_tool=result.get("suggested_tool"),
            similar_tools=similar_tools,  # Always include similar tools
        )

    except Exception as e:
        debug_log.debug(f"LLM classification failed: {e}, falling back to REQL")
        return QueryClassification(
            query_type=QueryType.REQL,
            confidence=0.5,
            reasoning=f"Fallback to REQL (classification error: {e})",
            similar_tools=similar_tools,  # Still include similar tools
        )


# ============================================================
# TOOL-AUGMENTED QUERY GENERATION
# ============================================================

REQL_GENERATION_PROMPT = """You are a REQL query generator. Generate valid REQL queries for code analysis.

You have tools available to help you:
- get_reql_grammar: Get the formal REQL grammar
- list_examples: See available query examples
- get_example: View a specific example

CRITICAL SYNTAX RULES:
1. Type patterns MUST use `type` predicate: `?x type oo:Class` NOT `?x oo:Class`
2. Use `oo:` prefix ONLY for types after `type`: oo:Class, oo:Method, oo:Function, oo:Module
3. Predicates have NO prefix: `name`, `inFile`, `definedIn`, `calls`, `inheritsFrom`, `atLine`
4. FILTER needs parentheses: `FILTER(?count > 5)`
5. Separate patterns with dots: `?x type oo:Class . ?x name ?n`

NOT SUPPORTED - DO NOT USE:
- BIND(...AS ?var) - NOT available in REQL
- VALUES ?var { ... } - NOT available in REQL
- If you need labels, just omit them or use separate queries per type

BOOLEAN PREDICATES (for code quality queries):
Many boolean predicates exist for code analysis. Use them directly with `true`:
- Exception handling: `?handler isSilentSwallow true`, `?handler bodyIsEmpty true`
- Code patterns: `?literal isMagicNumber true`, `?method isAsync true`
- Example: Find silent exception handlers:
  SELECT ?handler ?file ?line WHERE {
      ?handler type oo:CatchClause .
      ?handler isSilentSwallow true .
      ?handler inFile ?file .
      ?handler atLine ?line
  }

AGGREGATION PATTERNS (for counting, finding "more than N", etc.):
- Use (COUNT(?var) AS ?alias) in SELECT clause
- Use GROUP BY for the variables you want to group by
- Use HAVING (?alias > N) to filter by count (NOT FILTER!)
- Example: "Find classes with more than 10 methods":
  SELECT ?class ?name (COUNT(?method) AS ?method_count) WHERE {
      ?class type oo:Class . ?class name ?name .
      ?method type oo:Method . ?method definedIn ?class
  }
  GROUP BY ?class ?name
  HAVING (?method_count > 10)
  ORDER BY DESC(?method_count)

COMMON MISTAKES:
  WRONG: `?x oo:Class .`           <- missing `type` predicate!
  RIGHT: `?x type oo:Class .`      <- always include `type`

  WRONG: FILTER(?count > 5) for aggregation  <- FILTER is for row-level filtering
  RIGHT: HAVING (?count > 5) for aggregation <- HAVING filters after GROUP BY

  WRONG: BIND("label" AS ?type)    <- BIND not supported!
  WRONG: VALUES ?type { "label" }  <- VALUES not supported!
  RIGHT: Just query without labels, or run separate queries per type

When ready, return ONLY the REQL query (no explanation).
"""

CADSL_GENERATION_PROMPT = """You are a CADSL query generator. Generate valid CADSL pipelines for complex code analysis.

You have tools available to help you:
- get_cadsl_grammar: Get the formal CADSL grammar
- get_reql_grammar: Get the REQL grammar (for embedded reql blocks)
- list_examples: See available CADSL tool examples by category
- get_example: View a specific example to understand patterns

CRITICAL RULES:
1. CADSL uses `query`, `detector`, or `diagram` as tool types
2. Inside reql {} blocks, use `type` predicate with oo: prefix: `?x type oo:Class`
3. Entity types use `oo:` prefix: `oo:Class`, `oo:Method`, `oo:Module`, `oo:Function`
4. Predicates have NO prefix: `name`, `inFile`, `definedIn`, `calls`, `imports`
5. Pipeline steps use `|` operator: `| filter { }`, `| emit { }`

COMMON MISTAKE - NEVER do this in REQL blocks:
  WRONG: `?m1 oo:Module .`           <- missing `type` predicate!
  RIGHT: `?m1 type oo:Module .`      <- always include `type`

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
    sections.append("For REQL queries, extract the reql {} block and simplify as needed.\n")

    return "\n".join(sections)


async def generate_query_with_tools(
    question: str,
    query_type: QueryType,
    ctx,
    max_iterations: int = 5,
    similar_tools: Optional[List[SimilarTool]] = None
) -> str:
    """
    Generate a query using tool-augmented LLM with case-based reasoning.

    The LLM can call tools to fetch grammars and examples before generating.
    If similar tools are provided from case-based reasoning, they are included
    as templates that the LLM can adapt.

    Args:
        question: Natural language question
        query_type: REQL or CADSL
        ctx: MCP context for sampling
        max_iterations: Maximum tool-use iterations
        similar_tools: Similar CADSL tools from case-based reasoning

    Returns:
        Generated query string
    """
    system_prompt = REQL_GENERATION_PROMPT if query_type == QueryType.REQL else CADSL_GENERATION_PROMPT

    # Build user message with similar tools if available
    user_content = f"Generate a query for: {question}"

    if similar_tools:
        similar_section = build_similar_tools_section(similar_tools)
        user_content = f"{user_content}\n{similar_section}"
        debug_log.debug(f"Including {len(similar_tools)} similar tools as templates")

    messages = [
        {"role": "user", "content": user_content}
    ]

    for iteration in range(max_iterations):
        debug_log.debug(f"Query generation iteration {iteration + 1}")

        # Call LLM with tools
        response = await ctx.sample(
            messages[-1]["content"] if iteration == 0 else None,
            system_prompt=system_prompt,
            tools=QUERY_TOOLS,
            messages=messages if iteration > 0 else None
        )

        response_text = response.text if hasattr(response, 'text') else str(response)
        debug_log.debug(f"LLM response: {response_text[:500]}...")

        # Check if response contains tool calls
        tool_calls = getattr(response, 'tool_calls', None) or []

        # Also check for tool_use in content blocks (Claude format)
        if hasattr(response, 'content'):
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'tool_use':
                    tool_calls.append({
                        'name': block.name,
                        'input': block.input,
                        'id': getattr(block, 'id', f'tool_{iteration}')
                    })

        if not tool_calls:
            # No tool calls - LLM returned the final query
            return extract_query_from_response(response_text, query_type)

        # Handle tool calls
        for tool_call in tool_calls:
            tool_name = tool_call.get('name') or getattr(tool_call, 'name', '')
            tool_input = tool_call.get('input') or getattr(tool_call, 'input', {})
            tool_id = tool_call.get('id') or getattr(tool_call, 'id', f'tool_{iteration}')

            debug_log.debug(f"Tool call: {tool_name}({tool_input})")

            # Execute tool
            tool_result = handle_tool_call(tool_name, tool_input)

            # Add to conversation
            messages.append({
                "role": "assistant",
                "content": response_text,
                "tool_calls": [{"id": tool_id, "name": tool_name, "input": tool_input}]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result
            })

    # Max iterations reached
    debug_log.warning("Max iterations reached in query generation")
    return extract_query_from_response(response_text, query_type)


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
    duplicate_keywords = ["duplicate", "duplicated", "copy", "copied", "clone", "cloned",
                          "similar code", "same code", "repeated code", "code repetition"]
    cluster_keywords = ["cluster", "group similar", "semantic groups", "code groups",
                        "similar patterns", "related code blocks"]

    if any(kw in question_lower for kw in duplicate_keywords):
        params["analysis_type"] = "duplicates"
        params["similarity_threshold"] = 0.85  # Default similarity threshold
        params["max_results"] = 50
        params["exclude_same_file"] = True
        params["exclude_same_class"] = True
    elif any(kw in question_lower for kw in cluster_keywords):
        params["analysis_type"] = "clusters"
        params["n_clusters"] = 50
        params["min_size"] = 2
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
    """Result of executing a hybrid query."""
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
