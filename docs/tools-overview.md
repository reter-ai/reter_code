# Reter Code MCP Tools Overview

Reter Code provides 19 MCP tools organized into functional categories. This document provides an overview of all available tools.

## Tool Summary

| Tool | Category | Description |
|------|----------|-------------|
| `thinking` | Session | Record reasoning steps with operations |
| `session` | Session | Session lifecycle management |
| `items` | Session | Query and manage items |
| `project` | Session | Project analytics |
| `diagram` | Visualization | Generate diagrams (UML, Gantt, etc.) |
| `code_inspection` | Analysis | Python/JS/C#/C++ code analysis (26 actions) |
| `recommender` | Analysis | Refactoring and test coverage recommendations |
| `semantic_search` | RAG | Semantic search over code and docs |
| `rag_status` | RAG | RAG index status and statistics |
| `rag_reindex` | RAG | Trigger RAG index rebuild |
| `init_status` | RAG | Initialization and sync status |
| `find_similar_clusters` | RAG | Find semantically similar code clusters |
| `find_duplicate_candidates` | RAG | Find potential duplicate code |
| `analyze_documentation_relevance` | RAG | Analyze doc-code relevance |
| `add_knowledge` | Knowledge | Add knowledge to RETER |
| `add_external_directory` | Knowledge | Load external code directories |
| `quick_query` | Query | Execute REQL queries |
| `natural_language_query` | Query | Natural language to REQL |
| `instance_manager` | Management | Manage RETER instances |
| `reter_info` | Info | Version and diagnostic info |
| `initialize_project` | Init | Re-initialize project (background task) |

## Quick Start

```python
# 1. Restore session context (CRITICAL - do this first!)
session(action="context")

# 2. Check if server is ready
init_status()

# 3. Start analyzing code
code_inspection(action="list_classes")

# 4. Search semantically
semantic_search("authentication handling")

# 5. Record your reasoning
thinking(
    thought="Analyzing authentication module structure",
    thought_number=1,
    total_thoughts=3
)
```

## Tool Categories

### Session Management Tools
- [thinking](./tools/thinking.md) - Main reasoning tool
- [session](./tools/session.md) - Session lifecycle
- [items](./tools/items.md) - Item management
- [project](./tools/project.md) - Project analytics

### Code Analysis Tools
- [code_inspection](./tools/code-inspection.md) - Code structure analysis
- [recommender](./tools/recommender.md) - Refactoring recommendations

### RAG/Search Tools
- [semantic_search](./tools/semantic-search.md) - Vector similarity search
- [rag_status](./tools/rag-tools.md#rag_status) - Index status
- [find_similar_clusters](./tools/rag-tools.md#find_similar_clusters) - Code clustering
- [find_duplicate_candidates](./tools/rag-tools.md#find_duplicate_candidates) - Duplicate detection

### Knowledge Management Tools
- [add_knowledge](./tools/knowledge-tools.md#add_knowledge) - Add knowledge
- [instance_manager](./tools/instance-manager.md) - Instance management

### Visualization Tools
- [diagram](./tools/diagram.md) - Diagram generation

## Component Readiness

Different tools require different components to be ready:

| Component | Required By |
|-----------|-------------|
| SQLite | `thinking`, `session`, `items`, `project`, session diagrams |
| RETER | `code_inspection`, `recommender`, `quick_query`, UML diagrams |
| RAG Code Index | `semantic_search` (code), `find_similar_clusters`, `find_duplicate_candidates` |
| RAG Docs Index | `semantic_search` (docs), `analyze_documentation_relevance` |

Use `init_status()` to check component readiness before calling tools.
