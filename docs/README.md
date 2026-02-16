# Reter Code Documentation

Reter Code is an AI-powered MCP (Model Context Protocol) server for code reasoning.

## Documentation Index

### Getting Started
- [Tools Reference](./tools.md) - **Complete reference for all MCP tools**
- [Architecture Overview](./architecture.md) - System design, components, and patterns

### Core Tools (Design Docs Workflow)

| Tool | Description |
|------|-------------|
| [session](./tools/session.md) | Session lifecycle - **call `context` first!** |
| [thinking](./tools/thinking.md) | Record reasoning with design doc sections |
| [diagram](./tools/diagram.md) | Visualize design docs, Gantt, UML |

### Additional Tools
- [items](./tools/items.md) - Query and manage items
- [project](./tools/project.md) - Project analytics (also in session context)

#### Code Analysis
- [code_inspection](./tools/code-inspection.md) - Multi-language code analysis, 24 languages (26 actions)
- [recommender](./tools/recommender.md) - Refactoring recommendations (58 detectors)

#### RAG/Search
- [RAG Tools](./tools/rag-tools.md) - Semantic search and analysis
  - `semantic_search` - Vector similarity search
  - `rag_status` - Index status
  - `find_similar_clusters` - Code clustering
  - `find_duplicate_candidates` - Duplicate detection
  - `analyze_documentation_relevance` - Doc-code analysis

#### Knowledge Management
- [Knowledge Tools](./tools/knowledge-tools.md)
  - `add_knowledge` - Add knowledge to RETER
  - `add_external_directory` - Load external code
  - `reql` - Execute REQL queries
  - `natural_language_query` - Natural language to CADSL

#### Visualization
- [diagram](./tools/diagram.md) - UML and project diagrams

#### Instance Management
- [instance_manager](./tools/instance-manager.md) - Manage RETER instances

## Quick Start - Design Docs Workflow

```python
# 1. CRITICAL: Restore session context first!
session(action="context")

# 2. Document the problem/context
thinking(
    thought="We need to refactor the authentication module...",
    thought_number=1,
    total_thoughts=5,
    section="context"
)

# 3. Define goals
thinking(
    thought="Goals: 1) Simplify login flow 2) Add OAuth support",
    thought_number=2,
    total_thoughts=5,
    section="goals"
)

# 4. Analyze code structure
code_inspection(action="describe_class", target="AuthService")

# 5. Document design decision
thinking(
    thought="Will use NextAuth.js for OAuth integration because...",
    thought_number=3,
    total_thoughts=5,
    section="design",
    thought_type="decision"
)

# 6. Create tasks from design
thinking(
    thought="Creating implementation tasks",
    thought_number=4,
    total_thoughts=5,
    section="tasks",
    operations={
        "task": {"name": "Add NextAuth.js", "category": "feature", "priority": "high"},
        "milestone": {"name": "OAuth Ready", "date": "2025-01-15"}
    }
)

# 7. Visualize the design doc
diagram(diagram_type="design_doc", format="mermaid")
```

## Configuration

Create `reter_code.json` in your project root:

```json
{
  "project_root": "/path/to/project",
  "include_patterns": ["src/**/*.py"],
  "exclude_patterns": ["**/test_*.py", "**/__pycache__/**"],
  "rag_enabled": true,
  "rag_embedding_model": "sentence-transformers/all-mpnet-base-v2"
}
```

Or use environment variables:
- `RETER_PROJECT_ROOT` - Project directory
- `ANTHROPIC_API_KEY` - For LLM features
- `TRANSFORMERS_CACHE` - Embedding model cache

## Component Dependencies

| Tool Category | SQLite | RETER | RAG Code | RAG Docs |
|--------------|--------|-------|----------|----------|
| Session (thinking, session, items, project) | Required | - | - | - |
| Code Analysis (code_inspection, recommender) | - | Required | - | - |
| Semantic Search (semantic_search) | - | - | Required | Required |
| Duplication (find_similar_clusters) | - | - | Required | - |
| Documentation (analyze_documentation_relevance) | - | - | Required | Required |
| Knowledge (add_knowledge, reql, natural_language_query) | - | Required | - | - |
| Diagrams - Session | Required | - | - | - |
| Diagrams - UML | - | Required | - | - |

Use `init_status()` to check component readiness.
