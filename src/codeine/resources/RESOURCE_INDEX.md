# RETER MCP Server - Resource Index

**Complete index of all resources and tools for the RETER MCP server.**

This document lists all MCP resources with their URIs and all MCP tools with their capabilities.

---

## 🛠️ MCP Tools - Design Docs Approach

### Core Tools (Design Docs Workflow)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `session` | Session lifecycle - **call context first!** | action: start, context, end, clear |
| `thinking` | **PRIMARY** - Reasoning with design doc sections | thought, section, operations |
| `diagram` | Visualize design docs, Gantt, UML | diagram_type: design_doc, gantt, class_hierarchy |

### Additional Session Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `items` | Query/manage items | action: list, get, delete, update |
| `project` | Project analytics | action: health, critical_path, overdue, impact |

### Knowledge Tools (4)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `add_knowledge` | Add facts/rules to RETER | source, type, source_id |
| `add_external_python_directory` | Load external Python files | directory, recursive |
| `quick_query` | Execute REQL queries | query, type |
| `natural_language_query` | **RECOMMENDED** - Plain English queries | question |

### Instance & Analysis Tools (3)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `instance_manager` | Manage instances/sources | action: list, list_sources, get_facts, forget, reload, check |
| `code_inspection` | Python code analysis (26 actions) | action, target, module |
| `recommender` | Code analysis recommenders | recommender_type: refactoring, test_coverage |

---

## 📚 MCP Resources

### Quick Start Resources

| URI | File | Description |
|-----|------|-------------|
| `guide://logical-thinking/usage` | `AI_AGENT_USAGE_GUIDE.md` | **READ FIRST** - Complete usage guide |
| `guide://reter/session-context` | `SESSION_CONTEXT_GUIDE.md` | **CRITICAL** - Session continuity |
| `reference://reter/syntax-quick` | `SYNTAX_QUICK_REFERENCE.md` | One-page syntax reference |

### Core Documentation (8 resources)

| # | URI | File | Description |
|---|-----|------|-------------|
| 1 | `guide://logical-thinking/usage` | `AI_AGENT_USAGE_GUIDE.md` | Comprehensive usage patterns |
| 2 | `grammar://reter/dl-reql` | `GRAMMAR_REFERENCE.md` | DL and REQL syntax reference |
| 3 | `guide://reter/custom-knowledge` | `CUSTOM_KNOWLEDGE_GUIDE.md` | Building ontologies |
| 4 | `reference://reter/syntax-quick` | `SYNTAX_QUICK_REFERENCE.md` | Quick syntax reference |
| 5 | `guide://logical-thinking/refactoring` | `REFACTORING_SUMMARY.md` | API migration guide |
| 6 | `guide://plugins/system` | `PLUGIN_SYSTEM_GUIDE.md` | Plugin architecture |
| 7 | `guide://reter/session-context` | `SESSION_CONTEXT_GUIDE.md` | Session continuity |
| 8 | `guide://reter/recommendations` | `RECOMMENDATIONS_PLUGIN_GUIDE.md` | Recommendations tracking |

### Meta-Ontology (1 resource)

| # | URI | File | Description |
|---|-----|------|-------------|
| 9 | `ontology://reter/oo` | `oo_ontology.reol` | Language-independent OO concepts |

### Python Analysis (3 resources)

| # | URI | File | Description |
|---|-----|------|-------------|
| 10 | `python://reter/tools` | `PYTHON_TOOLS_REFERENCE.md` | Python analysis tools API |
| 11 | `python://reter/query-patterns` | `PYTHON_QUERY_PATTERNS.md` | Common query patterns |
| 12 | `python://reter/analysis` | `python/PYTHON_ANALYSIS_REFERENCE.md` | Facts extracted from Python |

### JavaScript Analysis (1 resource)

| # | URI | File | Description |
|---|-----|------|-------------|
| 13 | `javascript://reter/ontology` | `javascript/js_ontology.reol` | JavaScript semantic ontology |

### System Documentation (4 resources)

| # | URI | File | Description |
|---|-----|------|-------------|
| 13 | `system://reter/snapshots` | `system/AUTOMATIC_SNAPSHOTS.md` | State persistence |
| 14 | `system://reter/multiple-instances` | `system/MULTIPLE_INSTANCES.md` | Multi-tenant support |
| 15 | `system://reter/source-management` | `system/SOURCE_MANAGEMENT.md` | Source tracking |
| 16 | `system://reter/thread-safety` | `system/THREAD_SAFETY.md` | Concurrency patterns |

### Refactoring Recipes (14 resources)

| # | URI | File | Description |
|---|-----|------|-------------|
| 16 | `recipe://refactoring/index` | `rtp/REFACTORING_RECIPES_INDEX.md` | Recipe catalog |
| 17 | `recipe://refactoring/access-guide` | `rtp/REFACTORING_RECIPES_ACCESS.md` | Access guide |
| 18-29 | `recipe://refactoring/chapter-XX` | `rtp/recipe_chapter_*.md` | Chapters 1-12 |

---

## 🔧 Tool Details

### `thinking` - Primary Reasoning Tool (Design Docs)

**Always use this tool with design doc sections!**

```
See: guide://reter/session-context
```

**Design Doc Sections:** `context`, `goals`, `non_goals`, `design`, `alternatives`, `risks`, `implementation`, `tasks`

**Task Categories:** `feature`, `bug`, `refactor`, `test`, `docs`, `research`

**Operations:**
- Create items: `task` (with category), `milestone`
- Create relations: `traces`, `implements`, `depends_on`, `affects`
- Update items: `update_task`, `complete_task`
- RETER: `assert`, `query`

### `session` - Session Lifecycle

**CRITICAL: Call `session(action="context")` at:**
1. Start of every new session
2. After context compactification
3. When resuming work

**Actions:**
- `start` - Begin new session with goal
- `context` - **Restore full context**
- `end` - Archive session
- `clear` - Reset session

### `code_inspection` - Python Analysis (26 actions)

```
See: python://reter/tools
```

**Structure/Navigation:**
- `list_modules`, `list_classes`, `list_functions`
- `describe_class`, `get_docstring`, `get_method_signature`
- `get_class_hierarchy`, `get_package_structure`

**Search/Find:**
- `find_usages`, `find_subclasses`, `find_callers`, `find_callees`
- `find_decorators`, `find_tests`

**Analysis:**
- `analyze_dependencies`, `get_imports`, `get_external_deps`
- `predict_impact`, `get_complexity`, `get_magic_methods`
- `get_interfaces`, `get_public_api`, `get_type_hints`
- `get_api_docs`, `get_exceptions`, `get_architecture`

### `recommender` - Code Analysis

```
See: recipe://refactoring/index
```

**Types:**
- `refactoring` - Detect code smells and patterns
- `test_coverage` - Find untested code

**Usage:**
```python
recommender("refactoring")  # List detectors
recommender("refactoring", "god_class")  # Run detector
recommender("test_coverage", "untested_classes")
```

### `diagram` - Visualizations

**Design Doc Diagrams:**
- `design_doc` - **Design doc structure with sections** (context, goals, design, etc.)
- `gantt` - Project schedule with tasks and milestones
- `thought_chain` - Reasoning chain with branches
- `traceability` - Task traceability matrix

**UML/Code Diagrams:**
- `class_hierarchy` - Inheritance tree
- `class_diagram` - Full class details
- `sequence` - Method call flows
- `dependencies` - Module graph
- `call_graph` - Function calls
- `coupling` - Coupling matrix

### `instance_manager` - Instance Management

```
See: system://reter/multiple-instances
```

**Actions:**
- `list` - List all instances
- `list_sources` - Sources in instance
- `get_facts` - Facts from source
- `forget` - Remove source
- `reload` - Reload modified sources
- `check` - Consistency check

### `natural_language_query` - Plain English Queries

**RECOMMENDED** for querying the knowledge base!

```python
natural_language_query("What classes inherit from BaseTool?")
natural_language_query("Find methods with more than 5 parameters")
```

---

## 📖 Recommended Reading Order

### For New Users
1. `guide://reter/session-context` - **START HERE**
2. `guide://logical-thinking/usage` - Complete usage guide
3. `reference://reter/syntax-quick` - Quick reference

### For Python Analysis
1. `python://reter/tools` - Tool reference
2. `python://reter/query-patterns` - Query examples
3. `python://reter/analysis` - Fact reference

### For Code Quality
1. `recipe://refactoring/index` - Refactoring catalog
2. Use `recommender("refactoring")` to list detectors
3. Use `recommender("test_coverage")` for test gaps

### For Project Management
1. Use `session(action="start")` with project dates
2. Use `thinking` with task operations
3. Use `diagram(diagram_type="gantt")` for visualization

---

## 📂 File Structure

```
resources/
├── AI_AGENT_USAGE_GUIDE.md          [guide://logical-thinking/usage]
├── SESSION_CONTEXT_GUIDE.md         [guide://reter/session-context]
├── CUSTOM_KNOWLEDGE_GUIDE.md        [guide://reter/custom-knowledge]
├── GRAMMAR_REFERENCE.md             [grammar://reter/dl-reql]
├── SYNTAX_QUICK_REFERENCE.md        [reference://reter/syntax-quick]
├── PLUGIN_SYSTEM_GUIDE.md           [guide://plugins/system]
├── RECOMMENDATIONS_PLUGIN_GUIDE.md  [guide://reter/recommendations]
├── PYTHON_TOOLS_REFERENCE.md        [python://reter/tools]
├── PYTHON_QUERY_PATTERNS.md         [python://reter/query-patterns]
├── oo_ontology.reol                 [ontology://reter/oo] ** META-ONTOLOGY **
├── RESOURCE_INDEX.md                [This file]
├── javascript/
│   └── js_ontology.reol             [javascript://reter/ontology]
├── python/
│   ├── py_ontology.reol             [python://reter/ontology]
│   └── PYTHON_ANALYSIS_REFERENCE.md [python://reter/analysis]
├── rtp/
│   ├── REFACTORING_RECIPES_INDEX.md [recipe://refactoring/index]
│   └── recipe_chapter_*.md          [recipe://refactoring/chapter-XX]
└── system/
    ├── AUTOMATIC_SNAPSHOTS.md       [system://reter/snapshots]
    ├── MULTIPLE_INSTANCES.md        [system://reter/multiple-instances]
    ├── SOURCE_MANAGEMENT.md         [system://reter/source-management]
    └── THREAD_SAFETY.md             [system://reter/thread-safety]
```

---

## ✅ Status

| Category | Count | Status |
|----------|-------|--------|
| MCP Tools | 12 | ✅ All registered |
| MCP Resources | 31 | ✅ All accessible |
| code_inspection actions | 26 | ✅ Consolidated |
| diagram types | 10 | ✅ Available |
| recommender types | 2 | ✅ refactoring, test_coverage |
| Ontologies | 3 | ✅ oo (meta), Python, JavaScript |
| Languages supported | 2 | ✅ Python, JavaScript |

**Last Updated**: 2025-12-05
**Version**: 4.0 (Consolidated tools)

---

## 🔄 Recent Updates

### 2025-12-05: JavaScript Support
- Added **JavaScript language support** with full fact extraction
- New `javascript/js_ontology.reol` - semantic ontology for JS code
- JavaScript entities: Class, Function, Method, ArrowFunction, etc.
- Error handling: TryBlock, CatchClause, ThrowStatement
- Control flow: ReturnStatement, Call, Assignment
- JSDoc comment extraction as docstrings

### 2025-11-29: Tool Consolidation
- **12 MCP tools** (down from 100+)
- `thinking` as primary reasoning tool with operations
- `session` for lifecycle management
- `code_inspection` consolidates 26 Python analysis actions
- `recommender` provides refactoring and test_coverage
- `diagram` handles all visualizations
- `instance_manager` for source/instance management
