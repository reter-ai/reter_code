# AI Agent Guide: Using RETER MCP Effectively

**Target Audience:** AI Agents (Claude Code, etc.) using RETER for semantic reasoning
**Goal:** Master the Design Docs approach for structured reasoning

---

## CRITICAL: First Steps

### 1. Call `session(action="context")` First!

```python
# ALWAYS start every session with this
session(action="context")
```

This restores: **design doc sections**, **tasks with categories**, **project health**, **milestones**, and **suggestions**.

Call this:
1. At the START of every new session
2. After every context compactification
3. When resuming work after any interruption

### 2. Use `thinking` with Design Doc Sections

```python
# Document the problem/context
thinking(
    thought="We need to add OAuth to the auth module...",
    thought_number=1,
    total_thoughts=5,
    section="context"
)

# Define goals
thinking(
    thought="Goals: 1) Add OAuth 2) Keep existing login",
    thought_number=2,
    total_thoughts=5,
    section="goals"
)

# Document design decision
thinking(
    thought="Using NextAuth.js for OAuth integration",
    thought_number=3,
    total_thoughts=5,
    section="design",
    thought_type="decision"
)
```

### Design Doc Sections

| Section | Purpose |
|---------|---------|
| `context` | Problem statement, background |
| `goals` | What we want to achieve |
| `non_goals` | Explicitly out of scope |
| `design` | Technical approach |
| `alternatives` | Options considered and rejected |
| `risks` | What could go wrong |
| `implementation` | Implementation details |
| `tasks` | Task breakdown |

---

## Core Mental Model

### RETER is Your Persistent Semantic Memory

Think of RETER as a **persistent semantic brain** where:
- **Facts accumulate** across reasoning steps (don't reset!)
- **Rules fire automatically** when patterns match
- **Inferences propagate** without explicit commands
- **Knowledge persists** until explicitly forgotten
- **Python code** is automatically analyzed from RETER_PROJECT_ROOT
- **Multiple instances** supported for isolation

**Mental Shift Required:**
```
WRONG: Each step is isolated
RIGHT: Each step builds on accumulated knowledge
```

---

## The 12 MCP Tools

### Core Tools

| Tool | Purpose | Key Actions |
|------|---------|-------------|
| `thinking` | **PRIMARY** - Reasoning with operations | Create items, relations, RETER operations |
| `session` | Session lifecycle | start, context, end, clear |
| `items` | Query/manage items | list, get, delete, update |
| `project` | Project analytics | health, critical_path, overdue, impact |
| `diagram` | Generate diagrams | gantt, class_hierarchy, sequence, etc. |

### Knowledge Tools

| Tool | Purpose |
|------|---------|
| `add_knowledge` | Add facts/rules/Python files to RETER |
| `add_external_python_directory` | Load external Python codebases |
| `quick_query` | Execute REQL queries (advanced) |
| `natural_language_query` | **RECOMMENDED** - Plain English queries |

### Analysis Tools

| Tool | Purpose |
|------|---------|
| `instance_manager` | Manage instances and sources |
| `code_inspection` | Python analysis (26 actions) |
| `recommender` | Code analysis: refactoring, test_coverage |

---

## Effective Usage Patterns

### Pattern 1: Use `thinking` for Reasoning

**Always use the `thinking` tool when:**
- Analyzing problems or code
- Making decisions or planning
- Breaking down complex tasks
- Recording requirements, tasks, or recommendations
- Creating traceability between items

```python
# Reasoning with operations
thinking(
    thought="The UserService class has too many responsibilities",
    thought_number=1,
    total_thoughts=3,
    thought_type="analysis",
    operations={
        "recommendation": {
            "text": "Split UserService using Single Responsibility Principle",
            "severity": "high",
            "category": "refactoring"
        },
        "affects": ["user_service.py"]
    }
)
```

### Pattern 2: Use `natural_language_query` for Questions

**RECOMMENDED** for querying the knowledge base:

```python
# Ask questions in plain English
natural_language_query("What classes inherit from BaseTool?")
natural_language_query("Find methods with more than 5 parameters")
natural_language_query("Which modules have the most dependencies?")
```

### Pattern 3: Use `code_inspection` for Code Analysis

```python
# List all classes
code_inspection(action="list_classes")

# Describe a specific class
code_inspection(action="describe_class", target="UserService")

# Find tests for a function
code_inspection(action="find_tests", target="process_payment")

# Get architecture overview
code_inspection(action="get_architecture")
```

### Pattern 4: Use `recommender` for Code Quality

```python
# List all refactoring detectors
recommender("refactoring")

# Run specific detector
recommender("refactoring", "god_class")
recommender("refactoring", "long_method")

# Test coverage analysis
recommender("test_coverage")
recommender("test_coverage", "untested_classes")
```

---

## The `thinking` Tool Operations

### Create Tasks with Categories
```python
operations={
    "task": {
        "name": "Implement OAuth",
        "category": "feature",  # feature, bug, refactor, test, docs, research
        "priority": "high",     # critical, high, medium, low
        "start_date": "2024-01-15",
        "duration_days": 5
    },
    "milestone": {"name": "OAuth Ready", "date": "2024-02-01"}
}
```

### Task Categories

| Category | Use For |
|----------|---------|
| `feature` | New functionality |
| `bug` | Bug fixes |
| `refactor` | Code refactoring |
| `test` | Test creation |
| `docs` | Documentation |
| `research` | Research/investigation |

### Create Relations
```python
operations={
    "traces": ["REQ-001", "REQ-002"],
    "satisfies": ["REQ-003"],
    "implements": ["TASK-001"],
    "depends_on": ["TASK-002"],
    "affects": ["module.py", "class_name"]
}
```

### Update Items
```python
operations={
    "update_item": {"item_id": "REQ-001", "status": "verified"},
    "update_task": {"task_id": "TASK-001", "status": "in_progress"},
    "complete_task": "TASK-001"
}
```

### RETER Operations
```python
operations={
    "assert": "Person(alice)\nage(alice, 25)",
    "query": "SELECT ?c WHERE { ?c a py:Class }",
    "python_file": "/path/to/file.py",
    "forget_source": "source_id_to_forget"
}
```

---

## Example Workflows

### Workflow 1: Code Analysis Investigation

```python
# 1. Restore context
session(action="context")

# 2. Start reasoning
thinking(
    thought="Beginning analysis of the authentication module",
    thought_number=1,
    total_thoughts=5,
    thought_type="planning"
)

# 3. Analyze code structure
code_inspection(action="describe_class", target="AuthService")

# 4. Query for patterns
natural_language_query("What methods in AuthService handle passwords?")

# 5. Run detectors
recommender("refactoring", "god_class")

# 6. Record findings
thinking(
    thought="AuthService shows god class pattern with 15+ methods",
    thought_number=2,
    total_thoughts=5,
    thought_type="analysis",
    operations={
        "recommendation": {
            "text": "Split AuthService into AuthenticationService and AuthorizationService",
            "severity": "high"
        },
        "task": {
            "name": "Refactor AuthService",
            "duration_days": 3
        }
    }
)

# 7. Visualize
diagram(diagram_type="class_diagram", classes=["AuthService"])
```

### Workflow 2: Test Coverage Analysis

```python
# 1. Restore context
session(action="context")

# 2. Check test coverage
recommender("test_coverage")

# 3. Find untested code
recommender("test_coverage", "untested_classes")

# 4. Find existing tests
code_inspection(action="find_tests", target="PaymentService")

# 5. Record findings
thinking(
    thought="PaymentService lacks test coverage for error handling",
    thought_number=1,
    total_thoughts=3,
    operations={
        "recommendation": {
            "text": "Add tests for PaymentService error paths",
            "severity": "high",
            "category": "testing"
        }
    }
)
```

### Workflow 3: Project Planning

```python
# 1. Start session with goal
session(action="start", goal="Implement user authentication", project_start="2024-01-15", project_end="2024-02-28")

# 2. Create requirements
thinking(
    thought="Defining authentication requirements",
    thought_number=1,
    total_thoughts=4,
    operations={
        "requirement": {"text": "Users shall authenticate with username/password", "priority": "critical"},
        "requirement": {"text": "System shall support OAuth2", "priority": "high"}
    }
)

# 3. Create tasks
thinking(
    thought="Breaking down implementation into tasks",
    thought_number=2,
    total_thoughts=4,
    operations={
        "task": {"name": "Design auth schema", "duration_days": 2},
        "task": {"name": "Implement login endpoint", "duration_days": 3},
        "milestone": {"name": "Auth MVP Complete", "end_date": "2024-02-01"}
    }
)

# 4. Visualize
diagram(diagram_type="gantt")
```

---

## Common Mistakes to Avoid

### Mistake 1: Not Calling `session(action="context")` First

**WRONG:**
```python
# Starting without context
thinking(thought="Let me analyze...")  # Lost previous work!
```

**RIGHT:**
```python
session(action="context")  # First!
thinking(thought="Let me analyze...")
```

### Mistake 2: Not Using `thinking` for Reasoning

**WRONG:**
```python
# Just running queries without tracking
code_inspection(action="list_classes")
recommender("refactoring", "god_class")
# No record of what was found!
```

**RIGHT:**
```python
# Use thinking to track reasoning
thinking(
    thought="Analyzing for god classes",
    thought_number=1,
    total_thoughts=2,
    thought_type="analysis"
)
result = recommender("refactoring", "god_class")
thinking(
    thought=f"Found {len(result['findings'])} god classes",
    thought_number=2,
    total_thoughts=2,
    operations={
        "recommendation": {"text": "Address god class in UserService"}
    }
)
```

### Mistake 3: Using `quick_query` Instead of `natural_language_query`

**WRONG:**
```python
# Complex REQL that may have syntax errors
quick_query(query="SELECT ?c WHERE { ?c a py:Class . ?c py:hasMethod ?m . ?m py:hasName '__init__' }")
```

**RIGHT:**
```python
# Just ask in English
natural_language_query("Find classes that have __init__ methods")
```

---

## Quick Reference Card

| You Want To... | Use This |
|----------------|----------|
| Start session | `session(action="context")` |
| Record reasoning | `thinking(thought=..., operations=...)` |
| Query code | `natural_language_query(question=...)` |
| Analyze classes | `code_inspection(action="describe_class", target=...)` |
| Find tests | `code_inspection(action="find_tests", target=...)` |
| Find code smells | `recommender("refactoring", detector_name)` |
| Check test coverage | `recommender("test_coverage")` |
| Create diagram | `diagram(diagram_type=..., ...)` |
| View project status | `project(action="health")` |
| List items | `items(action="list", item_type=...)` |
| End session | `session(action="end")` |

---

## Related Resources

| Resource | URI | Description |
|----------|-----|-------------|
| Session Guide | `guide://reter/session-context` | Session continuity |
| Python Tools | `python://reter/tools` | Code analysis reference |
| Query Patterns | `python://reter/query-patterns` | REQL examples |
| Refactoring | `recipe://refactoring/index` | Refactoring recipes |
| Syntax Quick | `reference://reter/syntax-quick` | Syntax reference |

---

## Best Practices Summary

1. **Always call `session(action="context")` first**
2. **Use `thinking` for all reasoning** - captures your thought chain
3. **Use `natural_language_query`** - easier than REQL
4. **Use `recommender`** for code quality analysis
5. **Use `code_inspection(action="find_tests")`** to find tests
6. **Create traceability** with operations in `thinking`
7. **Test every change** - use `recommender("test_coverage")`

---

**Status:** Complete Guide for AI Agents
**Version:** 3.0
**Last Updated:** 2025-11-29
**Changes:** Updated for 12 consolidated MCP tools, added `thinking` as primary tool
