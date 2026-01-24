# Session Context Guide

**CRITICAL: Call `session(action="context")` at the START of every session!**

## Overview

The `session` tool with `action="context"` provides session continuity for RETER MCP. It must be called:

1. **At the START of every new session**
2. **After every context compactification** (by Claude Code)
3. **When resuming work after any interruption**

This restores: **design doc sections**, **tasks with categories**, **project health**, **milestones**, and **suggestions**.

## Quick Start

```python
# ALWAYS call this first!
session(action="context")
```

## Session Tool Actions

| Action | Purpose | Returns |
|--------|---------|---------|
| `start` | Begin new session with goal | session_id, goal, status |
| `context` | **Restore full context** | thoughts, requirements, recommendations, project, suggestions |
| `end` | Archive session (preserves data) | summary |
| `clear` | Reset session (deletes all) | items_deleted |

## Response Structure for `context` Action

```json
{
  "success": true,
  "session": {
    "session_id": "...",
    "goal": "Analyze and refactor codebase",
    "status": "active",
    "created_at": "2025-11-29T10:00:00"
  },
  "design_doc": {
    "total": 15,
    "by_section": {"context": 2, "goals": 1, "design": 3, "tasks": 2},
    "sections": {
      "context": [{"id": "...", "num": 1, "summary": "..."}],
      "goals": [...],
      "design": [...]
    },
    "latest_chain": [...]
  },
  "tasks": {
    "total": 10,
    "completed": 6,
    "in_progress": 2,
    "pending": 2,
    "progress_percent": 60,
    "by_category": {"feature": {...}, "bug": {...}},
    "by_priority": {"critical": 1, "high": 3, "medium": 6}
  },
  "project_health": {
    "percent_complete": 60,
    "timeline": {"days_remaining": 14, "on_track": true},
    "overdue_count": 0
  },
  "milestones": [
    {"id": "MS-001", "name": "MVP", "target_date": "2025-02-01", "days_until": 14}
  ],
  "suggestions": [
    "Continue thought chain from #15",
    "2 pending tasks to address"
  ],
  "mcp_guide": {
    "tools": {...},
    "design_doc_sections": {...},
    "task_categories": [...],
    "recommended_workflow": [...]
  }
}
```

## Why Call This First?

### Session Continuity
Without calling `session(action="context")`, you lose visibility into:
- Previous reasoning chain (thoughts)
- Outstanding recommendations
- Task progress
- Project timeline

### MCP Guide
The response includes complete documentation on:
- All 12 MCP tools
- Available resources
- Recommended workflow

### Actionable Suggestions
Based on current state, you get suggestions like:
- "Resume in-progress work - 4 items started"
- "Continue analysis - 11 recommendations pending"

## The 12 MCP Tools

After calling `session(action="context")`, you have access to:

### Core Tools
| Tool | Purpose |
|------|---------|
| `thinking` | **PRIMARY** - Reasoning with operations |
| `session` | Session lifecycle |
| `items` | Query/manage items |
| `project` | Project analytics |
| `diagram` | Generate diagrams |

### Knowledge Tools
| Tool | Purpose |
|------|---------|
| `add_knowledge` | Add facts/rules to RETER |
| `add_external_python_directory` | Load external Python files |
| `quick_query` | Execute REQL queries |
| `natural_language_query` | **RECOMMENDED** - Plain English queries |

### Analysis Tools
| Tool | Purpose |
|------|---------|
| `instance_manager` | Manage instances/sources |
| `code_inspection` | Python code analysis (26 actions) |
| `recommender` | refactoring, test_coverage |

## Using the `thinking` Tool - Design Docs Approach

After restoring context, use `thinking` with **design doc sections**:

```python
# Document context/problem
thinking(
    thought="The auth module needs OAuth support...",
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

# Create tasks with categories
thinking(
    thought="Breaking down implementation",
    thought_number=4,
    total_thoughts=5,
    section="tasks",
    operations={
        "task": {"name": "Add NextAuth.js", "category": "feature", "priority": "high"},
        "milestone": {"name": "OAuth Ready", "date": "2025-01-15"}
    }
)
```

**Design Doc Sections:** `context`, `goals`, `non_goals`, `design`, `alternatives`, `risks`, `implementation`, `tasks`

**Task Categories:** `feature`, `bug`, `refactor`, `test`, `docs`, `research`

**Operations in `thinking`:**
- Create: `task` (with category), `milestone`
- Relate: `traces`, `implements`, `depends_on`, `affects`
- Update: `update_task`, `complete_task`
- RETER: `assert`, `query`

## Example Workflow - Design Docs

```python
# 1. Always start with context
session(action="context")

# 2. Document the problem
thinking(thought="Need to refactor auth...", thought_number=1, total_thoughts=5, section="context")

# 3. Define goals
thinking(thought="Goals: 1) Simplify 2) Add OAuth", thought_number=2, total_thoughts=5, section="goals")

# 4. Analyze code
code_inspection(action="describe_class", target="AuthService")

# 5. Document design decision
thinking(thought="Will use Strategy pattern", thought_number=3, total_thoughts=5, section="design")

# 6. Create tasks
thinking(
    thought="Implementation tasks",
    thought_number=4,
    total_thoughts=5,
    section="tasks",
    operations={
        "task": {"name": "Refactor AuthService", "category": "refactor", "priority": "high"}
    }
)

# 7. Visualize design doc
diagram(diagram_type="design_doc")

# 8. When done
session(action="end")
```

## Related Resources

| Resource | URI |
|----------|-----|
| Complete usage guide | `guide://logical-thinking/usage` |
| Python tools | `python://reter/tools` |
| Refactoring recipes | `recipe://refactoring/index` |
| Multiple instances | `system://reter/multiple-instances` |

## See Also

- Use `items(action="list")` to query items with filters
- Use `project(action="health")` for project status
- Use `diagram(diagram_type="gantt")` for visualization

---

**Remember:** Call `session(action="context")` at the START of every session and after any context compactification!
