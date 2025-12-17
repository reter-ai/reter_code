# Session Context Guide

**CRITICAL: Call `session(action="context")` at the START of every session!**

## Overview

The `session` tool with `action="context"` provides session continuity for RETER MCP. It must be called:

1. **At the START of every new session**
2. **After every context compactification** (by Claude Code)
3. **When resuming work after any interruption**

This restores: thoughts, requirements, recommendations, project status, and suggestions.

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
  "thoughts": {
    "total": 15,
    "by_type": {"reasoning": 8, "analysis": 4, "decision": 3}
  },
  "requirements": {
    "total": 5,
    "by_status": {"verified": 2, "pending": 3}
  },
  "recommendations": {
    "total": 47,
    "completed": 32,
    "in_progress": 4,
    "pending": 11,
    "progress_percent": 68.1
  },
  "project": {
    "tasks": {"total": 10, "completed": 6},
    "milestones": [...]
  },
  "suggestions": [
    "Continue work - 11 recommendations pending (68.1% complete)",
    "4 tasks in progress"
  ],
  "mcp_guide": {
    "tools": [...],
    "resources": [...]
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

## Using the `thinking` Tool

After restoring context, use `thinking` for all reasoning:

```python
thinking(
    thought="Analyzing the authentication module for security issues",
    thought_number=1,
    total_thoughts=5,
    thought_type="analysis",
    operations={
        "requirement": {"text": "System shall validate all user inputs"},
        "traces": ["auth_module.py"]
    }
)
```

**Operations in `thinking`:**
- Create: `requirement`, `recommendation`, `task`, `milestone`, `decision`
- Relate: `traces`, `satisfies`, `implements`, `depends_on`, `affects`
- Update: `update_item`, `update_task`, `complete_task`
- RETER: `assert`, `query`, `python_file`, `forget_source`

## Example Workflow

```python
# 1. Always start with context
session(action="context")

# 2. Start reasoning with thinking tool
thinking(
    thought="Reviewing codebase structure",
    thought_number=1,
    total_thoughts=3,
    thought_type="planning"
)

# 3. Analyze code
code_inspection(action="list_classes")

# 4. Find issues
recommender("refactoring")

# 5. Track progress
thinking(
    thought="Found god class pattern in UserService",
    thought_number=2,
    total_thoughts=3,
    operations={
        "recommendation": {
            "text": "Split UserService into smaller classes",
            "severity": "high"
        }
    }
)

# 6. When done
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
