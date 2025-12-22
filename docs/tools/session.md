# session Tool

Session lifecycle management for reasoning sessions.

## Overview

The `session` tool manages the lifecycle of reasoning sessions. Sessions persist thoughts, requirements, tasks, and their relationships.

**CRITICAL**: Always call `session(action="context")` at the start of every conversation to restore your reasoning state.

## Basic Usage

```python
# Restore context (do this first!)
session(action="context")

# Start a new session with a goal
session(
    action="start",
    goal="Implement user authentication feature",
    project_start="2024-01-15",
    project_end="2024-02-15"
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | str | required | Action to perform |
| `instance_name` | str | "default" | Session instance name |
| `goal` | str | None | Session goal (for start action) |
| `project_start` | str | None | Project start date ISO format |
| `project_end` | str | None | Project end date ISO format |

## Actions

### `start`

Begin a new session with optional goal and timeline.

```python
session(
    action="start",
    goal="Refactor authentication module",
    project_start="2024-01-15",
    project_end="2024-02-28"
)
```

Returns:
```python
{
    "success": True,
    "session_id": "abc123",
    "goal": "Refactor authentication module",
    "status": "active",
    "created_at": "2024-01-15T10:00:00Z"
}
```

### `context`

**CRITICAL**: Restore full context after compactification or at session start.

```python
session(action="context")
```

Returns:
```python
{
    "success": True,
    "session": {
        "session_id": "abc123",
        "goal": "Refactor authentication module",
        "status": "active",
        "created_at": "2024-01-15T10:00:00Z"
    },
    "design_doc": {
        "total": 5,
        "max_number": 5,
        "by_section": {"context": 1, "goals": 1, "design": 2, "tasks": 1},
        "sections": {
            "context": [{"id": "...", "num": 1, "summary": "..."}],
            "goals": [...],
            "design": [...]
        },
        "latest_chain": [...]
    },
    "tasks": {
        "total": 8,
        "completed": 2,
        "in_progress": 3,
        "blocked": 1,
        "pending": 2,
        "progress_percent": 25,
        "by_category": {"feature": {...}, "bug": {...}},
        "by_priority": {"critical": 1, "high": 3, "medium": 4},
        "urgent": [...],
        "blocked_tasks": [...]
    },
    "project_health": {
        "percent_complete": 25,
        "timeline": {"project_start": "...", "project_end": "...", "days_remaining": 30},
        "overdue_count": 0,
        "overdue_tasks": []
    },
    "milestones": [
        {"id": "MS-001", "name": "MVP", "target_date": "2024-02-01", "days_until": 15}
    ],
    "suggestions": ["Continue thought chain from #5", "2 pending tasks to address"],
    "mcp_guide": {
        "tools": {...},
        "design_doc_sections": {...},
        "task_categories": [...],
        "recommended_workflow": [...]
    }
}
```

### `end`

Archive the session (preserves all data).

```python
session(action="end")
```

Returns:
```python
{
    "success": True,
    "session_id": "abc123",
    "status": "archived",
    "summary": {
        "total_thoughts": 25,
        "total_requirements": 5,
        "total_tasks": 12,
        "completion_rate": 0.75
    }
}
```

### `clear`

Reset session (deletes all data). Use with caution!

```python
session(action="clear")
```

Returns:
```python
{
    "success": True,
    "items_deleted": 45
}
```

## Workflow - Design Docs Approach

1. **Start of conversation**: `session(action="context")` - restore state
2. **Document context**: `thinking(section="context", ...)`
3. **Define goals**: `thinking(section="goals", ...)`
4. **Document design**: `thinking(section="design", ...)`
5. **Create tasks**: `thinking(section="tasks", operations={...})`
6. **Visualize**: `diagram(diagram_type="design_doc")`
7. **End of session**: `session(action="end")`

## Multiple Sessions

Use different `instance_name` values to maintain separate sessions:

```python
# Session for feature A
session(action="start", instance_name="feature-a", goal="Implement feature A")

# Session for feature B
session(action="start", instance_name="feature-b", goal="Implement feature B")

# Switch between sessions
session(action="context", instance_name="feature-a")
```

## Requirements

- Requires SQLite component to be ready
