# thinking Tool

The primary tool for recording reasoning steps, analysis, and decisions with integrated operations.

## Overview

The `thinking` tool is the main entry point for Codeine's reasoning system. It creates structured thoughts that can:
- Organize reasoning into **design doc sections** (context, goals, design, etc.)
- Create tasks with categories and milestones
- Create relationships between items
- Execute RETER operations

## Design Docs Workflow

```python
# 1. Document the context/problem
thinking(
    thought="We need to add OAuth support to the auth module...",
    thought_number=1,
    total_thoughts=5,
    section="context"
)

# 2. Define goals
thinking(
    thought="Goals: 1) Support Google/GitHub OAuth 2) Maintain existing login",
    thought_number=2,
    total_thoughts=5,
    section="goals"
)

# 3. Document design decision
thinking(
    thought="Using NextAuth.js because it has built-in providers...",
    thought_number=3,
    total_thoughts=5,
    section="design",
    thought_type="decision"
)

# 4. Create tasks
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

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `thought` | str | required | Your current reasoning step |
| `thought_number` | int | required | Current step number (1-indexed) |
| `total_thoughts` | int | required | Estimated total steps |
| `instance_name` | str | "default" | Session instance name |
| `thought_type` | str | "reasoning" | Type: reasoning, analysis, decision, planning, verification |
| `section` | str | None | Design doc section (see below) |
| `next_thought_needed` | bool | True | Whether more thoughts are needed |
| `branch_id` | str | None | ID for branching |
| `branch_from` | int | None | Thought number to branch from |
| `is_revision` | bool | False | Whether this revises a previous thought |
| `revises_thought` | int | None | Which thought number is being revised |
| `needs_more_thoughts` | bool | False | Signal that more analysis is needed |
| `operations` | dict | None | Operations to execute (see below) |

## Design Doc Sections

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

## Thought Types

| Type | Use For |
|------|---------|
| `reasoning` | General logical reasoning |
| `analysis` | Analyzing code, data, or situations |
| `decision` | Making choices between options |
| `planning` | Planning implementation steps |
| `verification` | Verifying or validating results |

## Operations

The `operations` parameter allows executing actions within a thought:

### Create Items

```python
thinking(
    thought="We need to implement user authentication",
    thought_number=1,
    total_thoughts=5,
    operations={
        "requirement": {
            "text": "System shall authenticate users via JWT tokens",
            "risk": "medium",
            "priority": "high"
        }
    }
)
```

Available item types:
- `requirement` - Create a requirement
- `recommendation` - Create a recommendation
- `task` - Create a task with scheduling
- `milestone` - Create a milestone
- `decision` - Record a decision

### Create Task with Category and Scheduling

```python
operations={
    "task": {
        "name": "Implement JWT authentication",
        "category": "feature",  # feature, bug, refactor, test, docs, research
        "priority": "high",     # critical, high, medium, low
        "start_date": "2024-01-15",
        "duration_days": 5
    }
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
    "traces": ["REQ-001", "REQ-002"],  # This thought traces to these items
    "satisfies": ["REQ-001"],           # This thought satisfies these requirements
    "implements": ["REC-001"],          # This thought implements these recommendations
    "depends_on": ["TASK-001"],         # This thought depends on these items
    "affects": ["module.py", "utils.py"] # This thought affects these files
}
```

### Update Items

```python
operations={
    "update_task": {"task_id": "TASK-001", "status": "in_progress"},
    "complete_task": "TASK-001"
}
```

### RETER Operations

```python
operations={
    "query": "SELECT ?class WHERE { ?class type py:Class }",
    "assert": "myFact(value1, value2)"
}
```

## Branching

Create alternative reasoning paths:

```python
# Main reasoning path
thinking(thought="Option A: Use JWT", thought_number=1, total_thoughts=3)
thinking(thought="JWT provides stateless auth", thought_number=2, total_thoughts=3)

# Branch to explore alternative
thinking(
    thought="Option B: Use sessions",
    thought_number=3,
    total_thoughts=5,
    branch_id="session-option",
    branch_from=1
)
```

## Revision

Revise previous thoughts:

```python
thinking(
    thought="After further analysis, JWT is better because...",
    thought_number=4,
    total_thoughts=5,
    is_revision=True,
    revises_thought=2
)
```

## Returns

```python
{
    "success": True,
    "thought_id": "C90F-THOUGHT-001",
    "thought_number": 1,
    "total_thoughts": 3,
    "next_thought_needed": True,
    "thought_type": "analysis",
    "items_created": ["REQ-001"],
    "items_updated": [],
    "relations_created": [{"from": "C90F-THOUGHT-001", "to": "REQ-001", "type": "creates"}],
    "reter_operations": {},
    "session_status": {
        "total_items": 5,
        "by_type": {"thought": 3, "requirement": 2},
        "by_status": {"pending": 2, "completed": 3},
        "thought_chain": {"max_number": 3, "total": 3}
    }
}
```

## Best Practices

1. **Start with context**: Use `session(action="context")` first
2. **Be specific**: Write clear, actionable thoughts
3. **Use appropriate types**: Match thought_type to what you're doing
4. **Create traceability**: Link thoughts to requirements and tasks
5. **Branch for alternatives**: Explore different approaches
6. **Revise when needed**: Update incorrect assumptions

## Requirements

- Requires SQLite component to be ready
- RETER operations require RETER component
