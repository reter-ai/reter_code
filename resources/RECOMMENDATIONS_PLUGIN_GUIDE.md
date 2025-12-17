# Recommendations Plugin Guide

A comprehensive MCP plugin for session continuity, tracking recommendations, activities, and artifacts.

## Quick Reference

| Tool | Description |
|------|-------------|
| `session_check_context` | **CALL FIRST** - Get MCP guide & previous work |
| `session_end` | Archive current session |
| `rec_create` | Create recommendation |
| `rec_update_status` | Update status (pending/in_progress/completed/skipped) |
| `rec_list` | List all recommendations |
| `rec_get` | Get recommendation details |
| `rec_delete` | Delete recommendation |
| `rec_get_progress` | Get progress summary |
| `rec_link_gantt` | Link to Gantt task |
| `activity_record` | Record tool activity |
| `activity_list` | List recent activities |
| `artifact_record` | Record generated file |
| `artifact_list` | List all artifacts |
| `artifact_check_freshness` | Check if artifact is stale |

## Tool Categories

### Session Management

```python
# ALWAYS call first!
session_check_context()
# Returns: mcp_guide, recommendations, artifacts, activities, suggestions

# Archive session when done
session_end()
```

### Recommendation CRUD

```python
# Create
rec_create(
    text="Extract DateExtractionService from LlmService",
    severity="high",      # critical, high, medium, low
    category="refactoring",
    affected_files=["services/llm_service.py"]
)

# Update status
rec_update_status(rec_id="REC_0001", status="in_progress")
rec_update_status(rec_id="REC_0001", status="completed")

# Query
rec_list(status="pending", severity="high")
rec_get(rec_id="REC_0001")
rec_get_progress()  # Returns total, completed, pending, progress_percent

# Link to Gantt
rec_link_gantt(rec_id="REC_0001", gantt_task_id="TASK-001")

# Delete
rec_delete(rec_id="REC_0001")
```

### Activity Tracking

```python
# Record tool invocation (usually automatic)
activity_record(
    recorded_tool_name="python_advanced_detect_code_smells",
    result_summary="Found 12 code smells",
    issues_found=12,
    files_analyzed=["services/llm_service.py"]
)

# List recent activities
activity_list(limit=10)
```

### Artifact Tracking

```python
# Record generated file
artifact_record(
    file_path="REFACTORING_RECOMMENDATIONS.md",
    artifact_type="recommendations",
    tool_name="refactoring_improving_detector",
    source_files=["services/llm_service.py", "services/date_service.py"]
)

# Check freshness (have source files changed?)
artifact_check_freshness(file_path="REFACTORING_RECOMMENDATIONS.md")
# Returns: {"fresh": false, "changed_source_files": ["services/llm_service.py"]}

# List all artifacts
artifact_list()
```

## Severity Levels

| Level | Description | Priority |
|-------|-------------|----------|
| `critical` | Security issues, data loss risks | 1 (highest) |
| `high` | Major code smells, architectural issues | 2 |
| `medium` | Moderate improvements | 3 |
| `low` | Minor cleanups, style issues | 4 (lowest) |

## Status Workflow

```
pending → in_progress → completed
                     ↘ skipped
```

- `pending`: Not started (default)
- `in_progress`: Currently being worked on
- `completed`: Done
- `skipped`: Intentionally not doing

## Integration with Refactoring Plugins

The `refactoring_improving` and `refactoring_to_patterns` plugins automatically create recommendations:

```python
# 1. Prepare creates recommendations for each available detector
refactoring_improving_prepare()
# Creates: REC_0001 (long_method), REC_0002 (god_class), ...

# 2. Running a detector marks its recommendation complete
#    and creates new recommendations for findings
refactoring_improving_detector(detector_name="long_method")
# Marks REC_0001 complete
# Creates: REC_0010 (LlmService.process is 150 lines), ...

# 3. Check overall progress
rec_get_progress()
# Returns: {"total": 60, "completed": 15, "progress_percent": 25.0}
```

## Integration with Gantt Plugin

```python
# Create recommendation
rec_create(
    text="Extract DateExtractionService",
    severity="high",
    rec_id="REC-EXTRACT-DATE"
)

# Create corresponding Gantt task
gantt_create_task(
    task_id="TASK-EXTRACT-DATE",
    name="Extract DateExtractionService",
    duration_days=3
)

# Link them for tracking
rec_link_gantt(
    rec_id="REC-EXTRACT-DATE",
    gantt_task_id="TASK-EXTRACT-DATE"
)
```

## Instance Defaults

All tools use `instance_name="current"` by default. You don't need to specify it.

## Technical Details

### FORGET-THEN-ADD Pattern

Status updates use RETER's FORGET-THEN-ADD pattern:

```python
# Updating status:
# 1. Forget old status
reter.forget_source("rec_REC_0001_status")

# 2. Add new status
reter.add_ontology('has_status(REC_0001, "completed")',
                   source="rec_REC_0001_status")
```

### Source ID Categories

```
IMMUTABLE (core facts):
  rec_{id}_core      # text, severity, category
  rec_{id}_files     # affected files
  activity_{id}      # tool invocation
  artifact_{id}      # generated file

MUTABLE (via FORGET-THEN-ADD):
  rec_{id}_status    # current status
  rec_{id}_gantt     # linked Gantt task
```

## See Also

- `guide://reter/session-context` - Session context guide
- `guide://plugins/system` - Plugin system overview
- `guide://logical-thinking/usage` - General usage guide
