# Multiple RETER Instances Support

**Version:** 2.2.0
**Date:** 2025-11-11
**Status:** ✅ Complete

## Overview

The logical-thinking MCP server now supports **multiple independent RETER instances**, allowing you to maintain separate knowledge bases for different purposes simultaneously.

## Key Design

### Auto-Creation on Demand

- **No instance switching** - Each tool call specifies which instance to use
- **Automatic creation** - If instance doesn't exist, it's created automatically
- **Independent state** - Each instance maintains its own knowledge base
- **Named instances** - Use meaningful names like "main", "experiment", "backup"
- **Thread-safe** - Per-instance locks ensure safe concurrent access

### Benefits

1. **Isolation**: Separate knowledge bases don't interfere with each other
2. **Experimentation**: Try different approaches without affecting production
3. **Comparison**: Run same queries on different datasets
4. **Backup/Restore**: Create snapshots without disrupting active instance
5. **Multi-Project**: Work on multiple projects in parallel
6. **Thread-Safety**: Built-in locking prevents data corruption from concurrent access
7. **Automatic Persistence**: All instances automatically saved/restored across server restarts

### Thread Safety

**IMPORTANT:** RETER is NOT thread-safe. The server implements per-instance locking:

- ✅ **Safe concurrent access**: Multiple operations can run on different instances in parallel
- ✅ **Serialized same-instance access**: Operations on the same instance are queued
- ✅ **Automatic lock management**: No user action required
- ✅ **No deadlocks**: Single lock acquisition per operation

See `THREAD_SAFETY.md` for technical details.

### Automatic Snapshots (v2.2.0)

**NEW:** All RETER instances now automatically persist across server restarts!

- ✅ **Zero configuration**: No manual save/load needed
- ✅ **On shutdown**: All instances automatically saved to `.reter/` directory
- ✅ **On startup**: All snapshots automatically restored
- ✅ **Per-project storage**: Each instance saved as `{instance_name}.reter`
- ✅ **Crash recovery**: Resume work from last shutdown state

**Example:**
```python
# Session 1: Add knowledge
add_knowledge(instance_name="main", source="Person is_a Thing", type="ontology")
# Server shutdown → main.reter saved automatically

# Session 2: After restart
query = quick_query(instance_name="main", query="SELECT ?x WHERE { ?x type Person }")
# Works! Knowledge automatically restored
```

See `AUTOMATIC_SNAPSHOTS.md` for complete documentation.

## Breaking Changes

**⚠️ All tools now require `instance_name` as the FIRST parameter**

### Before (v2.0.x):
```python
add_knowledge(source="Person is_a Thing", type="ontology")
quick_query(query="SELECT ?x WHERE { ?x type Person }", type="reql")
```

### After (v2.1.0):
```python
add_knowledge(instance_name="main", source="Person is_a Thing", type="ontology")
quick_query(instance_name="main", query="SELECT ?x WHERE { ?x type Person }", type="reql")
```

## Updated Tool Signatures

### logical_thinking
```python
logical_thinking(
    instance_name: str,          # NEW: Required first parameter
    thought: str,
    next_thought_needed: bool,
    thought_number: int,
    total_thoughts: int,
    ...
)
```

### add_knowledge
```python
add_knowledge(
    instance_name: str,          # NEW: Required first parameter
    source: str,
    type: str = "ontology",
    source_id: Optional[str] = None
)
```

### quick_query
```python
quick_query(
    instance_name: str,          # NEW: Required first parameter
    query: str,
    type: str = "reql"
)
```

### forget_source
```python
forget_source(
    instance_name: str,          # NEW: Required first parameter
    source: str
)
```

### save_state
```python
save_state(
    instance_name: str,          # NEW: Required first parameter
    filename: str
)
```

### load_state
```python
load_state(
    instance_name: str,          # NEW: Required first parameter
    filename: str
)
```

### check_consistency
```python
check_consistency(
    instance_name: str           # NEW: Required first parameter
)
```

## Usage Examples

### Example 1: Single Instance (Most Common)

```python
# All operations on "main" instance
add_knowledge(
    instance_name="main",
    source="Person is_a Thing",
    type="ontology"
)

quick_query(
    instance_name="main",
    query="SELECT ?x WHERE { ?x type Person }",
    type="reql"
)
```

### Example 2: Multiple Independent Projects

```python
# Project A: Refactoring analysis
add_knowledge(
    instance_name="refactoring",
    source="code_smells.py",
    type="python"
)

# Project B: Domain modeling
add_knowledge(
    instance_name="domain_model",
    source="Customer is_a Person",
    type="ontology"
)

# Query each independently
refactoring_smells = quick_query(
    instance_name="refactoring",
    query="SELECT ?func WHERE { ?func type py:Function }"
)

customers = quick_query(
    instance_name="domain_model",
    query="SELECT ?x WHERE { ?x type Customer }"
)
```

### Example 3: Experimentation Without Risk

```python
# Production instance
add_knowledge(
    instance_name="production",
    source="stable_ontology.reol",
    type="ontology"
)

# Experimental instance
add_knowledge(
    instance_name="experiment",
    source="new_rules.reol",
    type="ontology"
)

# Compare results
prod_results = quick_query(instance_name="production", query="...")
exp_results = quick_query(instance_name="experiment", query="...")

# If experiment works, promote to production
# If not, discard experiment instance (it's independent!)
```

### Example 4: Backup and Restore

```python
# Save snapshot of current state
save_state(
    instance_name="main",
    filename="backup_2025_11_10.reter"
)

# Try risky operation on main
add_knowledge(instance_name="main", source="risky_rules.reol")

# If something goes wrong, restore from backup
load_state(
    instance_name="main",
    filename="backup_2025_11_10.reter"
)
```

### Example 5: A/B Testing Knowledge Bases

```python
# Setup A: Using SWRL rules
add_knowledge(
    instance_name="approach_a",
    source="if Person(object x) then Adult(object x)",
    type="ontology"
)

# Setup B: Using class hierarchy
add_knowledge(
    instance_name="approach_b",
    source="Adult is_subclass_of Person",
    type="ontology"
)

# Compare performance and results
result_a = quick_query(instance_name="approach_a", query="...")
result_b = quick_query(instance_name="approach_b", query="...")
```

## Instance Lifecycle

### Creation
```python
# First call creates instance automatically
add_knowledge(instance_name="new_instance", source="...")
# Output: 🆕 Creating new RETER instance: 'new_instance'
```

### Persistence

**Automatic (v2.2.0):**
- **On shutdown**: All instances automatically saved to `.reter/` directory
- **On startup**: All instances automatically restored from `.reter/` directory
- **Zero configuration**: No manual action required
- **Crash recovery**: Resume from last shutdown state

**Manual (still available):**
- Use `save_state(instance_name, filename)` for explicit backups
- Use `load_state(instance_name, filename)` for explicit restores
- Useful for cross-machine transfer or specific checkpoints

### Naming Conventions
- Use descriptive names: "main", "production", "experiment", "backup"
- Alphanumeric + underscores recommended
- Case-sensitive: "Main" ≠ "main"

## Migration Guide

### Step 1: Update Tool Calls

Find all tool calls and add `instance_name` as first parameter:

```python
# Old (v2.0.x)
add_knowledge(source="...", type="ontology")

# New (v2.1.0)
add_knowledge(instance_name="main", source="...", type="ontology")
```

### Step 2: Choose Instance Names

For simple cases, use `"main"` everywhere:

```python
add_knowledge(instance_name="main", ...)
quick_query(instance_name="main", ...)
save_state(instance_name="main", ...)
```

### Step 3: Test

```python
# Verify instance creation
add_knowledge(
    instance_name="test",
    source="TestClass is_a Thing",
    type="ontology"
)

# Verify queries work
result = quick_query(
    instance_name="test",
    query="SELECT ?x WHERE { ?x type TestClass }",
    type="reql"
)

print(f"Success: {result['success']}")
```

## Implementation Details

### Server Changes

**File:** `src/logical_thinking_server/server.py`

```python
class LogicalThinkingServer:
    def __init__(self):
        # Before: Single instance
        # self.reter = ReterWrapper()

        # After: Dict of instances
        self.reter_instances: Dict[str, ReterWrapper] = {}

    def _get_or_create_instance(self, instance_name: str) -> ReterWrapper:
        """Get existing instance or create new one"""
        if instance_name not in self.reter_instances:
            print(f"🆕 Creating new RETER instance: '{instance_name}'")
            self.reter_instances[instance_name] = ReterWrapper()
        return self.reter_instances[instance_name]
```

### Each Tool Updated

Every tool now:
1. Takes `instance_name` as first parameter
2. Calls `self._get_or_create_instance(instance_name)`
3. Uses the returned instance for operations

## Best Practices

### 1. Use Consistent Naming

```python
# Good: Consistent naming across related operations
add_knowledge(instance_name="refactoring", ...)
quick_query(instance_name="refactoring", ...)
save_state(instance_name="refactoring", ...)

# Bad: Typos create unintended instances
add_knowledge(instance_name="refactoring", ...)
quick_query(instance_name="refactorign", ...)  # Typo! Creates new instance
```

### 2. Document Your Instances

```python
# At the start of your script/workflow
INSTANCES = {
    "main": "Primary knowledge base",
    "experiment": "Testing new inference rules",
    "backup": "Snapshot from 2025-11-10"
}

for name, description in INSTANCES.items():
    print(f"Instance '{name}': {description}")
```

### 3. Leverage Automatic Snapshots

```python
# v2.2.0+: Instances automatically persist across restarts
# No manual save/load needed for regular workflow!

# Still use manual save for:
# - Explicit backups before risky operations
# - Cross-machine transfer
# - Specific versioned checkpoints
save_state(
    instance_name="main",
    filename=f"main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.reter"
)
```

### 4. Manage Snapshot Directory

```python
# v2.2.0+: Snapshots persist across restarts in .reter/ directory

# Clean up obsolete snapshots manually:
# rm .reter/old_experiment.reter
# rm .reter/temp_test.reter

# View all snapshots:
# ls .reter/

# Check snapshot sizes:
# du -sh .reter/
```

## Limitations

1. **No instance deletion tool** - Remove snapshots manually from `.reter/` directory (v2.2.0+)
2. **Memory overhead** - Each instance has independent RETER network in memory
3. **No cross-instance queries** - Can't query across multiple instances
4. **No instance listing tool** - No tool to list all created instances (yet)

## Future Enhancements

Potential future additions:
- `list_instances()` - List all created instances
- `delete_instance(name)` - Explicitly remove instance
- `copy_instance(source, dest)` - Clone an instance
- `merge_instances(sources, dest)` - Combine multiple instances

## Testing

```python
# Test multi-instance independence
add_knowledge(
    instance_name="test1",
    source="Cat is_a Animal",
    type="ontology"
)

add_knowledge(
    instance_name="test2",
    source="Dog is_a Animal",
    type="ontology"
)

# Verify isolation
cats = quick_query(
    instance_name="test1",
    query="SELECT ?x WHERE { ?x type Cat }",
    type="reql"
)
assert cats['count'] == 1

dogs = quick_query(
    instance_name="test2",
    query="SELECT ?x WHERE { ?x type Dog }",
    type="reql"
)
assert dogs['count'] == 1

# Verify test1 doesn't have dogs
no_dogs = quick_query(
    instance_name="test1",
    query="SELECT ?x WHERE { ?x type Dog }",
    type="reql"
)
assert no_dogs['count'] == 0  # Success!
```

## Resources

- **Server Implementation:** `src/logical_thinking_server/server.py`
- **Automatic Snapshots:** `AUTOMATIC_SNAPSHOTS.md` (v2.2.0)
- **Thread Safety:** `THREAD_SAFETY.md`
- **Main Documentation:** `AI_AGENT_USAGE_GUIDE.md`
- **Refactoring Summary:** `REFACTORING_SUMMARY.md`

---

**Version:** 2.2.0
**Features:**
- v2.1.0: Multiple instances with thread-safe access
- v2.2.0: Automatic snapshot persistence (`.reter/` directory)

**Breaking Changes:** Yes (v2.1.0) - all tools require instance_name parameter
**Backward Compatible:** No - v2.0.x code will not work without updates
**Migration Effort:** Low - add `instance_name="main"` to all tool calls
