# Source Management Tools

**Version:** 2.2.0
**Date:** 2025-11-11
**Status:** ✅ Complete

## Overview

RETER tracks all knowledge by **source identifiers**, enabling selective forgetting and knowledge composition analysis. The logical-thinking MCP server now exposes tools to inspect which sources are loaded and what facts they contain.

## Source Tracking Basics

### What is a Source?

A **source** is a string identifier attached to knowledge when added to RETER:

```python
# Add knowledge with source identifier
add_knowledge(
    instance_name="main",
    source="base_ontology",  # ← Source ID
    ontology="Person is_a Thing",
    type="ontology"
)

add_knowledge(
    instance_name="main",
    source="user_data",  # ← Different source
    ontology="Person(Alice)",
    type="ontology"
)
```

### Why Track Sources?

1. **Selective Forgetting**: Remove specific knowledge fragments
2. **Knowledge Composition**: Understand what's loaded
3. **Debugging**: Track where facts came from
4. **Versioning**: Replace old rules with new ones
5. **Modularity**: Separate concerns (base ontology, rules, data)

## Available Tools

### 1. list_sources

List all source identifiers currently loaded in a RETER instance.

**Signature:**
```python
list_sources(instance_name: str) -> Dict[str, Any]
```

**Args:**
- `instance_name`: RETER instance to query

**Returns:**
```python
{
    "success": True,
    "sources": ["base_ontology", "rules", "user_data", "python_ontology"],
    "count": 4,
    "execution_time_ms": 0.5
}
```

**Example:**
```python
# List all sources in "main" instance
result = list_sources(instance_name="main")

if result["success"]:
    print(f"Found {result['count']} sources:")
    for source in result["sources"]:
        print(f"  - {source}")
```

**Output:**
```
Found 4 sources:
  - python_ontology
  - base_ontology
  - inference_rules
  - customer_data
```

### 2. get_source_facts

Get all fact IDs (WME identifiers) associated with a specific source.

**Signature:**
```python
get_source_facts(instance_name: str, source: str) -> Dict[str, Any]
```

**Args:**
- `instance_name`: RETER instance to query
- `source`: Source identifier to inspect

**Returns:**
```python
{
    "success": True,
    "source": "base_ontology",
    "fact_ids": [1, 2, 3, 5, 8, 13, 21],  # Internal WME IDs
    "count": 7,
    "execution_time_ms": 0.3
}
```

**Example:**
```python
# Check what facts came from "base_ontology"
result = get_source_facts(
    instance_name="main",
    source="base_ontology"
)

if result["success"]:
    print(f"Source '{result['source']}' contains {result['count']} facts")
    print(f"Fact IDs: {result['fact_ids']}")
```

**Output:**
```
Source 'base_ontology' contains 7 facts
Fact IDs: [1, 2, 3, 5, 8, 13, 21]
```

### 3. forget_source (Existing Tool)

Remove all facts from a specific source.

**Signature:**
```python
forget_source(instance_name: str, source: str) -> Dict[str, Any]
```

**Args:**
- `instance_name`: RETER instance
- `source`: Source identifier to forget

**Returns:**
```python
{
    "success": True,
    "message": "All facts from source 'old_rules' have been forgotten",
    "execution_time_ms": 1.2
}
```

## Usage Patterns

### Pattern 1: Inventory Check

```python
# Check what's loaded before adding more
sources = list_sources(instance_name="main")
print(f"Currently loaded: {sources['sources']}")

# Decide what to add based on what's already there
if "customer_ontology" not in sources["sources"]:
    add_knowledge(
        instance_name="main",
        source="customer_ontology",
        ontology="Customer is_subclass_of Person",
        type="ontology"
    )
```

### Pattern 2: Verify Loading

```python
# Add knowledge
add_knowledge(
    instance_name="main",
    source="test_data",
    ontology="Cat is_a Animal",
    type="ontology"
)

# Verify it was loaded
sources = list_sources(instance_name="main")
assert "test_data" in sources["sources"], "Source not loaded!"

# Check fact count
facts = get_source_facts(instance_name="main", source="test_data")
print(f"Loaded {facts['count']} facts from test_data")
```

### Pattern 3: Selective Cleanup

```python
# List all sources
all_sources = list_sources(instance_name="main")

# Find temporary/test sources
temp_sources = [s for s in all_sources["sources"] if s.startswith("temp_")]

# Remove them
for source in temp_sources:
    print(f"Removing temporary source: {source}")
    forget_source(instance_name="main", source=source)

# Verify cleanup
remaining = list_sources(instance_name="main")
print(f"Sources after cleanup: {remaining['sources']}")
```

### Pattern 4: Knowledge Replacement

```python
# Replace old rules with new ones
sources = list_sources(instance_name="main")

if "inference_rules_v1" in sources["sources"]:
    # Remove old version
    forget_source(instance_name="main", source="inference_rules_v1")
    print("Removed old inference rules")

# Add new version
add_knowledge(
    instance_name="main",
    source="inference_rules_v2",
    ontology="if Person(object x) also age(object x, var y) also (y >= 18) then Adult(object x)",
    type="ontology"
)
print("Added new inference rules")
```

### Pattern 5: Debugging Knowledge Base

```python
# List all sources
sources = list_sources(instance_name="main")

# Inspect each source
for source in sources["sources"]:
    facts = get_source_facts(instance_name="main", source=source)
    print(f"Source '{source}': {facts['count']} facts")

    # Show some fact IDs
    if facts['count'] > 0:
        sample = facts['fact_ids'][:5]
        print(f"  First few facts: {sample}")
```

**Output:**
```
Source 'python_ontology': 150 facts
  First few facts: [1, 2, 3, 4, 5]
Source 'base_ontology': 7 facts
  First few facts: [151, 152, 153, 154, 155]
Source 'customer_data': 25 facts
  First few facts: [158, 159, 160, 161, 162]
```

### Pattern 6: Source-Based Modularity

```python
# Modular ontology construction
ONTOLOGY_MODULES = {
    "base": "Person is_a Thing\\nCompany is_a Thing",
    "properties": "hasEmployee property Object\\nworksFor property Object",
    "rules": "if Company(object c) also hasEmployee(object c, object p) then worksFor(object p, object c)",
    "data": "Company(Acme)\\nPerson(Alice)\\nhasEmployee(Acme, Alice)"
}

# Load each module
for module_name, ontology in ONTOLOGY_MODULES.items():
    add_knowledge(
        instance_name="company_kb",
        source=f"module_{module_name}",
        ontology=ontology,
        type="ontology"
    )

# Verify all modules loaded
sources = list_sources(instance_name="company_kb")
expected = [f"module_{name}" for name in ONTOLOGY_MODULES.keys()]
loaded = [s for s in sources["sources"] if s.startswith("module_")]

print(f"Loaded {len(loaded)}/{len(expected)} modules")
for module in loaded:
    facts = get_source_facts(instance_name="company_kb", source=module)
    print(f"  {module}: {facts['count']} facts")
```

## Automatic Source IDs

If you don't provide a source ID, one is generated automatically:

```python
# Without source ID
add_knowledge(
    instance_name="main",
    ontology="Person is_a Thing",
    type="ontology"
)
# Auto-generates source like: "source_1731331200.123456"

# Check what was created
sources = list_sources(instance_name="main")
# Shows: ["python_ontology", "source_1731331200.123456"]
```

**Best Practice:** Always provide meaningful source IDs for better tracking!

```python
# ✅ GOOD: Meaningful source ID
add_knowledge(
    instance_name="main",
    source="base_ontology",
    ontology="Person is_a Thing",
    type="ontology"
)

# ❌ BAD: Auto-generated source
add_knowledge(
    instance_name="main",
    ontology="Person is_a Thing",
    type="ontology"
)
```

## Built-in Sources

### python_ontology

Every RETER instance automatically loads a `python_ontology` source:

```python
sources = list_sources(instance_name="main")
# Always includes: "python_ontology"
```

This ontology provides:
- Classes for Python code elements (Module, Class, Function, Method)
- Inference rules for code analysis
- Transitive relationships (calls, imports, inheritance)

**Note:** Don't forget this source unless you're sure you don't need Python code analysis!

## Thread Safety

All source management tools are **thread-safe**:
- Acquire per-instance lock before accessing RETER
- Operations on different instances run in parallel
- Operations on same instance are serialized

```python
# Thread-safe concurrent access
import asyncio

async def analyze_sources():
    # These run in parallel (different instances)
    await asyncio.gather(
        list_sources(instance_name="project_a"),
        list_sources(instance_name="project_b")
    )
```

## Performance

### list_sources
- **Complexity:** O(n) where n = number of sources
- **Typical time:** <1ms for <100 sources
- **Memory:** Negligible (returns list of strings)

### get_source_facts
- **Complexity:** O(m) where m = number of facts in source
- **Typical time:** <1ms for <1000 facts
- **Memory:** O(m) - returns list of fact IDs

### forget_source
- **Complexity:** O(m + r) where m = facts, r = derived facts
- **Typical time:** 1-10ms depending on inference chains
- **Memory:** Temporary (cleaned up after removal)

## Limitations

### 1. Fact IDs are Opaque

The fact IDs returned by `get_source_facts()` are internal RETER identifiers:

```python
result = get_source_facts(instance_name="main", source="base")
# Returns: [1, 2, 3, 5, 8]
# But what ARE these facts?
```

**No tool to convert fact ID → fact content** (yet).

**Workaround:** Use queries to inspect content:
```python
# Query all facts instead
result = quick_query(
    instance_name="main",
    query="SELECT ?s ?p ?o WHERE { ?s ?p ?o }",
    type="reql"
)
```

### 2. No Source Metadata

Sources are just string IDs - no metadata stored:

```python
# Can't query: When was this source added?
# Can't query: Who added this source?
# Can't query: What file did this come from?
```

**Best Practice:** Use descriptive source IDs that encode metadata:
```python
source = "ontology_v2_2025_11_11_by_alice"
source = "customer_data_production_2025_11"
```

### 3. Source Names Are Global

Within an instance, source names must be unique:

```python
# First load
add_knowledge(instance_name="main", source="data", ontology="Cat is_a Animal")

# Second load with SAME source ID
add_knowledge(instance_name="main", source="data", ontology="Dog is_a Animal")
# Both are under source "data" - can't distinguish!

# Forgetting removes BOTH
forget_source(instance_name="main", source="data")
# Both Cat and Dog facts removed!
```

**Best Practice:** Use versioned or timestamped source IDs:
```python
add_knowledge(instance_name="main", source="data_v1", ontology="Cat is_a Animal")
add_knowledge(instance_name="main", source="data_v2", ontology="Dog is_a Animal")
# Now can forget individually
```

## Examples

### Example 1: Audit Knowledge Base

```python
# Full audit of what's loaded
print("=== Knowledge Base Audit ===")

sources = list_sources(instance_name="main")
print(f"Total sources: {sources['count']}")
print()

total_facts = 0
for source in sources["sources"]:
    facts = get_source_facts(instance_name="main", source=source)
    total_facts += facts["count"]
    print(f"Source: {source}")
    print(f"  Facts: {facts['count']}")

print()
print(f"Total facts: {total_facts}")
```

**Output:**
```
=== Knowledge Base Audit ===
Total sources: 3

Source: python_ontology
  Facts: 150

Source: base_ontology
  Facts: 7

Source: customer_data
  Facts: 25

Total facts: 182
```

### Example 2: Clean Up Old Experiments

```python
# Find and remove experimental sources
sources = list_sources(instance_name="main")

experimental = [s for s in sources["sources"]
                if "experiment" in s.lower() or "test" in s.lower()]

print(f"Found {len(experimental)} experimental sources:")
for source in experimental:
    facts = get_source_facts(instance_name="main", source=source)
    print(f"  {source}: {facts['count']} facts")

if experimental:
    response = input("Remove all experimental sources? (y/n): ")
    if response.lower() == "y":
        for source in experimental:
            forget_source(instance_name="main", source=source)
            print(f"  ✓ Removed {source}")
```

### Example 3: Validate Loading Pipeline

```python
# Incremental loading with validation
pipeline = [
    ("base", "Thing is_a Entity"),
    ("types", "Person is_subclass_of Thing"),
    ("properties", "age property Integer"),
    ("rules", "if Person(object x) also age(object x, var y) also (y >= 18) then Adult(object x)"),
    ("data", "Person(Alice)\\nage(Alice, 25)")
]

for source_id, ontology in pipeline:
    # Load
    result = add_knowledge(
        instance_name="pipeline",
        source=source_id,
        ontology=ontology,
        type="ontology"
    )

    # Validate
    if result["success"]:
        facts = get_source_facts(instance_name="pipeline", source=source_id)
        print(f"✓ Loaded {source_id}: {facts['count']} facts")
    else:
        print(f"✗ Failed to load {source_id}: {result['errors']}")
        break

# Final check
sources = list_sources(instance_name="pipeline")
expected = [s for s, _ in pipeline]
loaded = [s for s in sources["sources"] if s in expected]

if len(loaded) == len(expected):
    print(f"\n✓ Pipeline complete: {len(loaded)}/{len(expected)} sources loaded")
else:
    print(f"\n✗ Pipeline incomplete: {len(loaded)}/{len(expected)} sources loaded")
```

## Integration with Automatic Snapshots

Source tracking information is preserved across server restarts:

```python
# Session 1: Add knowledge with sources
add_knowledge(instance_name="main", source="base", ontology="...")
add_knowledge(instance_name="main", source="rules", ontology="...")

# Server shutdown → automatic save to .reter/main.reter
# Server restart → automatic load from .reter/main.reter

# Session 2: Sources still tracked!
sources = list_sources(instance_name="main")
# Returns: ["python_ontology", "base", "rules"]
```

## Tool Count Summary

**Total Source Management Tools: 3**

1. ✅ `list_sources` - List all source IDs
2. ✅ `get_source_facts` - Get fact IDs for a source
3. ✅ `forget_source` - Remove a source (existing)

## Related Documentation

- **Multi-Instance Guide:** `MULTIPLE_INSTANCES.md`
- **Automatic Snapshots:** `AUTOMATIC_SNAPSHOTS.md`
- **Server Implementation:** `src/logical_thinking_server/server.py`
- **Wrapper Implementation:** `src/logical_thinking_server/reter_wrapper.py`

---

**Version:** 2.2.0
**New Tools:** `list_sources`, `get_source_facts`
**Date:** 2025-11-11
**Status:** ✅ Complete
