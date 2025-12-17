# RETER Syntax Quick Reference

**⚠️ READ THIS FIRST to avoid empty query results!**

---

## ⚡ Most Common Mistake

### ❌ WRONG (Creates Subclass, Not Instance)
```python
add_knowledge("task:Task1 is_a task:Task")
quick_query("SELECT ?t WHERE { ?t type task:Task }")  # Returns 0 results!
```

### ✅ CORRECT (Creates Instance)
```python
add_knowledge("task:Task(task:Task1)")
quick_query("SELECT ?t WHERE { ?t type task:Task }")  # Returns Task1 ✅
```

---

## Instance Creation

| What You Want | Correct Syntax | Example |
|---------------|---------------|---------|
| Create one instance | `Class(instance)` | `Person(Alice)` |
| Create multiple | `Class(inst1)` on separate lines | `Person(Alice)`<br>`Person(Bob)` |
| With namespace | `ns:Class(ns:instance)` | `task:Task(task:Task1)` |

**REQL Query Pattern:**
```sql
SELECT ?x WHERE { ?x type ClassName }
```

---

## Class Hierarchy

| What You Want | Correct Syntax | Example |
|---------------|---------------|---------|
| Create subclass | `SubClass is_subclass_of SuperClass` | `Dog is_subclass_of Animal` |
| Alternative | `SubClass is_a SuperClass` | `Cat is_a Animal` |

**Note:** `is_a` means "is a subclass of", NOT "is an instance of"!

---

## Property Assertions

| Property Type | Correct Syntax | Example |
|--------------|---------------|---------|
| Object property | `predicate(subject, object)` | `likes(Alice, Bob)` |
| String property | `predicate(subject, "value")` | `name(Alice, "Alice Smith")` |
| Number property | `predicate(subject, 25)` | `age(Alice, 25)` |

**REQL Query Pattern:**
```sql
SELECT ?subject ?value WHERE { ?subject propertyName ?value }
```

---

## Complete Example

### Assertion (DL Syntax)
```python
ontology = """
# 1. Define class hierarchy
task:Task is_subclass_of owl:Thing
task:Step is_subclass_of owl:Thing

# 2. Create instances (use parentheses!)
task:Task(task:RefactorDatabase)
task:Step(task:Step1)
task:Step(task:Step2)

# 3. Add properties
priority(task:RefactorDatabase, "high")
phase(task:Step1, "planning")
estimatedHours(task:Step1, 5)
phase(task:Step2, "execution")
estimatedHours(task:Step2, 10)
"""

add_knowledge(source=ontology, type="ontology")
```

### Queries (REQL Syntax)
```python
# Query 1: Find all tasks
quick_query("SELECT ?t WHERE { ?t type task:Task }", type="reql")
# Result: task:RefactorDatabase

# Query 2: Find task priority
quick_query("SELECT ?t ?p WHERE { ?t priority ?p }", type="reql")
# Result: task:RefactorDatabase, "high"

# Query 3: Find steps in planning phase
quick_query("SELECT ?s WHERE { ?s phase \"planning\" }", type="reql")
# Result: task:Step1

# Query 4: Count steps
quick_query("SELECT (COUNT(?s) AS ?count) WHERE { ?s type task:Step }", type="reql")
# Result: 2

# Query 5: Sum hours by phase
quick_query("""
    SELECT ?phase (SUM(?hours) AS ?total)
    WHERE { ?s phase ?phase . ?s estimatedHours ?hours }
    GROUP BY ?phase
""", type="reql")
# Result: planning=5, execution=10
```

---

## Assertion → Query Mapping

| You ASSERT (DL) | Query With (REQL) | Returns |
|-----------------|-------------------|---------|
| `Person(Alice)` | `SELECT ?x WHERE { ?x type Person }` | Alice |
| `age(Alice, 25)` | `SELECT ?x ?a WHERE { ?x age ?a }` | Alice, 25 |
| `age(Alice, 25)` | `SELECT ?x WHERE { ?x age 25 }` | Alice |
| `likes(Alice, Bob)` | `SELECT ?x ?y WHERE { ?x likes ?y }` | Alice, Bob |
| `priority(T1, "high")` | `SELECT ?t ?p WHERE { ?t priority ?p }` | T1, "high" |

---

## REQL Aggregation

```sql
-- Count
SELECT (COUNT(?x) AS ?count) WHERE { ?x type ClassName }

-- Count with GROUP BY
SELECT ?category (COUNT(?item) AS ?count)
WHERE { ?item category ?category }
GROUP BY ?category

-- Sum
SELECT (SUM(?hours) AS ?total) WHERE { ?task estimatedHours ?hours }

-- Average
SELECT (AVG(?value) AS ?average) WHERE { ?x metric ?value }

-- With HAVING
SELECT ?cat (COUNT(?x) AS ?cnt)
WHERE { ?x category ?cat }
GROUP BY ?cat
HAVING (?cnt > 5)
```

---

## Namespaces

If you use namespaces, **be consistent**:

```python
# ✅ CORRECT: Namespaces match
add_knowledge("task:Task(task:Task1)")
quick_query("SELECT ?t WHERE { ?t type task:Task }")

# ❌ WRONG: Namespace mismatch
add_knowledge("task:Task(task:Task1)")
quick_query("SELECT ?t WHERE { ?t type Task }")  # Missing task: prefix!
```

---

## Python Code Analysis

```python
# Analyze a Python file
add_knowledge(source="path/to/file.py", type="python")

# Query for classes
quick_query("SELECT ?class WHERE { ?class type py:Class }", type="reql")

# Query for methods
quick_query("SELECT ?method WHERE { ?method type py:Method }", type="reql")

# Find methods in a class
quick_query("""
    SELECT ?method WHERE { ?method definedIn 'MyClass' }
""", type="reql")
```

---

## Common Errors

### Error: "Query returns 0 results"

**Check:**
1. ✅ Did you use `Class(instance)` not `instance is_a Class`?
2. ✅ Do namespaces match in assertion and query?
3. ✅ Is the property name exactly the same (case-sensitive)?
4. ✅ Did you quote string values in assertions?

**Debug:**
```python
# See what's actually stored
all_facts = r.get_all_facts()
print(all_facts.to_pandas())
```

### Error: "Parse error"

**Check:**
1. ✅ String values must be quoted: `"value"` not `value`
2. ✅ Predicate syntax: `predicate(subject, object)` not `subject predicate object`
3. ✅ Parentheses for instances: `Class(inst)` not `Class inst`

---

## Need Help?

1. **Read the full guide:** `guide://logical-thinking/usage`
2. **Grammar reference:** `grammar://reter/dl-reql`
3. **Python analysis:** `python://reter/analysis`

---

## Remember

- **`Class(instance)`** = Creates an instance ✅
- **`instance is_a Class`** = Creates a subclass ❌
- **Namespaces must match** between assertions and queries
- **Property names are case-sensitive**
- **String values need quotes**: `"value"`

---

**Last Updated:** 2025-11-10
