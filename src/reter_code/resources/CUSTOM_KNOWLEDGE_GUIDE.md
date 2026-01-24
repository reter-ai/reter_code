# Creating and Exploring Custom Knowledge in RETER

**A Practical Guide to Building Your Own Ontologies**

---

## Table of Contents

1. [Quick Start: Your First Knowledge Base](#quick-start)
2. [Understanding DL Syntax](#understanding-dl-syntax)
3. [Creating Instances](#creating-instances)
4. [Adding Properties](#adding-properties)
5. [Building Class Hierarchies](#building-class-hierarchies)
6. [Querying Your Knowledge](#querying-your-knowledge)
7. [Exploring What's Stored](#exploring-whats-stored)
8. [Common Patterns](#common-patterns)
9. [Debugging Empty Results](#debugging-empty-results)
10. [Complete Examples](#complete-examples)

---

## Quick Start

### 1. Add Your First Knowledge

```python
from logical_thinking_server import add_knowledge, quick_query

# Step 1: Define your domain
ontology = """
# Define classes
Project is_subclass_of owl:Thing
Task is_subclass_of owl:Thing
Person is_subclass_of owl:Thing

# Create instances (use parentheses!)
Project(proj:WebsiteRedesign)
Task(task:DesignMockups)
Task(task:ImplementFrontend)
Person(person:Alice)

# Add properties
assignedTo(task:DesignMockups, person:Alice)
priority(task:DesignMockups, "high")
estimatedHours(task:DesignMockups, 8)
status(task:DesignMockups, "in-progress")
"""

result = add_knowledge(source=ontology, type="ontology")
print(f"Added {result['items_added']} facts")
```

### 2. Query Your Knowledge

```python
# Find all tasks
result = quick_query(
    query="SELECT ?task WHERE { ?task type Task }",
    type="reql"
)
print(f"Found {result['count']} tasks")
print(result['results'])

# Find who's assigned to each task
result = quick_query(
    query="SELECT ?task ?person WHERE { ?task assignedTo ?person }",
    type="reql"
)
print(result['results'])
```

### 3. Explore What's Stored

```python
# Get all facts to see what's in the knowledge base
from reter import Reter
r = Reter(variant='ai')

# Load your ontology
r.load_ontology(ontology)

# See everything
all_facts = r.get_all_facts()
print(f"Total facts: {all_facts.num_rows}")
print(all_facts.to_pandas())
```

---

## Understanding DL Syntax

### The Critical Distinction

**⚠️ Most Important Concept:**

| Syntax | Meaning | Creates | Use For |
|--------|---------|---------|---------|
| `Class(instance)` | "instance IS AN instance OF Class" | Instance | Actual data |
| `SubClass is_a SuperClass` | "SubClass IS A subclass OF SuperClass" | Subclass | Hierarchies |

### Why This Matters

```python
# ❌ WRONG - This creates a CLASS, not an instance!
add_knowledge("Alice is_a Person")

# What it means to RETER:
# "Alice is a SUBCLASS of Person"
# Creates: subsumption relationship

# Why your query fails:
quick_query("SELECT ?p WHERE { ?p type Person }")
# Looking for INSTANCES of Person
# But Alice is stored as a SUBCLASS, not an instance
# Result: 0 rows ❌

# ✅ CORRECT - This creates an INSTANCE
add_knowledge("Person(Alice)")

# What it means to RETER:
# "Alice is an INSTANCE of Person"
# Creates: instance_of relationship

# Now your query works:
quick_query("SELECT ?p WHERE { ?p type Person }")
# Result: Alice ✅
```

---

## Creating Instances

### Basic Instance Creation

```python
# Single instance
Person(Alice)

# Multiple instances
Person(Alice)
Person(Bob)
Person(Charlie)

# With namespaces
person:Person(person:Alice)
person:Person(person:Bob)

# Different classes
Task(task:Task1)
Project(proj:WebApp)
```

### Verification

After adding instances, verify they were created:

```python
# Add instances
add_knowledge("""
Person(Alice)
Person(Bob)
""")

# Verify with query
result = quick_query("SELECT ?p WHERE { ?p type Person }")
# Should return: [["Alice"], ["Bob"]]
```

---

## Adding Properties

### Property Syntax

```python
# Object properties (linking entities)
predicate(subject, object)

# Data properties (values)
predicate(subject, "string value")
predicate(subject, numeric_value)
```

### Examples

```python
ontology = """
# Create entities first
Person(Alice)
Person(Bob)
Project(proj:WebApp)

# Object properties (entity relationships)
worksOn(Alice, proj:WebApp)
managedBy(proj:WebApp, Bob)
reportsTo(Alice, Bob)

# Data properties (values)
age(Alice, 30)
name(Alice, "Alice Johnson")
salary(Alice, 75000)
email(Alice, "alice@example.com")
status(proj:WebApp, "active")
"""
```

### Query Properties

```python
# Find all people and what they work on
quick_query("SELECT ?person ?project WHERE { ?person worksOn ?project }")

# Find people with specific age
quick_query("SELECT ?person WHERE { ?person age 30 }")

# Find all properties of Alice
quick_query("SELECT ?property ?value WHERE { Alice ?property ?value }")
```

---

## Building Class Hierarchies

### Subclass Relationships

```python
# Use is_subclass_of or is_a for classes
ontology = """
# Top-level classes
Animal is_subclass_of owl:Thing
Vehicle is_subclass_of owl:Thing

# Subclasses
Mammal is_subclass_of Animal
Bird is_subclass_of Animal
Dog is_subclass_of Mammal
Cat is_subclass_of Mammal

Car is_subclass_of Vehicle
Truck is_subclass_of Vehicle

# Now create instances (not subclasses!)
Dog(Fido)
Cat(Whiskers)
Car(MyCar)
"""
```

### How Inference Works

```python
# Add hierarchy
add_knowledge("""
Mammal is_subclass_of Animal
Dog is_subclass_of Mammal
Dog(Fido)
""")

# Query for animals
result = quick_query("SELECT ?x WHERE { ?x type Animal }")
# Returns: Fido
# Why? Because:
#   Fido is a Dog
#   Dog is a Mammal
#   Mammal is an Animal
#   Therefore, Fido is an Animal (transitive inference)
```

---

## Querying Your Knowledge

### Basic REQL Patterns

```python
# Pattern 1: Find all instances of a type
SELECT ?x WHERE { ?x type ClassName }

# Pattern 2: Find property values
SELECT ?subject ?value WHERE { ?subject propertyName ?value }

# Pattern 3: Filter by specific value
SELECT ?subject WHERE { ?subject propertyName "specificValue" }

# Pattern 4: Multiple conditions (use . between patterns)
SELECT ?x ?y WHERE {
    ?x type Person .
    ?x worksOn ?y .
    ?y type Project
}
```

### Advanced REQL Patterns

```python
# Count instances
SELECT (COUNT(?x) AS ?count) WHERE { ?x type Task }

# Count with grouping
SELECT ?status (COUNT(?task) AS ?count)
WHERE { ?task status ?status }
GROUP BY ?status

# Sum numeric values
SELECT (SUM(?hours) AS ?total)
WHERE { ?task estimatedHours ?hours }

# Average
SELECT (AVG(?hours) AS ?average)
WHERE { ?task estimatedHours ?hours }

# Filter groups
SELECT ?person (COUNT(?task) AS ?taskCount)
WHERE { ?task assignedTo ?person }
GROUP BY ?person
HAVING (?taskCount > 5)

# Order results
SELECT ?task ?priority
WHERE { ?task priority ?priority }
ORDER BY DESC(?priority)

# Limit results
SELECT ?task WHERE { ?task type Task }
LIMIT 10
```

### Complex Queries

```python
# Find people working on high-priority tasks
SELECT ?person ?task WHERE {
    ?task type Task .
    ?task priority "high" .
    ?task assignedTo ?person
}

# Find tasks with no assignee
SELECT ?task WHERE {
    ?task type Task .
    FILTER NOT EXISTS { ?task assignedTo ?person }
}

# Find overloaded people (>40 hours)
SELECT ?person (SUM(?hours) AS ?total)
WHERE {
    ?task assignedTo ?person .
    ?task estimatedHours ?hours
}
GROUP BY ?person
HAVING (?total > 40)
```

---

## Exploring What's Stored

### Method 1: Get All Facts (Low-Level)

```python
from reter import Reter

r = Reter(variant='ai')
r.load_ontology(your_ontology)

# Get everything as Arrow table
all_facts = r.get_all_facts()
print(f"Total facts: {all_facts.num_rows}")

# Convert to pandas for easier viewing
df = all_facts.to_pandas()
print(df)

# Filter by fact type
instances = df[df['type'] == 'instance_of']
print("Instances:")
print(instances[['individual', 'concept']])

properties = df[df['type'] == 'data_assertion']
print("\nProperties:")
print(properties[['subject', 'property', 'value']])

subclasses = df[df['type'] == 'subsumption']
print("\nClass hierarchy:")
print(subclasses[['sub', 'sup']])
```

### Method 2: Query-Based Exploration

```python
# List all classes
result = quick_query("""
    SELECT DISTINCT ?class WHERE { ?x type ?class }
""")
print("Classes:", result['results'])

# List all properties used
# (This requires iterating through facts - use Method 1)

# Count instances per class
result = quick_query("""
    SELECT ?class (COUNT(?instance) AS ?count)
    WHERE { ?instance type ?class }
    GROUP BY ?class
""")
print("Instance counts:", result['results'])

# Sample instances of each class
result = quick_query("""
    SELECT ?class ?instance
    WHERE { ?instance type ?class }
    LIMIT 100
""")
```

### Method 3: Systematic Exploration

```python
def explore_knowledge_base():
    """Comprehensive KB exploration"""
    r = Reter(variant='ai')
    r.load_ontology(your_ontology)

    all_facts = r.get_all_facts()
    df = all_facts.to_pandas()

    print("="*80)
    print("KNOWLEDGE BASE SUMMARY")
    print("="*80)

    # 1. Total facts
    print(f"\nTotal facts: {len(df)}")

    # 2. Fact types
    print("\nFact types:")
    print(df['type'].value_counts())

    # 3. Classes and instances
    instances = df[df['type'] == 'instance_of']
    if len(instances) > 0:
        print(f"\nInstances by class:")
        print(instances.groupby('concept')['individual'].count())

        print(f"\nSample instances:")
        print(instances[['concept', 'individual']].head(10))

    # 4. Properties
    data_assertions = df[df['type'] == 'data_assertion']
    if len(data_assertions) > 0:
        print(f"\nProperties used:")
        print(data_assertions['property'].value_counts())

        print(f"\nSample property assertions:")
        print(data_assertions[['subject', 'property', 'value']].head(10))

    # 5. Class hierarchy
    subsumptions = df[df['type'] == 'subsumption']
    if len(subsumptions) > 0:
        print(f"\nClass hierarchy:")
        print(subsumptions[['sub', 'sup']].head(10))

    return df
```

---

## Common Patterns

### Pattern 1: Project Management

```python
ontology = """
# Classes
Project is_subclass_of owl:Thing
Task is_subclass_of owl:Thing
Person is_subclass_of owl:Thing

# Instances
Project(proj:WebsiteRedesign)
Task(task:Design)
Task(task:Development)
Task(task:Testing)
Person(person:Alice)
Person(person:Bob)

# Structure
partOf(task:Design, proj:WebsiteRedesign)
partOf(task:Development, proj:WebsiteRedesign)
partOf(task:Testing, proj:WebsiteRedesign)

# Assignments
assignedTo(task:Design, person:Alice)
assignedTo(task:Development, person:Bob)
assignedTo(task:Testing, person:Alice)

# Properties
status(task:Design, "completed")
status(task:Development, "in-progress")
status(task:Testing, "not-started")
estimatedHours(task:Design, 20)
estimatedHours(task:Development, 40)
estimatedHours(task:Testing, 16)
"""

# Useful queries:
# - Find Alice's tasks
# - Sum hours per person
# - Find incomplete tasks
# - List tasks by project
```

### Pattern 2: Knowledge Graph

```python
ontology = """
# Entities
Person(Alice)
Person(Bob)
Person(Charlie)
Company(TechCorp)
City(NewYork)
City(SanFrancisco)

# Relationships
worksAt(Alice, TechCorp)
worksAt(Bob, TechCorp)
livesIn(Alice, NewYork)
livesIn(Bob, SanFrancisco)
knows(Alice, Bob)
knows(Bob, Charlie)

# Properties
age(Alice, 30)
age(Bob, 35)
founded(TechCorp, 2010)
population(NewYork, 8000000)
"""

# Useful queries:
# - Who works at TechCorp?
# - Who lives in NewYork?
# - Who knows who?
# - Social network analysis
```

### Pattern 3: Domain Model

```python
ontology = """
# Domain hierarchy
Product is_subclass_of owl:Thing
Category is_subclass_of owl:Thing
Customer is_subclass_of owl:Thing
Order is_subclass_of owl:Thing

# Product taxonomy
Electronics is_subclass_of Product
Clothing is_subclass_of Product
Laptop is_subclass_of Electronics
Phone is_subclass_of Electronics

# Instances
Laptop(prod:MacBookPro)
Phone(prod:iPhone15)
Customer(cust:John)
Order(order:ORD001)

# Relationships
belongsTo(prod:MacBookPro, Electronics)
purchasedBy(order:ORD001, cust:John)
contains(order:ORD001, prod:MacBookPro)

# Properties
price(prod:MacBookPro, 2499)
quantity(order:ORD001, 1)
date(order:ORD001, "2024-01-15")
"""
```

---

## Debugging Empty Results

### Step-by-Step Debugging

```python
# Step 1: Verify what was added
result = add_knowledge(your_ontology)
print(f"Items added: {result['items_added']}")  # Should be > 0

# Step 2: Check if it's stored
from reter import Reter
r = Reter(variant='ai')
r.load_ontology(your_ontology)

all_facts = r.get_all_facts()
print(f"Total facts: {all_facts.num_rows}")  # Should match items_added

# Step 3: Look at what's actually stored
df = all_facts.to_pandas()
print(df)  # Examine the structure

# Step 4: Check fact types
print(df['type'].value_counts())
# If you see 'subsumption' but expected 'instance_of', you used is_a incorrectly!

# Step 5: Check for instance_of facts
instances = df[df['type'] == 'instance_of']
print(f"Instances: {len(instances)}")
print(instances[['concept', 'individual']])

# Step 6: Check property names
data_facts = df[df['type'] == 'data_assertion']
print("Properties:")
print(data_facts['property'].unique())
# Are these the names you're querying for?

# Step 7: Check namespaces
print("Subjects:", df['subject'].unique()[:10])
print("Concepts:", df['concept'].unique()[:10])
# Do namespaces match your query?
```

### Common Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Used `is_a` for instance | 0 results, see 'subsumption' | Use `Class(instance)` |
| Namespace mismatch | 0 results, facts exist | Match namespaces in query |
| Wrong property name | 0 results for property | Check `df['property'].unique()` |
| Case sensitivity | 0 results | Property names are case-sensitive |
| Forgot quotes | Parse error | String values need `"quotes"` |

---

## Complete Examples

### Example 1: Software Architecture

```python
ontology = """
# Architecture components
Component is_subclass_of owl:Thing
Service is_subclass_of Component
Database is_subclass_of Component
API is_subclass_of Component

# Instances
Service(arch:UserService)
Service(arch:OrderService)
Database(arch:PostgreSQL)
Database(arch:Redis)
API(arch:RestAPI)
API(arch:GraphQLAPI)

# Dependencies
dependsOn(arch:UserService, arch:PostgreSQL)
dependsOn(arch:OrderService, arch:PostgreSQL)
dependsOn(arch:UserService, arch:Redis)
usesAPI(arch:OrderService, arch:RestAPI)

# Properties
language(arch:UserService, "Python")
language(arch:OrderService, "Go")
port(arch:UserService, 8001)
port(arch:OrderService, 8002)
version(arch:PostgreSQL, "15.2")
"""

# Query: Find all Python services
result = quick_query('''
    SELECT ?service WHERE {
        ?service type Service .
        ?service language "Python"
    }
''')

# Query: Find service dependencies
result = quick_query('''
    SELECT ?service ?dep WHERE {
        ?service type Service .
        ?service dependsOn ?dep
    }
''')

# Query: Count services per language
result = quick_query('''
    SELECT ?lang (COUNT(?service) AS ?count)
    WHERE {
        ?service type Service .
        ?service language ?lang
    }
    GROUP BY ?lang
''')
```

### Example 2: Refactoring Plan

```python
ontology = """
# Refactoring taxonomy
Refactoring is_subclass_of owl:Thing
CodeSmell is_subclass_of owl:Thing
Task is_subclass_of owl:Thing

ExtractClass is_subclass_of Refactoring
ExtractMethod is_subclass_of Refactoring
GodObject is_subclass_of CodeSmell

# Detected issues
CodeSmell(smell:FSACompilerGodObject)
smellType(smell:FSACompilerGodObject, GodObject)
affectsClass(smell:FSACompilerGodObject, "FSACompiler")
methodCount(smell:FSACompilerGodObject, 41)
severity(smell:FSACompilerGodObject, "high")

# Refactoring tasks
Task(task:ExtractParsing)
Task(task:ExtractValidation)
Task(task:ExtractGeneration)

addresses(task:ExtractParsing, smell:FSACompilerGodObject)
addresses(task:ExtractValidation, smell:FSACompilerGodObject)
addresses(task:ExtractGeneration, smell:FSACompilerGodObject)

refactoringType(task:ExtractParsing, ExtractClass)
estimatedHours(task:ExtractParsing, 8)
priority(task:ExtractParsing, "high")
phase(task:ExtractParsing, "decomposition")

refactoringType(task:ExtractValidation, ExtractClass)
estimatedHours(task:ExtractValidation, 6)
priority(task:ExtractValidation, "high")
phase(task:ExtractValidation, "decomposition")
"""

# Query: Find high-severity smells
result = quick_query('''
    SELECT ?smell ?class ?count WHERE {
        ?smell type CodeSmell .
        ?smell severity "high" .
        ?smell affectsClass ?class .
        ?smell methodCount ?count
    }
''')

# Query: Sum hours per phase
result = quick_query('''
    SELECT ?phase (SUM(?hours) AS ?total)
    WHERE {
        ?task type Task .
        ?task phase ?phase .
        ?task estimatedHours ?hours
    }
    GROUP BY ?phase
''')

# Query: Find tasks addressing each smell
result = quick_query('''
    SELECT ?smell (COUNT(?task) AS ?taskCount)
    WHERE {
        ?task addresses ?smell
    }
    GROUP BY ?smell
''')
```

### Example 3: Research Knowledge Base

```python
ontology = """
# Research entities
Paper is_subclass_of owl:Thing
Author is_subclass_of owl:Thing
Topic is_subclass_of owl:Thing

# Papers
Paper(paper:NeuralNets2023)
Paper(paper:TransformerArch2024)
Paper(paper:AttentionMech2023)

# Authors
Author(author:Smith)
Author(author:Johnson)
Author(author:Chen)

# Topics
Topic(topic:DeepLearning)
Topic(topic:NLP)
Topic(topic:ComputerVision)

# Authorship
writtenBy(paper:NeuralNets2023, author:Smith)
writtenBy(paper:NeuralNets2023, author:Johnson)
writtenBy(paper:TransformerArch2024, author:Johnson)
writtenBy(paper:TransformerArch2024, author:Chen)

# Topics
hasTopic(paper:NeuralNets2023, topic:DeepLearning)
hasTopic(paper:TransformerArch2024, topic:NLP)
hasTopic(paper:AttentionMech2023, topic:NLP)

# Citations
cites(paper:TransformerArch2024, paper:AttentionMech2023)

# Properties
year(paper:NeuralNets2023, 2023)
year(paper:TransformerArch2024, 2024)
venue(paper:NeuralNets2023, "NeurIPS")
venue(paper:TransformerArch2024, "ACL")
"""

# Query: Find co-authors (people who wrote together)
result = quick_query('''
    SELECT DISTINCT ?author1 ?author2 WHERE {
        ?paper writtenBy ?author1 .
        ?paper writtenBy ?author2 .
        FILTER(?author1 != ?author2)
    }
''')

# Query: Most prolific authors
result = quick_query('''
    SELECT ?author (COUNT(?paper) AS ?count)
    WHERE { ?paper writtenBy ?author }
    GROUP BY ?author
    ORDER BY DESC(?count)
''')

# Query: Papers per topic
result = quick_query('''
    SELECT ?topic (COUNT(?paper) AS ?count)
    WHERE { ?paper hasTopic ?topic }
    GROUP BY ?topic
''')
```

---

## Best Practices

### 1. Use Meaningful Namespaces

```python
# ✅ Good: Clear namespaces
person:Person(person:Alice)
project:Project(project:WebApp)

# ❌ Avoid: No namespaces (can collide)
Person(Alice)
Project(WebApp)
```

### 2. Be Consistent

```python
# ✅ Good: Consistent naming
estimatedHours, actualHours, remainingHours

# ❌ Avoid: Inconsistent
estimated_hours, actualHrs, hoursRemaining
```

### 3. Use Descriptive Names

```python
# ✅ Good: Clear meaning
assignedTo(task:T1, person:Alice)
estimatedCompletionDate(task:T1, "2024-12-31")

# ❌ Avoid: Cryptic
a2(t1, p1)
ecd(t1, "2024-12-31")
```

### 4. Document Your Ontology

```python
ontology = """
# ==========================================
# Project Management Ontology
# Version: 1.0
# Created: 2024-01-15
# ==========================================

# Classes
# -------
# Project: A project with tasks
# Task: An atomic unit of work
# Person: A team member

Project is_subclass_of owl:Thing
Task is_subclass_of owl:Thing
Person is_subclass_of owl:Thing

# Properties
# ----------
# assignedTo: Links task to person
# estimatedHours: Time estimate (numeric)
# priority: high/medium/low (string)

...
"""
```

### 5. Validate As You Go

```python
# After adding knowledge, verify
result = add_knowledge(ontology)
assert result['items_added'] > 0, "No items added!"

# Test a query immediately
test_result = quick_query("SELECT ?x WHERE { ?x type Task }")
assert test_result['count'] > 0, "No tasks found!"
```

---

## Next Steps

1. **Start Small:** Begin with 5-10 entities
2. **Add Gradually:** Build up complexity incrementally
3. **Query Often:** Verify each addition with queries
4. **Explore Regularly:** Use `get_all_facts()` to see what's stored
5. **Refine:** Adjust your ontology based on what you learn

---

## Resources

- **Full Usage Guide:** `guide://logical-thinking/usage`
- **Grammar Reference:** `grammar://reter/dl-reql`
- **Python Analysis:** `python://reter/analysis`
- **Quick Reference:** `SYNTAX_QUICK_REFERENCE.md`

---

**Last Updated:** 2025-11-10
**Status:** Complete
