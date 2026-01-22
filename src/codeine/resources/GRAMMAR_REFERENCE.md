# RETER Grammar Reference - AI Lexer Syntax

This document provides the **exact** Description Logic (DL) and REQL syntax supported by the RETER MCP Server.

**IMPORTANT**: The RETER MCP Server uses the **AI-friendly lexer** which provides natural language-style syntax with case-insensitive keywords.

## Key Features

- **Case-insensitive keywords**: Write `is_a`, `IS_A`, `Is_A` or `isa` - all work!
- **Manchester syntax support** ✨ **NEW**: Use standard OWL Manchester syntax (`hasChild some Person`) alongside prefix syntax (`some hasChild that_is Person`)
- **Natural language operators**: `is_subclass_of`, `is_equivalent_to`, `has`, `some`, `all`, `and`, `or`, `not`
- **Flexible alternatives**: Most operators have multiple forms (e.g., `is_a`, `isa`, `extends`)
- **Standard ASCII**: Uses regular parentheses `()` and commas `,`
- **Cardinality support**: Both symbols (`>=`, `<=`, `=`) and keywords (`at_least`, `at_most`, `exactly`)
- **Programming identifiers**: Supports most common patterns like `MyClass`, `my_variable`, `Person123`
- **Backtick escaping**: Use backticks for complex identifiers: `` `complex-name` ``, `` `List<String>` ``

---

## Description Logic (DL) Grammar - AI Lexer

### 1. Class Subsumption

Multiple equivalent forms (case-insensitive):

```
ClassName1 is_sub_concept_of ClassName2
ClassName1 is_subclass_of ClassName2
ClassName1 is_a ClassName2
ClassName1 isa ClassName2
ClassName1 extends ClassName2
```

**Examples:**
```
Dog is_a Mammal
dog ISA mammal
Cat extends Animal
Person is_subclass_of Thing
```

### 2. Class Equivalence

Multiple forms:
```
ClassName1 is_same_concept_as ClassName2
ClassName1 is_equivalent_to ClassName2
ClassName1 equals ClassName2
ClassName1 same_as ClassName2
```

**Examples:**
```
Mother is_equivalent_to (Female and Parent)
Adult equals (Person and (age >= 18))
```

Multiple equivalences using braces:
```
equals {ClassName1, ClassName2, ClassName3}
```

### 3. Class Disjointness

```
not equals (ClassName1, ClassName2, ClassName3)
```

**Example:**
```
not equals (Dog, Cat, Bird)
```

### 4. Class Assertions (Individuals)

Standard ASCII parentheses and comma:

```
ClassName(IndividualName)
```

**Examples:**
```
Dog(Fido)
Cat(Whiskers)
Person(Alice)
```

### 5. Property Assertions

**Object properties** (relating two individuals):
```
propertyName(subject, object)
```

**Data properties** (relating individual to value):
```
propertyName(subject, value)
```

**Examples:**
```
worksFor(Alice, Acme)
age(Alice, 25)
knows(Alice, Bob)
hasChild(Alice, Bob)
```

### 6. Class Expressions

#### Intersection (AND)
Multiple forms:
```
Class1 intersection_with Class2
Class1 and Class2
Class1 with Class2
```

#### Union (OR)
```
Class1 union_with Class2
Class1 or Class2
```

#### Complement (NOT)
```
complement_of Class
not Class
except Class
```

#### Complex expressions with parentheses
```
(Person and not Teacher)
Student or (Person and not Teacher)
```

### 7. Property Restrictions

RETER supports **two syntaxes** for property restrictions:
1. **Prefix syntax** (original): `some propertyName that_is ClassName`
2. **Manchester syntax** (NEW): `propertyName some ClassName`

Both syntaxes are fully equivalent and can be mixed freely.

#### Existential Quantification (SomeValuesFrom)

**Prefix forms:**
```
some propertyName that_is ClassName
exists propertyName that ClassName
has propertyName where ClassName
```

**Manchester form (NEW):**
```
propertyName some ClassName
propertyName exists ClassName
propertyName has ClassName
```

**Examples:**
```
# Prefix syntax
some hasChild that_is Thing
exists worksFor that Company
has hasChild where Person

# Manchester syntax (NEW)
hasChild some Thing
worksFor exists Company
hasMethod some py:Method
```

Used in subsumptions:
```
# Prefix
(some worksFor that_is Thing) is_subclass_of Employee

# Manchester (NEW)
(worksFor some Thing) is_subclass_of Employee
```

#### Universal Quantification (AllValuesFrom)

**Prefix forms:**
```
all propertyName that_is ClassName
only propertyName that ClassName
must propertyName where ClassName
```

**Manchester form (NEW):**
```
propertyName all ClassName
propertyName only ClassName
propertyName must ClassName
```

**Examples:**
```
# Prefix syntax
Thing is_subclass_of (all hasChild that_is Person)
Thing is_subclass_of (only worksFor that Company)

# Manchester syntax (NEW)
Thing is_subclass_of (hasChild all Person)
Thing is_subclass_of (worksFor only Company)
```

#### Self Restriction
```
some propertyName that_is self
propertyName some self
```

### 8. Cardinality Restrictions

RETER supports **two syntaxes** for cardinality restrictions:

#### Prefix Syntax (Original)

Cardinality with **symbols**:
```
= n propertyName that_is ClassName
>= n propertyName that_is ClassName
<= n propertyName that_is ClassName
> n propertyName that_is ClassName
< n propertyName that_is ClassName
!= n propertyName that_is ClassName
```

Cardinality with **keywords** (case-insensitive):
```
exactly n propertyName that_is ClassName
at_least n propertyName that_is ClassName
at_most n propertyName that_is ClassName
greater_than n propertyName that_is ClassName
less_than n propertyName that_is ClassName
not_equal n propertyName that_is ClassName
```

#### Manchester Syntax (NEW)

```
propertyName = n ClassName
propertyName >= n ClassName
propertyName <= n ClassName
propertyName > n ClassName
propertyName < n ClassName
propertyName != n ClassName
```

**Examples:**
```
# Prefix syntax
Parent is_subclass_of (>= 1 hasChild that_is Thing)
Parent is_subclass_of (at_least 1 hasChild that Thing)
ExactlyTwoChildren equals (= 2 hasChild that_is Thing)
ExactlyTwoChildren EQUALS (EXACTLY 2 hasChild THAT_IS Thing)

# Manchester syntax (NEW)
Parent is_subclass_of (hasChild >= 1 Thing)
Parent is_subclass_of (hasChild >= 1 Thing)
ExactlyTwoChildren equals (hasChild = 2 Thing)
LargeFamily equals (hasChild >= 4 Person)
```

### 9. Data Property Restrictions

With facets (constraints on values):
```
all propertyName >= value
some propertyName >= value
all propertyName <= value
some propertyName <= value
```

**Examples:**
```
Adult is_equivalent_to (some age >= 18)
Senior equals (exists age >= 65)
```

### 10. Role (Property) Subsumption

```
propertyName1 is_sub_role_of propertyName2
propertyName1 is_subproperty_of propertyName2
```

**Example:**
```
parentOf is_subproperty_of ancestorOf
```

### 11. Role Equivalence

```
propertyName1 is_same_role_as propertyName2
propertyName1 is_same_property_as propertyName2
```

**Example (symmetric property):**
```
knows is_same_role_as knows inverse
```

### 12. Property Chains

```
property1 composition_with property2 is_sub_role_of property3
property1 composed_with property2 is_subproperty_of property3
```

**Examples:**
```
parentOf composition_with parentOf is_sub_role_of grandparentOf
ancestorOf composed_with ancestorOf is_subproperty_of ancestorOf
```

### 13. Inverse Properties

Use the `inverse` keyword as a postfix operator:

```
propertyName inverse
```

**Example:**
```
hasChild is_same_role_as hasParent inverse
```

### 14. Individual Equality and Inequality

#### Same individuals
```
Individual1 = Individual2
```

Multiple same:
```
= {Individual1, Individual2, Individual3}
```

**Examples:**
```
BobSmith = Robert
= {USA, UnitedStates, America}
```

#### Different individuals
```
Individual1 != Individual2
```

Multiple different:
```
!= {Individual1, Individual2, Individual3}
```

### 15. Instance Sets

Create anonymous classes from individuals:
```
{Individual1, Individual2, Individual3}
```

**Example:**
```
FamousPeople equals {Einstein, Newton, Darwin}
```

### 16. Unnamed Instances

Create anonymous individuals within instance sets (cannot be used as standalone statements):

```
{[ClassName], namedIndividual}
{[ClassName1], [ClassName2], Alice}
```

**Example:**
```
FamousPeople equals {[Person], [Person], Einstein}
```

**Note**: Anonymous instances `[ClassName]` can only appear inside braces `{}` as part of an instance set. They cannot be used standalone.

### 17. SWRL Rules

SWRL (Semantic Web Rule Language) rules allow you to define inference rules using if-then logic.

#### Basic Syntax

```
if antecedent then consequent
when antecedent implies consequent
```

#### Variables

SWRL rules use variables to match patterns. Variables can be written two ways:
- **With `object` keyword**: `object x`, `object y` (recommended)
- **With `?` prefix**: `?x`, `?y` (also supported)

#### Multiple Conditions with `also`

**IMPORTANT**: Use the `also` keyword to combine multiple conditions in the antecedent (NOT `and`):

```
if condition1 also condition2 also condition3 then consequent
```

**Note**: The keyword is `also`, not `and`. Using `and` will cause parse errors.

#### Single Condition Rules

For rules with only one condition, omit `also`:

```
if Person(object x) then Human(object x)
```

#### Examples

**Simple inference:**
```
if Person(object x) then Human(object x)
```

**Transitive property:**
```
if hasAncestor(object x, object y) also hasAncestor(object y, object z) then hasAncestor(object x, object z)
```

**Complex rule with multiple conditions:**
```
if Person(object x) also hasParent(object x, object y) also Female(object y) then hasMother(object x, object y)
```

**Data property reasoning:**
```
if Person(object x) also age(object x, var a) also greaterThan(var a, 18) then Adult(object x)
```

**Negation in rules:**
```
if Function(object f) also not hasDocstring(object f, object d) then undocumented(object f, true)
```

#### Variable Types

- **Object variables**: `object x` - matches individuals/instances
- **Data variables**: `var x` - matches literal values (numbers, strings, etc.)

#### Common Patterns

**1. Inheritance transitivity:**
```
if inheritsFrom(object x, object y) also inheritsFrom(object y, object z) then inheritsFrom(object x, object z)
```

**2. Property chains:**
```
if hasParent(object x, object y) also hasSibling(object y, object z) then hasUncleOrAunt(object x, object z)
```

**3. Decorator-based inference:**
```
if Method(object m) also hasDecorator(object m, "property") then isProperty(object m, true)
```

**4. Conditional inference with data:**
```
if Employee(object e) also salary(object e, var s) also greaterThan(var s, 100000) then HighEarner(object e)
```

**5. Multiple condition inference:**
```
if Class(object c) also hasMethod(object c, object m1) also hasMethod(object c, object m2) also distinct(object m1, object m2) then hasMultipleMethods(object c, true)
```

#### Syntax Comparison

| ✅ Correct | ❌ Incorrect |
|------------|--------------|
| `if A(object x) also B(object x) then C(object x)` | `if A(object x) and B(object x) then C(object x)` |
| `if A(object x) then B(object x)` | `if A(object x) and then B(object x)` |
| `object x` | `?x` (works but `object x` preferred) |

### 18. Special Concepts

```
top
bottom
everything
```

Alternative forms:
- `top`: Thing, Anything (represents owl:Thing)
- `bottom`: Nothing (represents owl:Nothing)
- `everything`: any_value (top datatype)

### 19. Has Key

Functional dependencies:
```
ClassName has_key (propertyName1, propertyName2)
ClassName identified_by (propertyName1, propertyName2)
```

### 20. Data Types

**Boolean**:
```
false
true
```
(Case-insensitive: FALSE, True, etc. all work)

**Numbers**:
```
42
-10
3.14
1.5e10
$3.99
```

**Strings**:
```
'Hello World'
"Hello World"
'It''s'
```
(Use double quotes or single quote inside single quotes to escape)

**Date-Time** (ISO 8601):
```
2024-01-15
2024-01-15T10:30:00Z
2024-01-15T10:30:00+01:00
```

**Duration**:
```
1Y2M3D
T5H30M
P1DT12H
```

### 21. Datatype Definitions

```
DatatypeName is_same_data_type_as constraint
DatatypeName has_datatype constraint
```

**Examples with facets:**
```
AdultAge has_datatype (>= 18)
PositiveInt has_datatype (> 0)
ShortString has_datatype (len <= 100)
EmailPattern has_datatype (pattern '.*@.*')
```

Value sets:
```
DayOfWeek has_datatype {'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'}
```

### 22. Identifiers

**Simple identifiers**: Most programming language patterns work
```
Person
MyClass
my_variable
Person123
_private
__dunder__
$variable
@decorator
file.txt
```

**Backtick quoted identifiers** for complex patterns:
```
`complex-name`
`List<String>`
`Dict[str, int]`
`path/to/file`
`http://example.com`
`name with spaces`
```

---

## Complete DL Example (AI Syntax)

```
Animal is_a Thing
Mammal is_subclass_of Animal
Dog ISA Mammal
Cat extends Mammal

not equals (Dog, Cat)

Person is_a Thing
Female is_subclass_of Person
Male is_subclass_of Person
Parent is_subclass_of Person

Mother is_equivalent_to (Female and Parent)
Father equals (Male AND Parent)

# Prefix syntax
(some hasChild that_is Thing) is_subclass_of Parent
Thing is_subclass_of (all hasChild that_is Person)

# Manchester syntax (NEW - fully equivalent)
(hasChild some Thing) is_subclass_of Parent
Thing is_subclass_of (hasChild all Person)

# Cardinality - Prefix syntax
Parent is_subclass_of (at_least 1 hasChild that_is Thing)
SingleParent equals (exactly 1 hasChild that_is Thing)
LargeFamily is_equivalent_to (>= 4 hasChild that Person)

# Cardinality - Manchester syntax (NEW)
Parent is_subclass_of (hasChild >= 1 Thing)
SingleParent equals (hasChild = 1 Thing)
LargeFamily is_equivalent_to (hasChild >= 4 Person)

ancestorOf composed_with ancestorOf is_subproperty_of ancestorOf

knows is_same_role_as knows inverse

Person(Alice)
Person(Bob)
Female(Alice)
Male(Bob)
Parent(Alice)

hasChild(Alice, Bob)
age(Alice, 30)
age(Bob, 5)

AliceSmith = Alice

if Person(?x) also hasChild(?x, ?y) then Parent(?x)
```

---

## DL Query Tools (MCP)

RETER MCP Server provides two specialized tools for querying using DL class expressions with AI variant syntax:

### `query_logic(query, limit)` - Find Instances

Query for all individuals that satisfy a DL class expression.

**Syntax:**
```python
result = query_logic(
    query="DL_class_expression",
    limit=1000
)
```

**Parameters:**
- `query` (string): DL class expression using AI variant syntax
- `limit` (int, optional): Maximum number of results (default: 1000)

**Returns:** Dictionary with:
- `entities`: List of matching individuals with metadata
- `execution_time_ms`: Query execution time

**Examples:**

```python
# Simple class query
result = query_logic(query="Person")
# Returns all individuals that are Persons

# Intersection (AND)
result = query_logic(query="Person and Doctor")
# Returns individuals that are BOTH Person AND Doctor

# Union (OR)
result = query_logic(query="Doctor or Nurse")
# Returns individuals that are EITHER Doctor OR Nurse

# Complement (NOT)
result = query_logic(query="Person and not Doctor")
# Returns Persons who are not Doctors

# Existential restriction (SOME) - Prefix syntax
result = query_logic(query="some hasChild that_is Doctor")
# Returns individuals who have at least one child who is a Doctor

# Existential restriction (SOME) - Manchester syntax (NEW)
result = query_logic(query="hasChild some Doctor")
# Same as above - Manchester syntax

# Universal restriction (ALL) - Prefix syntax
result = query_logic(query="all hasChild that_is Doctor")
# Returns individuals where ALL children are Doctors

# Universal restriction (ALL) - Manchester syntax (NEW)
result = query_logic(query="hasChild all Doctor")
# Same as above - Manchester syntax

# Cardinality - Prefix syntax
result = query_logic(query="at_least 2 hasChild that_is Person")
# Returns individuals with at least 2 children

# Cardinality - Manchester syntax (NEW)
result = query_logic(query="hasChild >= 2 Person")
# Same as above - Manchester syntax

# Complex expressions with Manchester syntax (NEW)
result = query_logic(query="py:Class and (hasMethod some py:Method)")
# Returns Python classes that have at least one method

# Mixed syntax
result = query_logic(query="(Person and Employee) or Manager")
# Returns individuals who are (Person AND Employee) OR Manager

# With limit
result = query_logic(query="Person", limit=10)
# Returns first 10 Persons
```

**Important Notes:**
- Uses AI variant DL syntax (case-insensitive keywords)
- Supports **two syntaxes**: Prefix (`some prop that_is Class`) and Manchester (`prop some Class`)
- Both syntaxes are fully equivalent and can be mixed in the same query
- Supports: `and`, `or`, `not`, `some`, `all`, cardinality, etc.
- Returns all instances matching the class expression (up to `limit`)
- Result has `entities` list and `execution_time_ms`

### `is_true_logic(axiom)` - Check Existence

Check if ANY individual satisfies a DL class expression (boolean result).

**Syntax:**
```python
result = is_true_logic(
    axiom="DL_class_expression"
)
```

**Parameters:**
- `axiom` (string): DL class expression using AI variant syntax

**Returns:** Dictionary with:
- `result`: Boolean (`True` if at least one match exists, `False` otherwise)
- `explanation`: Optional explanation (if available)
- `execution_time_ms`: Query execution time

**IMPORTANT - Return Format:**
The function ALWAYS returns a dictionary, not a boolean directly. You MUST access the `"result"` key to get the boolean value:

```python
# ✓ CORRECT - Access result["result"]
result = is_true_logic(axiom="Person")
if result["result"]:  # Access the "result" key
    print("At least one Person exists")

# ✗ WRONG - Don't use the dictionary directly as boolean
if is_true_logic(axiom="Person"):  # This checks if dict exists (always True!)
    print("This is wrong!")
```

**Examples:**

```python
# Check if any Person exists
if is_true_logic(axiom="Person")["result"]:
    print("At least one Person exists")

# Check intersection
result = is_true_logic(axiom="Person and Doctor")
if result["result"]:
    print("At least one individual is both Person and Doctor")
else:
    print("No Person-Doctor combination exists")

# Check complex expression
if is_true_logic(axiom="Person and Employee and Manager")["result"]:
    print("Found a Person who is both Employee and Manager")

# Check existence with restrictions - Prefix syntax
if is_true_logic(axiom="some hasChild that_is Doctor")["result"]:
    print("Someone has a Doctor child")

# Check existence with restrictions - Manchester syntax (NEW)
if is_true_logic(axiom="hasChild some Doctor")["result"]:
    print("Someone has a Doctor child")

# Check Python classes with methods - Manchester syntax (NEW)
if is_true_logic(axiom="py:Class and (hasMethod some py:Method)")["result"]:
    print("At least one Python class has methods")
```

**Important Notes:**
- Returns boolean in `result` field (not the individuals themselves)
- Supports **two syntaxes**: Prefix and Manchester (fully equivalent)
- Faster than `query_logic()` if you only need existence check
- Use for boolean queries ("Does X exist?")

### Common Mistakes ⚠️

**1. WRONG** - Using `and` instead of `also` in SWRL rules:
```
# ✗ DON'T DO THIS
if Person(object x) and hasParent(object x, object y) then hasAncestor(object x, object y)
```

**CORRECT** - Use `also` keyword:
```
# ✓ DO THIS
if Person(object x) also hasParent(object x, object y) then hasAncestor(object x, object y)
```

**Note**: SWRL uses `also` to combine conditions, not `and`. Using `and` will cause parse errors like:
```
Parse error: mismatched input 'and' expecting SWRLTHEN
```

---

**2. WRONG** - Using `dl_ask()` to check individual assertions:
```python
# ✗ DON'T DO THIS
result = reter.dl_ask("Alice type Mother")  # WRONG! Not a class expression
```

**CORRECT** - Use `pattern()` for individual checks:
```python
# ✓ DO THIS - Check if specific individual is in a class
result = reter.pattern(('Alice', 'type', 'Mother'))
if len(result) > 0:
    print("Alice is a Mother")

# OR use instances_of() and check
mothers = reter.instances_of("Mother")
alice_is_mother = 'Alice' in mothers.to_pydict()['?x']
```

**CORRECT** - Use `dl_ask()` for class expression queries:
```python
# ✓ DO THIS - Check if Mother class has any instances
if reter.dl_ask("Mother"):
    print("At least one Mother exists in the knowledge base")

# ✓ DO THIS - Check if complex expression has instances
if reter.dl_ask("Female and Parent"):
    print("At least one individual is both Female and Parent")
```

### When to Use Which Method

| Goal | Method | Example |
|------|--------|---------|
| Find all instances of a class | `dl_query()` | `dl_query("Person")` |
| Check if class has instances | `dl_ask()` | `dl_ask("Person and Doctor")` |
| Check if individual is in class | `pattern()` | `pattern(('Alice', 'type', 'Person'))` |
| Get all instances of a class | `instances_of()` | `instances_of("Person")` |
| Complex SELECT queries | `reql()` with SELECT | `reql("SELECT ?x WHERE { ?x type Person }")` |
| Pattern existence check | `reql()` with ASK | `reql("ASK WHERE { ?x type Person }")` |
| Query with UNION/MINUS/OPTIONAL | `reql()` | `reql("SELECT ?x WHERE { ?x type Person MINUS { ?x type Doctor } }")` |

---

## REQL Grammar

RETER supports a simplified REQL dialect for querying the knowledge base. REQL uses standard syntax regardless of the DL variant.

### Unified Query Interface

All query types (SELECT, ASK, DESCRIBE) use the same **`reql()`** method:

```python
# SELECT query - returns data table
result = reter.reql("SELECT ?x WHERE { ?x type Person }")

# ASK query - returns single-row table with boolean 'result' column
result = reter.reql("ASK WHERE { ?x type Person }")
exists = result['result'][0].as_py()  # Extract boolean

# DESCRIBE query - returns table with subject, predicate, object, object_type columns
result = reter.reql("DESCRIBE Alice")
print(result.to_pandas())  # View as DataFrame
```

### Query Types

REQL supports three query types:
1. **SELECT** - Retrieve data (returns table with result columns)
2. **ASK** - Check existence (returns table with boolean 'result' column)
3. **DESCRIBE** - Explore resource facts (returns table with subject, predicate, object, object_type columns)

### SELECT Query Structure

```sparql
SELECT [DISTINCT] ?var1 ?var2 ... [(SELECT ...) AS ?alias]
WHERE {
  pattern1 .
  pattern2 .
  ...
}
[GROUP BY ?var]
[HAVING (condition)]
[ORDER BY ?var]
[LIMIT n]
[OFFSET n]
```

**Features:**
- Variables: `?var1`, `?var2`
- Aggregations: `COUNT(*)`, `SUM(?var)`, `AVG(?var)`, etc.
- Subqueries: `(SELECT COUNT(?x) WHERE { ... }) AS ?alias` ✨ **NEW**

### ASK Query Structure

```sparql
ASK WHERE {
  pattern1 .
  pattern2 .
  ...
}
```

**Returns:** Single-row table with `result` column containing boolean value.

**Example:**
```python
# Check if any Person exists
result = reter.reql("ASK WHERE { ?x type Person }")
exists = result['result'][0].as_py()  # True or False
```

### DESCRIBE Query Structure

```sparql
DESCRIBE resource1 [resource2 ...]
[WHERE {
  pattern1 .
  pattern2 .
  ...
}]
```

**Returns:** Table with columns: `subject`, `predicate`, `object`, `object_type`

DESCRIBE queries return all facts (triples) involving the specified resources. The resource can appear as either subject or object in the returned triples (bidirectional search).

**Syntax Variants:**

1. **Direct resources** - Describe specific entities:
```sparql
DESCRIBE Alice
DESCRIBE Alice Bob Charlie
```

2. **Variable-based with WHERE** - Describe resources matching a pattern:
```sparql
DESCRIBE ?person WHERE {
  ?person type Doctor
}
```

3. **All entities** - Describe all entities in the knowledge base:
```sparql
DESCRIBE *
```

**Output Schema:**
- `subject` (string) - The subject of the triple
- `predicate` (string) - The property/relationship
- `object` (string) - The object value
- `object_type` (string) - Type hint: "entity", "number", "boolean", or "string"

**Examples:**
```python
# Describe a single person
result = reter.reql("DESCRIBE Alice")
# Returns all triples where Alice is subject or object:
#   Alice type Person
#   Alice age 30
#   Alice hasChild Bob
#   hasParent(Bob, Alice)  # Alice appears as object

# Describe all doctors
result = reter.reql("""
    DESCRIBE ?doc WHERE {
        ?doc type Doctor
    }
""")

# Describe multiple resources
result = reter.reql("DESCRIBE Alice Bob Charlie")

# Describe with complex WHERE clause
result = reter.reql("""
    DESCRIBE ?person WHERE {
        ?person type Person .
        ?person age ?age .
        FILTER(?age >= 30)
    }
""")
```

### Triple Patterns

Basic pattern: `subject predicate object`

**Each element can be:**
- Variable: `?var` or `$var`
- Identifier: `ClassName`, `propertyName`, `individualName`
- Literal: `'string'`, `42`, `3.14`, `true`, `false`

**Examples:**
```sparql
?person type Person
?person hasChild ?child
Alice hasChild ?child
?x worksFor Acme
```

### Variables

Variables start with `?` or `$`:
```sparql
?person  ?x  ?y  ?name  ?age
$person  $x  $y  $name  $age
```

### Multiple Patterns (Join)

Patterns separated by `.` (period):

```sparql
SELECT ?person ?age WHERE {
  ?person type Person .
  ?person hasAge ?age
}
```

### UNION

Alternative patterns (logical OR):

```sparql
SELECT ?animal WHERE {
  { ?animal type Dog } UNION { ?animal type Cat }
}
```

Multiple alternatives:

```sparql
SELECT ?x WHERE {
  { ?x type Person }
  UNION
  { ?x type Doctor }
  UNION
  { ?x type Nurse }
}
```

### MINUS

Exclude matching patterns (set difference):

```sparql
# Find people who are NOT doctors
SELECT ?person WHERE {
  ?person type Person
  MINUS {
    ?person type Doctor
  }
}
```

Multiple MINUS patterns:

```sparql
# Find people who are neither doctors nor employees
SELECT ?person WHERE {
  ?person type Person
  MINUS { ?person type Doctor }
  MINUS { ?person type Employee }
}
```

### OPTIONAL

Left outer join (include results even if optional pattern doesn't match):

```sparql
# Find all people and their ages (include people without age)
SELECT ?person ?age WHERE {
  ?person type Person
  OPTIONAL {
    ?person hasAge ?age
  }
}
```

Multiple OPTIONAL patterns:

```sparql
# Find people with optional age and email
SELECT ?person ?age ?email WHERE {
  ?person type Person
  OPTIONAL { ?person hasAge ?age }
  OPTIONAL { ?person hasEmail ?email }
}
```

### FILTER

Constrain query results with boolean expressions:

```sparql
SELECT ?person ?age WHERE {
  ?person type Person .
  ?person hasAge ?age .
  FILTER(?age >= 18)
}
```

#### Filter Operators

**Comparison:**
- `=` Equal
- `!=` Not equal
- `<` Less than
- `>` Greater than
- `<=` Less than or equal
- `>=` Greater than or equal

**Logical:**
- `&&` AND
- `||` OR
- `!` NOT

#### Built-in Functions

- **STR(?var)** - Convert to string
- **BOUND(?var)** - Check if variable is bound
- **REGEX(?string, pattern)** - Regular expression matching
- **CONTAINS(?string, substring)** - Check if string contains substring
- **STRSTARTS(?string, prefix)** - Check if string starts with prefix
- **STRENDS(?string, suffix)** - Check if string ends with suffix
- **LEVENSHTEIN(?str1, ?str2)** - Compute edit distance between two strings (returns integer)
- **EXISTS { pattern }** - Check if pattern has solutions (subquery existence check)
- **NOT EXISTS { pattern }** - Check if pattern has no solutions (subquery non-existence check)

**LEVENSHTEIN Examples:**

```sparql
# Find methods with similar names (edit distance ≤ 2)
SELECT ?m1 ?m2 ?name1 ?name2 WHERE {
  ?m1 type oo:Method . ?m1 name ?name1 .
  ?m2 type oo:Method . ?m2 name ?name2 .
  FILTER(?m1 != ?m2)
  FILTER(LEVENSHTEIN(?name1, ?name2) <= 2)
}

# Find potential typos in class names
SELECT ?class ?name WHERE {
  ?class type oo:Class . ?class name ?name .
  FILTER(LEVENSHTEIN(?name, 'UserService') <= 2)
}
```

**EXISTS/NOT EXISTS Examples:**

```sparql
# Find people who have at least one child
SELECT ?person WHERE {
  ?person type Person .
  FILTER(EXISTS { ?person hasChild ?child })
}

# Find people without children
SELECT ?person WHERE {
  ?person type Person .
  FILTER(NOT EXISTS { ?person hasChild ?child })
}

# Find people who have a child who is a doctor
SELECT ?person WHERE {
  ?person type Person .
  FILTER(EXISTS {
    ?person hasChild ?child .
    ?child type Doctor
  })
}
```

### Aggregation Functions

REQL supports SQL-style aggregation with GROUP BY and HAVING clauses.

#### Aggregation Functions

- **COUNT(*)** - Count all rows
- **COUNT(?var)** - Count non-null values of variable
- **COUNT(DISTINCT ?var)** - Count distinct values
- **SUM(?var)** - Sum of numeric values
- **AVG(?var)** - Average of numeric values
- **MIN(?var)** - Minimum value
- **MAX(?var)** - Maximum value

#### Basic Aggregation

```sparql
# Count all people
SELECT COUNT(*) WHERE {
  ?person type Person
}

# Count people with ages
SELECT COUNT(?age) WHERE {
  ?person type Person .
  ?person age ?age
}
```

#### GROUP BY

Group results and apply aggregations:

```sparql
# Count people by department
SELECT ?dept COUNT(*) AS ?count WHERE {
  ?person type Person .
  ?person worksIn ?dept
}
GROUP BY ?dept

# Average salary by department
SELECT ?dept AVG(?salary) AS ?avg_salary WHERE {
  ?person type Employee .
  ?person worksIn ?dept .
  ?person salary ?salary
}
GROUP BY ?dept
```

#### HAVING

Filter aggregated results:

```sparql
# Departments with more than 5 people
SELECT ?dept COUNT(*) AS ?count WHERE {
  ?person type Person .
  ?person worksIn ?dept
}
GROUP BY ?dept
HAVING (?count > 5)

# Departments with average salary > 50000
SELECT ?dept AVG(?salary) AS ?avg_salary WHERE {
  ?person type Employee .
  ?person worksIn ?dept .
  ?person salary ?salary
}
GROUP BY ?dept
HAVING (?avg_salary > 50000)
```

#### Aggregation with AS Alias

Use `AS` to name aggregation results:

```sparql
SELECT ?category (COUNT(*) AS ?total) (AVG(?price) AS ?avg_price) WHERE {
  ?product type Product .
  ?product category ?category .
  ?product price ?price
}
GROUP BY ?category
ORDER BY DESC(?total)
```

### Subqueries ✨ **NEW**

REQL now supports scalar subqueries in the SELECT clause, allowing nested queries that return single values.

#### Basic Syntax

Subqueries must be wrapped in parentheses and have an `AS` alias:

```sparql
SELECT ?variable (SELECT ... WHERE { ... }) AS ?alias
WHERE { ... }
```

#### Scalar Subqueries

Scalar subqueries return a single value (typically from an aggregation):

```sparql
# Count total people in a subquery
SELECT ?person (SELECT COUNT(?x) WHERE { ?x type Person }) AS ?total
WHERE { ?person type Person }
```

**Result**:
```
?person | total
--------|------
Alice   | 4.0
Bob     | 4.0
Charlie | 4.0
Diana   | 4.0
```

#### Uncorrelated Subqueries

Uncorrelated subqueries don't reference variables from the parent query. They execute once and broadcast the result:

```sparql
# Average age across all people
SELECT ?person (SELECT AVG(?age) WHERE { ?x age ?age }) AS ?avg_age
WHERE { ?person type Person }

# Sum of all salaries
SELECT ?dept (SELECT SUM(?salary) WHERE { ?e salary ?salary }) AS ?total_payroll
WHERE { ?dept type Department }
```

#### Correlated Subqueries

Correlated subqueries reference variables from the parent query:

```sparql
# Count friends per person (correlated on ?person)
SELECT ?person (SELECT COUNT(?friend) WHERE { ?person knows ?friend }) AS ?friend_count
WHERE { ?person type Person }

# Count employees per department (correlated on ?dept)
SELECT ?dept (SELECT COUNT(?emp) WHERE { ?emp worksIn ?dept }) AS ?employee_count
WHERE { ?dept type Department }
```

**Note**: Full row-level binding for correlated subqueries is in development. Uncorrelated subqueries are production-ready.

#### Multiple Subqueries

You can include multiple subqueries in the same SELECT:

```sparql
SELECT ?person
       (SELECT COUNT(?friend) WHERE { ?person knows ?friend }) AS ?friends
       (SELECT COUNT(?food) WHERE { ?person likes ?food }) AS ?likes
       (SELECT AVG(?age) WHERE { ?x age ?age }) AS ?avg_age
WHERE { ?person type Person }
```

#### Subqueries with Aggregations

Common pattern - use aggregations in subqueries:

```sparql
# Count and sum together
SELECT ?category
       (SELECT COUNT(?p) WHERE { ?p category ?category }) AS ?product_count
       (SELECT SUM(?price) WHERE { ?p category ?category . ?p price ?price }) AS ?total_value
WHERE { ?category type Category }

# Min and max
SELECT ?dept
       (SELECT MIN(?salary) WHERE { ?e worksIn ?dept . ?e salary ?salary }) AS ?min_sal
       (SELECT MAX(?salary) WHERE { ?e worksIn ?dept . ?e salary ?salary }) AS ?max_sal
WHERE { ?dept type Department }
```

#### Subqueries with FILTER

Subqueries can use FILTER expressions, including references to parent variables:

```sparql
# Count friends older than 25
SELECT ?person
       (SELECT COUNT(?friend)
        WHERE {
            ?person knows ?friend .
            ?friend age ?age .
            FILTER(?age > 25)
        }) AS ?adult_friends
WHERE { ?person type Person }

# Count high-value products per category
SELECT ?category
       (SELECT COUNT(?p)
        WHERE {
            ?p category ?category .
            ?p price ?price .
            FILTER(?price >= 100)
        }) AS ?premium_count
WHERE { ?category type Category }
```

#### Subqueries with Complex Patterns

Subqueries work with UNION in the parent query:

```sparql
SELECT ?entity
       (SELECT COUNT(?x) WHERE { ?x type Person }) AS ?total
WHERE {
    { ?entity type Person }
    UNION
    { ?entity type Organization }
}
```

#### Requirements and Limitations

**Required**:
- Subqueries must have `AS ?alias` clause
- Subqueries must be wrapped in parentheses: `(SELECT ...)`
- Subqueries should return scalar values (single row, single column)

**Current Limitations**:
- Correlated subquery row-level binding is in development
- Nested subqueries (subquery within subquery) not yet supported
- Table subqueries (multi-row results) not yet supported

**Best Practices**:
- Use aggregations (COUNT, SUM, AVG, MIN, MAX) in subqueries
- Prefer uncorrelated subqueries when possible (better performance)
- Give descriptive aliases: `AS ?friend_count`, `AS ?avg_salary`

#### Performance Notes

- **Uncorrelated subqueries**: Execute once, excellent performance
- **Correlated subqueries**: Memoization cache for repeated values
- **Multiple subqueries**: Execute independently in sequence

### DISTINCT

Remove duplicate results:

```sparql
SELECT DISTINCT ?person WHERE {
  ?person type Person
}
```

### ORDER BY

Sort results:

```sparql
ORDER BY ?age
ORDER BY DESC(?age)
ORDER BY ?name ?age
```

### LIMIT and OFFSET

Pagination:

```sparql
LIMIT 10
OFFSET 20
LIMIT 10 OFFSET 20
```

### Complete REQL Examples

#### SELECT Query Examples

**Example 1: Find all dogs**
```sparql
SELECT ?dog WHERE {
  ?dog type Dog
}
```

**Example 2: Find people and their ages**
```sparql
SELECT ?person ?age WHERE {
  ?person type Person .
  ?person age ?age
}
ORDER BY DESC(?age)
LIMIT 10
```

**Example 3: Find employees and their companies**
```sparql
SELECT ?person ?company WHERE {
  ?person type Employee .
  ?person worksFor ?company
}
```

**Example 4: Find animals (dogs or cats)**
```sparql
SELECT DISTINCT ?animal WHERE {
  { ?animal type Dog } UNION { ?animal type Cat }
}
```

#### ASK Query Examples

**Example 1: Check if any Person exists**
```sparql
ASK WHERE {
  ?x type Person
}
```

**Example 2: Check if anyone has children**
```sparql
ASK WHERE {
  ?x hasChild ?y
}
```

**Example 3: Check if there are any Doctors who are also Teachers**
```sparql
ASK WHERE {
  ?x type Doctor .
  ?x type Teacher
}
```

**Example 4: Check with UNION patterns**
```sparql
ASK WHERE {
  { ?x type Dog } UNION { ?x type Cat }
}
```

**Example 5: Check with MINUS patterns**
```sparql
ASK WHERE {
  ?x type Person
  MINUS { ?x type Doctor }
}
```

**Example 6: Using ASK results in Python**
```python
# Query for existence
result = reter.reql("ASK WHERE { ?x type Person }")

# Extract boolean value from Arrow table
exists = result['result'][0].as_py()

# Use in conditional
if exists:
    print("At least one Person exists in the knowledge base")
else:
    print("No Persons found")
```

---

## Usage Tips

### For DL Queries

1. **Keywords are case-insensitive**: `is_a`, `IS_A`, `Is_A` all work
2. **Two syntax styles available** ✨:
   - **Prefix**: `some hasChild that_is Person` (original)
   - **Manchester**: `hasChild some Person` (NEW - OWL standard)
   - Both are fully equivalent - use whichever reads better!
3. **Use standard ASCII**: Regular parentheses `()` and commas `,`
4. **Multiple operator forms**: Choose what reads best
5. **Backticks for complex names**: `` `my-complex-name` ``

### For REQL Queries

1. **Use simple identifiers** (no namespace prefixes)
2. **Always end patterns with period** (except last in block)
3. **Use LIMIT** to prevent unbounded results
4. **FILTER works** - use for post-filtering results
5. **Aggregations supported** - use COUNT, SUM, AVG, MIN, MAX with GROUP BY
6. **ASK queries are optimized** - use for boolean existence checks (early termination)

### Performance

1. **DL queries** are faster for class membership checks
2. **REQL queries** are better for joins and data extraction
3. **ASK queries are optimized** with early termination - use instead of SELECT + COUNT for existence checks
4. **Use `save_state`/`load_state`** to persist reasoning results
5. **Filter early** in REQL queries
6. **Use specific patterns** before general ones

#### ASK Query Optimizations

ASK queries use several optimizations for fast existence checking:

- **Early termination**: Stops immediately after finding first match
- **Internal LIMIT 1**: Only retrieves one result
- **Skip unnecessary operations**: No sorting (ORDER BY) or deduplication (DISTINCT)
- **Optimized across all patterns**: UNION, MINUS, and OPTIONAL patterns all support early exit

**Example:**
```python
# Instead of:
result = reter.reql("SELECT ?x WHERE { ?x type Person } LIMIT 1")
exists = result.num_rows > 0  # Slower

# Use optimized ASK:
result = reter.reql("ASK WHERE { ?x type Person }")
exists = result['result'][0].as_py()  # Faster - early termination
```

---

## Quick Reference Tables

### AI Lexer DL Operators

| AI Syntax | Alternative Forms | Meaning | Example (Prefix) | Example (Manchester) |
|-----------|------------------|---------|------------------|----------------------|
| `is_a` | `isa`, `extends`, `is_subclass_of` | SubClassOf | `Dog is_a Animal` | N/A |
| `equals` | `is_equivalent_to`, `same_as` | EquivalentClasses | `Mother equals (Female and Parent)` | N/A |
| `and` | `intersection_with`, `with` | Intersection | `Person and Doctor` | N/A |
| `or` | `union_with` | Union | `Dog or Cat` | N/A |
| `not` | `complement_of`, `except` | Complement | `not Doctor` | N/A |
| `some` | `exists`, `has` | SomeValuesFrom | `some hasChild that_is Person` | `hasChild some Person` ✨ |
| `all` | `only`, `must` | AllValuesFrom | `all hasChild that_is Person` | `hasChild all Person` ✨ |
| `>=` | `at_least` | Min cardinality | `>= 2 hasChild that_is Thing` | `hasChild >= 2 Thing` ✨ |
| `<=` | `at_most` | Max cardinality | `<= 3 hasChild that_is Thing` | `hasChild <= 3 Thing` ✨ |
| `=` | `exactly` | Exact cardinality | `= 2 hasChild that_is Thing` | `hasChild = 2 Thing` ✨ |

**Note**: ✨ Indicates NEW Manchester syntax support (property before quantifier)

### REQL Keywords

| Keyword | Purpose |
|---------|---------|
| `SELECT` | Query for specific variables (returns data table) |
| `ASK` | Check pattern existence (returns boolean) |
| `WHERE` | Graph pattern block |
| `FILTER` | Constrain values |
| `UNION` | Alternative patterns |
| `MINUS` | Remove matching patterns (set difference) |
| `OPTIONAL` | Left join patterns (include if exists) |
| `DISTINCT` | Remove duplicates |
| `ORDER BY` | Sort results |
| `GROUP BY` | Group results for aggregation |
| `HAVING` | Filter aggregated results |
| `LIMIT` | Maximum number of results |
| `OFFSET` | Skip first N results |

---

## REQL Limitations

RETER's REQL is intentionally simplified. The following are **NOT supported**:

1. ❌ **PREFIX/BASE declarations** - Use simple identifiers
2. ❌ **IRIs in angle brackets** - No `<http://...>`
3. ❌ **CONSTRUCT queries** - Only SELECT, ASK, and DESCRIBE supported
4. ❌ **Property paths** - No `+`, `*`, `/`, `^` on predicates
5. ❌ **Named graphs** - Single default graph only
6. ✅ **Scalar subqueries** ✨ **NEW** - Nested SELECT queries in SELECT clause (uncorrelated fully working, correlated in development)
7. ❌ **VALUES clause** - Cannot inject inline data
8. ❌ **BIND** - Cannot create new variables
9. ❌ **Service federation** - No REQL endpoints

### Fully Supported Features ✅

- **SELECT queries** - Retrieve data from knowledge base
- **ASK queries** - Boolean existence checks with early termination optimization
- **DESCRIBE queries** - Explore all facts (triples) about resources with support for UNION, MINUS, OPTIONAL, and FILTER in WHERE clauses
- **Scalar subqueries** ✨ **NEW** - Nested queries in SELECT clause with aggregations (uncorrelated subqueries production-ready)
- **FILTER expressions** - All comparison and logical operators (properly scoped to include variables from WHERE clause)
- **EXISTS/NOT EXISTS** - Subquery existence checks in FILTER
- **UNION patterns** - Alternative graph patterns (fully supported in all query types including DESCRIBE)
- **MINUS patterns** - Set difference (exclusion patterns)
- **OPTIONAL patterns** - Left outer join semantics
- **Aggregation** - COUNT, SUM, AVG, MIN, MAX with GROUP BY and HAVING
- **DISTINCT** - Remove duplicate results
- **ORDER BY** - Sort results (ASC/DESC)
- **LIMIT/OFFSET** - Pagination