# Python Refactoring to Patterns Plugin

**Version**: 1.0.0
**Category**: Python Analysis, Refactoring, Design Patterns
**Requires RETER**: Yes

## Overview

This plugin detects refactoring opportunities based on patterns from **"Refactoring to Patterns"** by Joshua Kerievsky, adapted for Python code. It analyzes Python codebases loaded into RETER and identifies opportunities to apply classic design patterns through refactoring.

## Motivation

The book "Refactoring to Patterns" describes how to evolve code toward better design patterns incrementally. This plugin automates the detection of these opportunities by analyzing:

- Constructor complexity
- Code duplication patterns
- Opportunities for pattern application
- Anti-patterns that could be refactored

## Tools

### 1. `detect_chain_constructors`

Detects opportunities to apply the **Chain Constructors** pattern.

#### Pattern Description

**Problem**: Constructors with many parameters, duplicated initialization logic, or "telescoping constructor" anti-pattern (multiple constructors or many default parameters).

**Solution**: Create simpler factory methods that delegate to a more complete constructor, reducing duplication and providing cleaner APIs.

#### Python Adaptation

In Java, this pattern uses constructor overloading:
```java
public Loan(float notional, float outstanding, int rating, Date expiry) {
    this(new TermROC(), notional, outstanding, rating, expiry, null);
}
```

In Python, we achieve this with `@classmethod` factory methods:
```python
@classmethod
def from_basic(cls, notional, outstanding, rating, expiry):
    """Create loan with default strategy."""
    return cls(TermROC(), notional, outstanding, rating, expiry, None)
```

#### What It Detects

1. **Complex Constructors**: `__init__` methods with many parameters (configurable threshold)
2. **Telescoping Constructors**: Many parameters with default values
3. **Missing Factory Methods**: Classes that could benefit from convenience constructors
4. **Existing Factories**: Classes already using factory methods (for review)

#### Parameters

- `instance_name` (required): RETER instance containing Python code
- `min_parameters` (default: 5): Minimum parameters to flag as complex
- `include_defaults` (default: true): Include parameters with defaults in analysis
- `class_name` (optional): Analyze specific class only

#### Example Usage

```python
# Analyze all classes
result = detect_chain_constructors(instance_name="my_project", min_parameters=4)

# Analyze specific class
result = detect_chain_constructors(
    instance_name="my_project",
    class_name="UserService",
    min_parameters=3
)
```

#### Output Structure

```json
{
  "success": true,
  "pattern": "Chain Constructors",
  "opportunities": [
    {
      "class_name": "UserService",
      "severity": "high",
      "parameter_count": 8,
      "parameters": [
        {"name": "db_connection", "has_default": false},
        {"name": "cache", "has_default": true},
        {"name": "logger", "has_default": true}
      ],
      "default_parameter_count": 5,
      "has_telescoping_constructor": true,
      "existing_factory_methods": ["from_config", "from_env"],
      "recommendations": [
        {
          "type": "telescoping_constructor",
          "description": "Telescoping constructor detected: 5 parameters with defaults",
          "suggestion": "Consider creating @classmethod factory methods for common parameter combinations"
        }
      ]
    }
  ],
  "count": 1,
  "summary": {
    "total_flagged": 1,
    "high_severity": 1,
    "medium_severity": 0,
    "low_severity": 0,
    "with_telescoping": 1,
    "with_existing_factories": 1
  }
}
```

#### Severity Levels

- **High**: 8+ parameters - Urgent refactoring needed
- **Medium**: 6-7 parameters - Should consider refactoring
- **Low**: 5 parameters - Minor opportunity

#### Recommendations Types

1. **telescoping_constructor**: Too many parameters with defaults
2. **complex_constructor**: Many parameters, no factory methods
3. **existing_factories**: Already has factory methods (review for chaining)
4. **suggested_factory**: Specific factory method suggestion

## Refactoring Example

### Before

```python
class Loan:
    def __init__(self, strategy, notional, outstanding, rating, expiry,
                 maturity=None, collateral=None, risk_level=1):
        self.strategy = strategy
        self.notional = notional
        self.outstanding = outstanding
        self.rating = rating
        self.expiry = expiry
        self.maturity = maturity
        self.collateral = collateral
        self.risk_level = risk_level
```

**Issues**:
- 8 parameters (overwhelming)
- 3 parameters with defaults (telescoping)
- No convenient construction methods

### After - Using Chain Constructors

```python
class Loan:
    def __init__(self, strategy, notional, outstanding, rating, expiry,
                 maturity=None, collateral=None, risk_level=1):
        """Full constructor - all parameters."""
        self.strategy = strategy
        self.notional = notional
        self.outstanding = outstanding
        self.rating = rating
        self.expiry = expiry
        self.maturity = maturity
        self.collateral = collateral
        self.risk_level = risk_level

    @classmethod
    def term_loan(cls, notional, outstanding, rating, expiry):
        """Create a simple term loan."""
        return cls(TermROC(), notional, outstanding, rating, expiry)

    @classmethod
    def revolving_loan(cls, notional, outstanding, rating, expiry, maturity):
        """Create a revolving loan."""
        return cls(RevolvingTermROC(), notional, outstanding, rating,
                  expiry, maturity)

    @classmethod
    def secured_loan(cls, strategy, notional, outstanding, rating,
                     expiry, collateral):
        """Create a secured loan with collateral."""
        return cls(strategy, notional, outstanding, rating, expiry,
                  collateral=collateral)
```

**Benefits**:
- Clear, intention-revealing factory methods
- Users don't need to know about strategies
- Easy to add new construction patterns
- Reduced cognitive load

## Future Patterns (Planned)

1. **Replace Constructor with Factory Method**
2. **Introduce Polymorphic Creation with Factory Method**
3. **Move Embellishment to Decorator**
4. **Replace Conditional Logic with Strategy**
5. **Form Template Method**
6. **Extract Composite**
7. **Replace One/Many Distinctions with Composite**
8. **Replace State-Altering Conditionals with State**
9. **Replace Implicit Tree with Composite**

## Integration

This plugin integrates with:
- **python_basic**: Uses class and method detection
- **python_advanced**: Complements code smell detection
- **UML plugin**: Visualize before/after refactoring

## Technical Details

### REQL Queries Used

1. **Find __init__ methods**:
```reql
SELECT ?class ?className ?init ?paramCount WHERE {
    ?class concept "py:Class" .
    ?class name ?className .
    ?init concept "py:Method" .
    ?init name "__init__" .
    ?init definedIn ?class .
    ?init parameterCount ?paramCount
}
```

2. **Get parameter details**:
```reql
SELECT ?param ?paramName ?hasDefault WHERE {
    <init_id> hasParameter ?param .
    ?param name ?paramName .
    OPTIONAL { ?param hasDefaultValue ?hasDefault }
}
```

3. **Find factory methods**:
```reql
SELECT ?method ?methodName WHERE {
    <class_id> concept "py:Class" .
    ?method concept "py:Method" .
    ?method definedIn <class_id> .
    ?method hasDecorator ?decorator .
    ?decorator name "classmethod"
}
```

## References

- **Book**: "Refactoring to Patterns" by Joshua Kerievsky (2004)
- **Pattern**: Chain Constructors (Chapter 6)
- **Related**: Martin Fowler's "Refactoring" (2nd Edition)

## License

Part of the RETER Logical Thinking Server
