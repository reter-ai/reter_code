# Python Basic Plugin - User Guide

**Version**: 1.0.0
**Plugin**: python_basic
**Category**: Python Code Analysis

## Overview

The Python Basic Plugin provides fundamental code analysis capabilities for Python codebases. It offers 10 essential tools for exploring and understanding Python code structure, including module listings, class hierarchies, method signatures, and dependency analysis.

## Key Features

✅ Module and file discovery
✅ Class and function enumeration
✅ Detailed class descriptions
✅ Method signature extraction
✅ Docstring retrieval
✅ Usage tracking (where code is called)
✅ Inheritance analysis
✅ Class hierarchy visualization
✅ Dependency graph generation

## Tools Reference

### 1. list_modules
**Purpose**: List all Python modules in the codebase

**Parameters**:
- `instance_name` (required): RETER instance name

**Returns**: List of module names with file paths

**Example**:
```json
{
  "instance_name": "main",
  "params": "{}"
}
```

**Use Cases**:
- Understanding codebase structure
- Finding specific modules
- Initial codebase exploration

---

### 2. list_classes
**Purpose**: List all classes in the codebase or a specific module

**Parameters**:
- `instance_name` (required): RETER instance name
- `module_name` (optional): Filter by specific module

**Returns**: List of class names with locations

**Example**:
```json
{
  "instance_name": "main",
  "params": "{\"module_name\": \"services.plugin_manager\"}"
}
```

**Use Cases**:
- Enumerating all classes in project
- Finding classes in a specific module
- Understanding class organization

---

### 3. describe_class
**Purpose**: Get detailed description of a class including all methods and parameters

**Parameters**:
- `instance_name` (required): RETER instance name
- `class_name` (required): Name of the class to describe

**Returns**: Class description with methods, parameters, and docstrings

**Example**:
```json
{
  "instance_name": "main",
  "params": "{\"class_name\": \"PluginManager\"}"
}
```

**Use Cases**:
- Understanding class structure
- Finding available methods
- API exploration

---

### 4. find_usages
**Purpose**: Find where a class or method is used (called) in the codebase

**Parameters**:
- `instance_name` (required): RETER instance name
- `target_name` (required): Name of class or method
- `target_type` (required): "class" or "method"

**Returns**: List of locations where target is used

**Example**:
```json
{
  "instance_name": "main",
  "params": "{\"target_name\": \"PluginManager\", \"target_type\": \"class\"}"
}
```

**Use Cases**:
- Impact analysis before refactoring
- Understanding code dependencies
- Finding all callers of a method

---

### 5. find_subclasses
**Purpose**: Find all subclasses of a specified class

**Parameters**:
- `instance_name` (required): RETER instance name
- `class_name` (required): Base class name

**Returns**: List of subclasses

**Example**:
```json
{
  "instance_name": "main",
  "params": "{\"class_name\": \"AnalysisPlugin\"}"
}
```

**Use Cases**:
- Understanding inheritance hierarchies
- Finding plugin implementations
- Analyzing class extensions

---

### 6. get_method_signature
**Purpose**: Get the signature of a method including parameters and return type

**Parameters**:
- `instance_name` (required): RETER instance name
- `class_name` (required): Class containing the method
- `method_name` (required): Method name

**Returns**: Method signature with parameter types and return type

**Example**:
```json
{
  "instance_name": "main",
  "params": "{\"class_name\": \"PluginManager\", \"method_name\": \"load_plugin\"}"
}
```

**Use Cases**:
- Understanding method interfaces
- API documentation
- Type checking

---

### 7. get_docstring
**Purpose**: Get the docstring of a class or method

**Parameters**:
- `instance_name` (required): RETER instance name
- `entity_name` (required): Class or method name
- `entity_type` (required): "class" or "method"

**Returns**: Docstring content

**Example**:
```json
{
  "instance_name": "main",
  "params": "{\"entity_name\": \"PluginManager\", \"entity_type\": \"class\"}"
}
```

**Use Cases**:
- Reading documentation
- Understanding code intent
- Documentation generation

---

### 8. list_functions
**Purpose**: List top-level functions in the codebase or a specific module

**Parameters**:
- `instance_name` (required): RETER instance name
- `module_name` (optional): Filter by specific module

**Returns**: List of function names with locations

**Example**:
```json
{
  "instance_name": "main",
  "params": "{\"module_name\": \"utils\"}"
}
```

**Use Cases**:
- Finding utility functions
- Understanding module structure
- API discovery

---

### 9. get_class_hierarchy
**Purpose**: Get the class hierarchy showing parent and child classes

**Parameters**:
- `instance_name` (required): RETER instance name
- `class_name` (optional): Root class for hierarchy

**Returns**: Hierarchical tree of classes

**Example**:
```json
{
  "instance_name": "main",
  "params": "{\"class_name\": \"AnalysisPlugin\"}"
}
```

**Use Cases**:
- Visualizing inheritance
- Understanding OOP structure
- Architecture analysis

---

### 10. analyze_dependencies
**Purpose**: Analyze the dependency graph of the codebase

**Parameters**:
- `instance_name` (required): RETER instance name

**Returns**: Module dependency graph with imports

**Example**:
```json
{
  "instance_name": "main",
  "params": "{}"
}
```

**Use Cases**:
- Understanding module relationships
- Finding circular dependencies
- Architecture review

---

## Workflow Examples

### Example 1: Exploring a New Codebase

```
1. list_modules() - See all available modules
2. list_classes() - Enumerate all classes
3. describe_class("MainClass") - Understand key class
4. get_class_hierarchy("MainClass") - See inheritance
5. analyze_dependencies() - Understand module structure
```

### Example 2: Understanding a Specific Class

```
1. find_subclasses("BaseClass") - Find implementations
2. describe_class("ConcreteClass") - See methods
3. get_method_signature("ConcreteClass", "key_method") - Check signature
4. find_usages("key_method", "method") - See where it's used
```

### Example 3: Impact Analysis Before Refactoring

```
1. find_usages("ClassToChange", "class") - Find all usage sites
2. find_subclasses("ClassToChange") - Find derived classes
3. get_class_hierarchy("ClassToChange") - Understand inheritance
4. analyze_dependencies() - Check module dependencies
```

## Best Practices

### 1. Always Start with Knowledge Loading
Before using any tools, load your Python codebase into RETER:

```python
add_knowledge(
    instance_name="main",
    source="/path/to/your/code.py",
    type="python"
)
```

### 2. Use Consistent Instance Names
Maintain consistent RETER instance names throughout your analysis session:
- Use `"main"` for primary analysis
- Use descriptive names for multiple projects: `"project_a"`, `"project_b"`

### 3. Leverage Queries for Complex Analysis
For advanced filtering and analysis, use REQL queries after loading code:

```python
quick_query(
    instance_name="main",
    query="SELECT ?class ?method WHERE (class ?class) (method ?class ?method)",
    type="reql"
)
```

### 4. Combine with Python Advanced Plugin
For deeper analysis, use python_basic tools for exploration, then python_advanced for:
- Code smell detection
- Refactoring opportunities
- Architecture analysis

## Troubleshooting

### Issue: "No modules found"
**Cause**: Python code not loaded into RETER
**Solution**: Use `add_knowledge()` to load your Python files first

### Issue: "Class not found"
**Cause**: Class name typo or class not in loaded modules
**Solution**: Use `list_classes()` to see available classes

### Issue: "Empty results"
**Cause**: Code not analyzed yet
**Solution**: Ensure `add_knowledge()` completed successfully

### Issue: "Instance not found"
**Cause**: Using wrong instance name
**Solution**: Check instance name consistency across calls

## Performance Tips

1. **Load Once**: Load code into RETER once, then run multiple queries
2. **Module Filtering**: Use `module_name` parameter to scope queries
3. **Incremental Analysis**: Analyze one subsystem at a time for large codebases
4. **Instance Reuse**: Reuse RETER instances across analysis sessions

## Related Documentation

- **Python Advanced Plugin**: For advanced analysis capabilities
- **REQL Reference**: For custom query creation
- **Plugin System Guide**: For plugin development

## Support

For issues or questions about the Python Basic Plugin:
- Check the main plugin documentation
- Review REQL query patterns
- Test with simple examples first

---

**Note**: This plugin provides **basic** Python analysis. For advanced features like code smell detection, refactoring analysis, and architectural insights, use the **python_advanced** plugin.
