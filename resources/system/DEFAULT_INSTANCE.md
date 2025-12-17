# Default Instance - Auto-Syncing Project Analysis

## Overview

The **"default"** instance is a special RETER instance that automatically synchronizes with your project's Python files. It always appears in `list_all_instances()` and provides seamless project-wide code analysis without manual file management.

## Key Features

### 1. Automatic File Synchronization
- **Detects new files** → Automatically loaded
- **Detects modifications** → MD5-based change detection, automatic reload
- **Detects deletions** → Automatically forgotten from knowledge base
- **Runs on every access** → Always up-to-date with filesystem

### 2. Environment-Based Configuration
```bash
# Required: Project directory to monitor
export RETER_PROJECT_ROOT=/path/to/your/project

# Optional: Glob patterns to exclude
export RETER_PROJECT_EXCLUDE="test_*.py,**/tests/*,**/__pycache__/*"
```

### 3. Always Available
The default instance always appears in `list_all_instances()` with one of these statuses:
- `not_configured` - RETER_PROJECT_ROOT not set
- `configured` - RETER_PROJECT_ROOT set, not yet accessed
- `available` - Snapshot exists, not loaded in memory
- `loaded` - Active in memory with synced files

## Configuration

### RETER_PROJECT_ROOT (Required)
Path to your project directory. All Python files in this directory will be automatically analyzed.

**Requirements**:
- Must be an absolute path
- Must exist and be a directory
- Will be scanned recursively for `.py` files

**Example**:
```bash
export RETER_PROJECT_ROOT=/home/user/my-python-project
```

### RETER_PROJECT_EXCLUDE (Optional)
Comma-separated glob patterns to exclude from analysis.

**Supported Patterns**:
- `test_*.py` - Files starting with "test_"
- `**/tests/*` - All files in any "tests" directory
- `**/__pycache__/*` - All files in any "__pycache__" directory
- `*.bak` - All backup files

**Example**:
```bash
export RETER_PROJECT_EXCLUDE="test_*.py,**/tests/*,**/__pycache__/*,*.bak"
```

## How It Works

### File Change Detection

The default instance uses **MD5-based content hashing** to detect file changes:

1. **Source ID Format**: `{md5_hash}|{relative_path}`
   - Example: `e2a8065dc214e3db0a6a2d57a397d0d5|src/analyzer.py`

2. **On Each Access**:
   - Scan RETER_PROJECT_ROOT for `.py` files
   - Calculate MD5 hash of each file
   - Compare against existing sources in RETER
   - Sync changes automatically

3. **Sync Operations**:
   - **New file**: Load with `load_python_file()`
   - **Modified file**: Forget old version, reload new version
   - **Deleted file**: Forget from RETER

### Automatic Snapshot Management

- **Auto-save**: Saved to `.reter/default.reter` after changes
- **Auto-load**: Loaded from snapshot on first access (if exists)
- **Lazy loading**: Only loaded when first used

## Usage Examples

### Basic Usage
```python
# Just use instance_name="default" in any tool
result = quick_query(
    instance_name="default",
    query="Class(?x, ?name)"
)

# Files from RETER_PROJECT_ROOT are automatically loaded and synced
```

### Querying Project Code
```python
# Find all classes in your project
classes = quick_query(
    instance_name="default",
    query="Class(?class_id, ?class_name)"
)

# Find all methods in a specific class
methods = quick_query(
    instance_name="default",
    query="Method(?method_id, ?method_name) also definedIn(?method_id, 'MyClass')"
)

# Find function call relationships
calls = quick_query(
    instance_name="default",
    query="calls(?caller, ?callee)"
)
```

### Viewing Loaded Files
```python
# List all source files
sources = list_sources(instance_name="default")

# Sources will have format: md5_hash|relative/path.py
# Example: ["e2a8065dc2...|src/main.py", "a1b2c3d4e5...|lib/utils.py"]
```

### Adding Domain Knowledge
```python
# Add ontology rules to default instance
add_knowledge(
    instance_name="default",
    source="APIEndpoint is_a Class.",
    type="ontology"
)

# Rules apply to all loaded Python code
```

## Important Notes

### When to Use Default Instance

✅ **Use "default" when**:
- Analyzing your entire project codebase
- Want automatic file sync without manual loading
- Need persistent project-wide knowledge base
- Working with frequently modified code

❌ **Don't use "default" when**:
- Analyzing external libraries (use named instances)
- Need isolated temporary analysis
- Want full control over loaded files
- Working with non-Python files

### Forgotten Files Are Reloaded

⚠️ **Important**: If you call `forget_source()` on a file in the default instance, it will be **reloaded on next access** if the file still exists in RETER_PROJECT_ROOT.

To permanently remove a file from default instance:
1. Delete the file from filesystem, OR
2. Add it to RETER_PROJECT_EXCLUDE

### Performance Considerations

- **File Scanning**: ~50-100ms for 100 files (MD5 calculation)
- **Sync Operation**: ~10ms per changed file
- **Runs on every access**: Small overhead but guarantees freshness

For large projects (1000+ files), consider:
- Using RETER_PROJECT_EXCLUDE to filter unnecessary files
- Using named instances for isolated analysis
- Future: Filesystem watchers (not yet implemented)

## Comparison: Default vs Named Instances

| Feature | Default Instance | Named Instances |
|---------|------------------|-----------------|
| File Sync | Automatic | Manual |
| Configuration | Environment vars | Programmatic |
| Always Listed | Yes | Only if exists |
| Source Format | MD5\|path | Timestamp or custom |
| Best For | Project analysis | Isolated tasks |
| Persistence | Auto-snapshot | Auto-snapshot |

## Troubleshooting

### "Default instance shows 'not_configured'"
**Solution**: Set RETER_PROJECT_ROOT environment variable before starting server.

```bash
export RETER_PROJECT_ROOT=/path/to/project
# Restart server
```

### "Files not being detected"
**Solutions**:
1. Check RETER_PROJECT_ROOT points to correct directory
2. Verify files are `.py` extension
3. Check files aren't excluded by RETER_PROJECT_EXCLUDE
4. Access the instance to trigger sync (it's lazy-loaded)

### "Old file versions still in RETER"
**Solution**: The instance syncs on access. Make a query or call any tool with `instance_name="default"` to trigger sync.

### "How to reset default instance"
**Options**:
1. Delete `.reter/default.reter` snapshot file
2. Access instance - it will rebuild from project files
3. Or: Change RETER_PROJECT_ROOT to new directory

## Advanced Usage

### Combining with Named Instances

```python
# Load external library in named instance
add_python_directory(
    instance_name="django-analysis",
    directory="/usr/lib/python/django"
)

# Use default for your project code
# Use "django-analysis" for library code
# Keep them separate and clean
```

### Incremental Knowledge Building

```python
# 1. Default instance loads project code automatically
# 2. Add domain-specific rules
add_knowledge(
    instance_name="default",
    source="""
        APIEndpoint is_a Class.
        if hasDecorator(?method, 'route') then APIEndpoint(?method).
    """,
    type="ontology"
)

# 3. Query combined knowledge (code + rules)
endpoints = quick_query(
    instance_name="default",
    query="APIEndpoint(?endpoint)"
)
```

### Monitoring File Changes

```python
# Before making changes
sources_before = list_sources(instance_name="default")

# Make changes to project files...

# Trigger sync
sources_after = list_sources(instance_name="default")

# Compare to see what changed
# New sources = new/modified files
# Missing sources = deleted files
```

## Best Practices

1. **Set exclusions early**: Configure RETER_PROJECT_EXCLUDE before first use
2. **Use descriptive queries**: Default instance has many files, query specifically
3. **Don't mix contexts**: Use named instances for external/library code analysis
4. **Trust the sync**: Files are automatically kept up-to-date, no manual intervention needed
5. **Delete unwanted files**: Remove from filesystem to remove from default instance

## Summary

The default instance provides **zero-configuration project-wide code analysis** with automatic file synchronization. Simply set RETER_PROJECT_ROOT, and all Python files in your project are automatically analyzed and kept up-to-date. Perfect for project-wide queries, code understanding, and persistent knowledge accumulation.

**Key Takeaway**: Use `instance_name="default"` anywhere you'd use a regular instance name, and enjoy automatic file management!
