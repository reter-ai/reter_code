# Reter Code

AI-powered code reasoning MCP server.

## Installation

### Step 1: Pre-cache dependencies and sync project (run from your project directory)

```bash
uvx --from git+https://github.com/reter-ai/reter_code --find-links https://raw.githubusercontent.com/reter-ai/reter/main/reter_core/index.html reter_code
```

This will:
1. Download dependencies (~400MB, cached for future runs)
2. Sync your project files to the RETER index
3. Exit automatically

Run this from your project root directory.

### Step 2: Add to Claude Code

```bash
claude mcp add reter_code -s user -e ANTHROPIC_API_KEY=your-api-key -- uvx --from git+https://github.com/reter-ai/reter_code --find-links https://raw.githubusercontent.com/reter-ai/reter/main/reter_core/index.html reter_code
```

Now Claude starts fast because everything is cached.

---

## Configure with Claude Desktop

**Config file location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "reter_code": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/reter-ai/reter_code",
        "--find-links", "https://raw.githubusercontent.com/reter-ai/reter/main/reter_core/index.html",
        "reter_code"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "your-api-key"
      },
      "timeout": 120000
    }
  }
}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RETER_PROJECT_ROOT` | Path to project for auto-loading code | Auto-detected from CWD |
| `ANTHROPIC_API_KEY` | API key for sampling handler | - |

---

## Tools

### Design Docs (Core Workflow)

| Tool | Description |
|------|-------------|
| `session` | Session lifecycle - **call `context` first to restore state** |
| `thinking` | Record reasoning with sections (context, goals, design, tasks) |
| `diagram` | Visualize design docs, Gantt charts, UML diagrams |

### Code Analysis

| Tool | Description |
|------|-------------|
| `code_inspection` | Multi-language code analysis (Python, JS, C#, C++) |
| `recommender` | Refactoring and test coverage recommendations |
| `natural_language_query` | Ask questions about code in plain English |

### Semantic Search (RAG)

| Tool | Description |
|------|-------------|
| `semantic_search` | Find code by meaning, not just keywords |
| `find_similar_clusters` | Detect code duplication patterns |

### Knowledge Management

| Tool | Description |
|------|-------------|
| `instance_manager` | Manage RETER instances and sources |
| `add_knowledge` | Add external code/ontologies to RETER |

---

## Supported Languages

- Python (.py)
- C# (.cs)
- C++ (.cpp, .hpp, .h, .cc)
- JavaScript (.js)

## Features

- **Design Docs Workflow** - Structured reasoning with sections (context, goals, design, alternatives, risks, tasks)
- **Multi-language Analysis** - Python, JavaScript, C#, C++ code inspection
- **Session Persistence** - Thoughts, tasks, milestones tracked across conversations
- **Semantic Search (RAG)** - Find code by meaning with vector similarity
- **UML Diagrams** - Class hierarchy, sequence, call graphs, dependencies
- **Gantt Charts** - Task scheduling with critical path analysis
- **Refactoring Recommendations** - 58 code smell detectors
- **Test Coverage Analysis** - Find untested code paths

## License

MIT License
