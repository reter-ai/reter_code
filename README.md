# Codeine

AI-powered code reasoning MCP server.

## Installation

### Step 1: Pre-cache dependencies and sync project (run from your project directory)

```bash
uvx --from git+https://github.com/codeine-ai/codeine --find-links https://raw.githubusercontent.com/codeine-ai/reter/main/reter_core/index.html codeine
```

This will:
1. Download dependencies (~400MB, cached for future runs)
2. Sync your project files to the RETER index
3. Exit automatically

Run this from your project root directory.

### Step 2: Add to Claude Code

```bash
claude mcp add codeine -s user -e ANTHROPIC_API_KEY=your-api-key -- uvx --from git+https://github.com/codeine-ai/codeine --find-links https://raw.githubusercontent.com/codeine-ai/reter/main/reter_core/index.html codeine
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
    "codeine": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/codeine-ai/codeine",
        "--find-links", "https://raw.githubusercontent.com/codeine-ai/reter/main/reter_core/index.html",
        "codeine"
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

### Thinking & Reasoning

| Tool | Description |
|------|-------------|
| `thinking` | Record reasoning steps, analysis, decisions |
| `session` | Manage reasoning sessions (start, context, end) |
| `items` | Query and manage thoughts, requirements, tasks |
| `project` | Project analytics (health, critical path, impact) |

### Code Analysis

| Tool | Description |
|------|-------------|
| `code_inspection` | Python code analysis (26 actions) |
| `recommender` | Refactoring and test coverage recommendations |
| `diagram` | Generate UML diagrams (class, sequence, etc.) |

### Instance Management

| Tool | Description |
|------|-------------|
| `instance_manager` | Manage RETER instances and sources |

---

## Supported Languages

- Python (.py)
- C# (.cs)
- C++ (.cpp, .hpp, .h, .cc)
- JavaScript (.js)

## Features

- Logical reasoning with RETER engine
- Multi-language code analysis
- Session-based thinking with persistence
- Requirements and task tracking
- UML diagram generation
- Refactoring recommendations
- Test coverage analysis

## License

MIT License
