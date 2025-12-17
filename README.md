# Codeine

AI-powered code reasoning MCP server.

## Installation

### One-liner with uvx

```bash
uvx --from git+https://github.com/codeine-ai/codeine --find-links https://raw.githubusercontent.com/codeine-ai/reter/main/reter_core/index.html codeine
```

### Install with uv/pip

```bash
uv pip install git+https://github.com/codeine-ai/codeine --find-links https://raw.githubusercontent.com/codeine-ai/reter/main/reter_core/index.html
```

Or with pip:

```bash
pip install git+https://github.com/codeine-ai/codeine --find-links https://raw.githubusercontent.com/codeine-ai/reter/main/reter_core/index.html
```

### Local development

```bash
git clone https://github.com/codeine-ai/codeine.git
cd codeine
uv pip install -e . --find-links https://raw.githubusercontent.com/codeine-ai/reter/main/reter_core/index.html
```

> **Note**: The `--find-links` flag is required because `reter-core` (the C++ engine) is distributed as platform-specific wheels from a private index.

## Usage

### Run the server

```bash
codeine
```

Or:

```bash
python -m codeine
```

### Configure with Claude Code

Add to your project (saves to `.claude/settings.local.json`):

```bash
claude mcp add codeine -s project -e ANTHROPIC_API_KEY=your-api-key -- uvx --from git+https://github.com/codeine-ai/codeine --find-links https://raw.githubusercontent.com/codeine-ai/reter/main/reter_core/index.html codeine
```

Or add globally (saves to `~/.claude/settings.json`):

```bash
claude mcp add codeine -e ANTHROPIC_API_KEY=your-api-key -- uvx --from git+https://github.com/codeine-ai/codeine --find-links https://raw.githubusercontent.com/codeine-ai/reter/main/reter_core/index.html codeine
```

> **Note**: On first run, add `"timeout": 120000` to the server config in settings file (first startup downloads ~400MB of dependencies).

### Configure with Claude Desktop

Add to your Claude Desktop config:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RETER_PROJECT_ROOT` | Path to project for auto-loading code | Auto-detected from CWD |
| `ANTHROPIC_API_KEY` | API key for sampling handler | - |

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
