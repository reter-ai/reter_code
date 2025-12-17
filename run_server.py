#!/usr/bin/env python
"""
Wrapper script to run RETER Logical Thinking MCP Server

This ensures proper Python path setup for RETER imports.
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run server
from logical_thinking_server.server import main

if __name__ == "__main__":
    main()
