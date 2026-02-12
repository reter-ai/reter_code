"""
Reter Code MCP Server

AI-powered code reasoning server using RETER engine.
Remote-only mode - all operations go through ReterClient via ZeroMQ.
"""

import logging
import os
import signal
import sys
from typing import Any, Optional, List

from fastmcp import FastMCP

from .logging_config import configure_logger_for_debug_trace
logger = configure_logger_for_debug_trace(__name__)

# Sampling handler imports
from mcp.types import SamplingMessage, CreateMessageRequestParams, CreateMessageResult, TextContent

from .services import (
    DocumentationProvider,
    ResourceRegistrar,
    ToolRegistrar
)
from .server.reter_client import ReterClient
from .server.config import ClientConfig


async def anthropic_sampling_handler(
    messages: List[SamplingMessage],
    params: CreateMessageRequestParams,
    context: Any
) -> CreateMessageResult:
    """
    Sampling handler that uses Anthropic's API with prompt caching.

    Uses Anthropic's prompt caching feature to cache system prompts,
    reducing latency and costs for repeated queries with the same prompts.

    Environment variables:
        ANTHROPIC_API_KEY: Required. Your Anthropic API key.
        ANTHROPIC_MODEL_NAME: Optional. Model to use (default: claude-opus-4-5-20251101)
        ANTHROPIC_MAX_TOKENS: Optional. Max tokens for response (default: 1024)
    """
    try:
        import anthropic
    except ImportError:
        raise ValueError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Get configurable settings from environment
    model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-opus-4-5-20251101")
    default_max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))

    client = anthropic.Anthropic(api_key=api_key)

    # Convert MCP messages to Anthropic format
    anthropic_messages = []
    for msg in messages:
        content = msg.content
        if hasattr(content, 'text'):
            text = content.text
        else:
            text = str(content)
        anthropic_messages.append({
            "role": msg.role,
            "content": text
        })

    # Build system prompt with cache_control for prompt caching
    # Prompt caching requires minimum 1024 tokens for Sonnet/Opus 4
    # If prompt is shorter, caching is silently skipped (no error)
    system_prompt = params.systemPrompt or ""
    if system_prompt:
        system_content = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    else:
        system_content = []

    # Call Anthropic API with prompt caching
    response = client.messages.create(
        model=model_name,
        max_tokens=params.maxTokens or default_max_tokens,
        system=system_content,
        messages=anthropic_messages
    )

    # Log cache performance (helps track caching effectiveness)
    usage = response.usage
    if hasattr(usage, 'cache_read_input_tokens') or hasattr(usage, 'cache_creation_input_tokens'):
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
        cache_write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        if cache_read > 0 or cache_write > 0:
            print(f"[CACHE] read={cache_read}, write={cache_write}, input={usage.input_tokens}",
                  file=sys.stderr)

    # Extract response text
    response_text = response.content[0].text if response.content else ""

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model=response.model,
        stopReason=response.stop_reason
    )


class ReterCodeServer:
    """
    RETER Code MCP Server - Remote-only mode.

    All operations go through ReterClient via ZeroMQ to the RETER server.

    ::: This is-in-layer Presentation-Layer.
    ::: This is a model-context-protocol-server.
    ::: This is-in-process Model-Context-Protocol-Server-Process.
    ::: This is a process-entry-point.
    ::: This is stateless.
    ::: This is a ipc-proxy.
    """

    def __init__(self):
        import time as _time
        _init_start = _time.time()

        # Documentation provider (static content, no RETER needed)
        self.doc_provider = DocumentationProvider()

        # Resource registrar for documentation resources
        self.resource_registrar = ResourceRegistrar(self.doc_provider)

        # Create ReterClient for remote RETER server connection
        self.reter_client = None
        server_host = os.environ.get("RETER_SERVER_HOST")
        project_root = os.environ.get("RETER_PROJECT_ROOT")
        if not project_root:
            project_root = os.getcwd()
            # Set env var so logging_config can find it
            os.environ["RETER_PROJECT_ROOT"] = project_root

        try:
            if server_host:
                # Explicit server connection
                query_port = int(os.environ.get("RETER_SERVER_QUERY_PORT", "5555"))
                endpoint = f"tcp://{server_host}:{query_port}"
                self.reter_client = ReterClient.for_endpoint(endpoint)
                logger.info(f"Connecting to RETER server at {endpoint}")
            else:
                # Use discovery based on project root
                client_config = ClientConfig.for_project(project_root)
                self.reter_client = ReterClient(config=client_config)
                logger.info(f"Using project discovery for {project_root}")
        except Exception as e:
            logger.error(f"Failed to create ReterClient: {e}")
            # In remote-only mode, we continue but tools will return errors
            self.reter_client = None

        # Tool registrar - passes ReterClient for remote operations
        # instance_manager and persistence are None in remote-only mode
        _t = _time.time()
        self.tool_registrar = ToolRegistrar(
            reter_ops=None,
            persistence=None,
            instance_manager=None,
            default_manager=None,
            reter_client=self.reter_client
        )
        logger.info("[TIMING] ToolRegistrar created in %.3fs", _time.time() - _t)

        # Initialize FastMCP with Anthropic sampling handler
        self.app = FastMCP(
            "reter",
            instructions="""Reter Code is an AI-powered code reasoning MCP server.

CRITICAL: Call `session(action="context")` at the START of every conversation to restore your reasoning state.

Key tools:
- `thinking` - Record reasoning steps with design doc sections (context, goals, design, alternatives, risks, implementation, tasks)
- `session` - Manage reasoning sessions (start, context, end)
- `diagram` - Generate UML diagrams (gantt, class, sequence)
- `code_inspection` - Python/JS/C#/C++ code analysis
- `recommender` - Refactoring and test coverage recommendations

Design doc workflow:
1. thinking(section="context") - Document problem/background
2. thinking(section="goals") - Define objectives
3. thinking(section="design") - Propose solution
4. thinking(section="tasks", operations={task:{...}}) - Create work items

Always use the `thinking` tool when analyzing problems or making decisions.""",
            website_url="https://reter.ai",
            sampling_handler=anthropic_sampling_handler,
            sampling_handler_behavior="fallback"
        )
        logger.info("[TIMING] FastMCP created in %.3fs", _time.time() - _init_start)

    def run(self):
        """
        Start the MCP server with graceful shutdown support.
        """
        import time as _time

        def signal_handler(signum, frame):
            """Handle shutdown signals gracefully"""
            sig_name = signal.Signals(signum).name
            logger.warning("Received %s, initiating graceful shutdown...", sig_name)
            sys.exit(0)

        # Register signal handlers (SIGINT = Ctrl+C, SIGTERM = kill command)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            _init_start = _time.time()
            logger.info("Reter Code MCP Server starting (remote-only mode)...")

            # Check ReterClient connection
            if self.reter_client is None:
                logger.warning("ReterClient not connected - tools will return errors")
            else:
                # Verify connection to RETER server
                try:
                    status = self.reter_client.get_status()
                    if status.get("success"):
                        logger.info("Connected to RETER server successfully")
                    else:
                        logger.warning("RETER server status check failed: %s", status.get("error"))
                except Exception as e:
                    logger.warning("Could not verify RETER server connection: %s", e)

            # Register resources and tools
            _t = _time.time()
            self.resource_registrar.register_all_resources(self.app)
            self.tool_registrar.register_all_tools(self.app)
            logger.info("[TIMING] Tools registered in %.3fs", _time.time() - _t)
            logger.info("All tools registered (total init: %.3fs)", _time.time() - _init_start)

            # Run the FastMCP app (blocking)
            self.app.run()

        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt received, shutting down...")

        except Exception as e:
            logger.error("Unexpected error during server operation: %s", e)
            raise

        finally:
            # Close ReterClient connection
            if self.reter_client is not None:
                try:
                    self.reter_client.close()
                except Exception:
                    pass

            try:
                logger.info("Server shutdown complete")
            except:
                pass


def create_server():
    """Factory function to create server instance"""
    return ReterCodeServer()


_UVX_FROM = "git+https://github.com/reter-ai/reter_code"


def _print_setup_instructions():
    """Print setup instructions when run directly from terminal."""
    print(f"""
Reter Code - AI-powered code reasoning MCP server

  Install (persistent):
    uv tool install --from {_UVX_FROM} reter_code

  After install, start the server:
    cd /path/to/your/project
    reter

  Add MCP to Claude Code:
    claude mcp add reter -- reter_code --stdio

  Or without installing (uvx):
    claude mcp add reter -- uvx --from {_UVX_FROM} reter_code --stdio

  Config locations:
    macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
    Windows: %APPDATA%\\Claude\\claude_desktop_config.json
    Linux:   ~/.config/Claude/claude_desktop_config.json
""")


def main():
    """Main entry point.

    When run from a terminal (TTY), prints setup instructions.
    When run with --stdio (by an MCP host), starts the MCP server.
    """
    if "--stdio" in sys.argv:
        server = create_server()
        server.run()
    else:
        _print_setup_instructions()


if __name__ == "__main__":
    main()
