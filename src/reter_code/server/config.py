"""
RETER Server Configuration.

Configuration dataclass and environment variable support for RETER ZeroMQ server.
Includes project-based discovery for multi-project support.

::: This is-in-layer Protocol-Layer.
::: This is-in-component Message-Protocol.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import json
import os
import sys
import time


# =============================================================================
# Default Values
# =============================================================================

DEFAULT_HOST = "127.0.0.1"
DEFAULT_QUERY_PORT = 5555
DEFAULT_EVENT_PORT = 5556
DEFAULT_TIMEOUT_MS = 30000  # 30 seconds
DEFAULT_RECONNECT_INTERVAL_MS = 1000  # 1 second

# Discovery file location within project
DISCOVERY_DIR = ".reter_code"
DISCOVERY_FILE = "server.json"

# Port range for automatic allocation (5555-5654 = 50 projects)
PORT_RANGE_START = 5555
PORT_RANGE_SIZE = 100


def get_default_ipc_path() -> str:
    """Get platform-appropriate IPC path."""
    if sys.platform == "win32":
        # Windows doesn't support Unix domain sockets in the same way
        # Use TCP for Windows
        return ""
    else:
        return "/tmp/reter"


def get_discovery_file_path(project_root: Path) -> Path:
    """Get the discovery file path for a project."""
    # Ensure project_root is a Path, not a string
    if isinstance(project_root, str):
        project_root = Path(project_root)
    return project_root / DISCOVERY_DIR / DISCOVERY_FILE


@dataclass
class ServerDiscovery:
    """Discovery information written by server, read by client.

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a value-object.
    ::: This is stateless.
    ::: This is serializable.
    """
    project_root: str
    query_endpoint: str
    event_endpoint: str
    query_port: int
    event_port: int
    host: str
    pid: int
    started_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_root": self.project_root,
            "query_endpoint": self.query_endpoint,
            "event_endpoint": self.event_endpoint,
            "query_port": self.query_port,
            "event_port": self.event_port,
            "host": self.host,
            "pid": self.pid,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerDiscovery":
        return cls(
            project_root=data["project_root"],
            query_endpoint=data["query_endpoint"],
            event_endpoint=data["event_endpoint"],
            query_port=data["query_port"],
            event_port=data["event_port"],
            host=data["host"],
            pid=data["pid"],
            started_at=data["started_at"],
        )

    def save(self, project_root: Path) -> None:
        """Write discovery file to project directory."""
        # Ensure project_root is a Path, not a string
        if isinstance(project_root, str):
            project_root = Path(project_root)
        discovery_dir = project_root / DISCOVERY_DIR
        discovery_dir.mkdir(parents=True, exist_ok=True)
        discovery_file = discovery_dir / DISCOVERY_FILE
        with open(discovery_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, project_root: Path) -> Optional["ServerDiscovery"]:
        """Read discovery file from project directory."""
        discovery_file = get_discovery_file_path(project_root)
        if not discovery_file.exists():
            return None
        try:
            with open(discovery_file, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return None

    @classmethod
    def remove(cls, project_root: Path) -> bool:
        """Remove discovery file (called on server shutdown)."""
        discovery_file = get_discovery_file_path(project_root)
        if discovery_file.exists():
            discovery_file.unlink()
            return True
        return False

    def is_server_alive(self) -> bool:
        """Check if the server process is still running."""
        try:
            import psutil
            return psutil.pid_exists(self.pid)
        except ImportError:
            # Fallback without psutil - try to connect
            return True  # Assume alive, let connection fail if not


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ServerConfig:
    """Configuration for RETER ZeroMQ server.

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a value-object.
    ::: This is stateless.
    ::: This is serializable.

    Supports environment variables:
    - RETER_HOST: Server host (default: 127.0.0.1)
    - RETER_QUERY_PORT: Query socket port (default: 5555, or auto from project)
    - RETER_EVENT_PORT: Event socket port (default: 5556, or auto from project)
    - RETER_TIMEOUT_MS: Request timeout in milliseconds (default: 30000)
    - RETER_USE_IPC: Use IPC instead of TCP on Unix (default: false)
    - RETER_IPC_PATH: IPC socket path prefix (default: /tmp/reter)
    - RETER_CONSOLE_ENABLED: Enable rich console output (default: true)
    - RETER_PROJECT_ROOT: Project root directory for discovery
    """

    # Project configuration (for multi-project discovery)
    project_root: Optional[Path] = field(default=None)

    # Network configuration
    host: str = field(default_factory=lambda: os.environ.get("RETER_HOST", DEFAULT_HOST))
    query_port: int = field(default_factory=lambda: int(os.environ.get("RETER_QUERY_PORT", DEFAULT_QUERY_PORT)))
    event_port: int = field(default_factory=lambda: int(os.environ.get("RETER_EVENT_PORT", DEFAULT_EVENT_PORT)))

    # IPC configuration (Unix only)
    use_ipc: bool = field(default_factory=lambda: os.environ.get("RETER_USE_IPC", "false").lower() == "true")
    ipc_path: str = field(default_factory=lambda: os.environ.get("RETER_IPC_PATH", get_default_ipc_path()))

    # Timeout configuration
    timeout_ms: int = field(default_factory=lambda: int(os.environ.get("RETER_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)))
    reconnect_interval_ms: int = field(default_factory=lambda: int(os.environ.get(
        "RETER_RECONNECT_INTERVAL_MS", DEFAULT_RECONNECT_INTERVAL_MS
    )))

    # Console configuration
    console_enabled: bool = field(default_factory=lambda: os.environ.get("RETER_CONSOLE_ENABLED", "true").lower() == "true")
    console_log_queries: bool = field(default_factory=lambda: os.environ.get("RETER_LOG_QUERIES", "true").lower() == "true")
    console_show_progress: bool = field(default_factory=lambda: os.environ.get("RETER_SHOW_PROGRESS", "true").lower() == "true")

    # Server configuration
    max_workers: int = field(default_factory=lambda: int(os.environ.get("RETER_MAX_WORKERS", "4")))
    heartbeat_interval_ms: int = field(default_factory=lambda: int(os.environ.get("RETER_HEARTBEAT_MS", "5000")))

    def _build_endpoint(self, name: str, port: int, bind: bool = False) -> str:
        """Build socket endpoint URL.

        Args:
            name: Socket name suffix (e.g., "query", "events")
            port: TCP port number
            bind: If True, use wildcard host for binding

        Returns:
            Endpoint URL (ipc:// or tcp://)
        """
        if self.use_ipc and sys.platform != "win32":
            return f"ipc://{self.ipc_path}-{name}"
        host = "*" if bind else self.host
        return f"tcp://{host}:{port}"

    @property
    def query_endpoint(self) -> str:
        """Get the query socket endpoint."""
        return self._build_endpoint("query", self.query_port)

    @property
    def event_endpoint(self) -> str:
        """Get the event socket endpoint."""
        return self._build_endpoint("events", self.event_port)

    @property
    def query_bind_endpoint(self) -> str:
        """Get the query socket bind endpoint (for server)."""
        return self._build_endpoint("query", self.query_port, bind=True)

    @property
    def event_bind_endpoint(self) -> str:
        """Get the event socket bind endpoint (for server)."""
        return self._build_endpoint("events", self.event_port, bind=True)

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if self.query_port == self.event_port:
            warnings.append("Query and event ports are the same - they must be different")

        if self.timeout_ms < 1000:
            warnings.append(f"Timeout {self.timeout_ms}ms is very short, may cause issues")

        if self.use_ipc and sys.platform == "win32":
            warnings.append("IPC mode requested but Windows doesn't support Unix domain sockets - using TCP")

        return warnings

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def for_testing(cls, query_port: int = 15555, event_port: int = 15556) -> "ServerConfig":
        """Create configuration for testing with non-standard ports."""
        return cls(
            query_port=query_port,
            event_port=event_port,
            console_enabled=False,
            timeout_ms=5000
        )

    @classmethod
    def for_project(cls, project_root: Path, use_dynamic_ports: bool = True) -> "ServerConfig":
        """Create configuration for a specific project.

        Args:
            project_root: Project directory (discovery file will be written here)
            use_dynamic_ports: If True, use port 0 for dynamic allocation
        """
        # Ensure project_root is a Path, not a string
        if isinstance(project_root, str):
            project_root = Path(project_root)
        config = cls(project_root=project_root)
        if use_dynamic_ports:
            # Port 0 means OS will assign a free port
            config.query_port = 0
            config.event_port = 0
        return config

    def create_discovery(self, actual_query_port: int, actual_event_port: int) -> ServerDiscovery:
        """Create discovery info after server binds to ports.

        Call this after ZeroMQ binds to get the actual assigned ports.
        """
        import os
        return ServerDiscovery(
            project_root=str(self.project_root) if self.project_root else "",
            query_endpoint=f"tcp://{self.host}:{actual_query_port}",
            event_endpoint=f"tcp://{self.host}:{actual_event_port}",
            query_port=actual_query_port,
            event_port=actual_event_port,
            host=self.host,
            pid=os.getpid(),
            started_at=time.time(),
        )

    def write_discovery(self, actual_query_port: int, actual_event_port: int) -> ServerDiscovery:
        """Create and write discovery file after server starts."""
        if not self.project_root:
            raise ValueError("Cannot write discovery without project_root")
        discovery = self.create_discovery(actual_query_port, actual_event_port)
        discovery.save(self.project_root)
        return discovery

    def cleanup_discovery(self) -> bool:
        """Remove discovery file (call on shutdown)."""
        if self.project_root:
            return ServerDiscovery.remove(self.project_root)
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "query_port": self.query_port,
            "event_port": self.event_port,
            "query_endpoint": self.query_endpoint,
            "event_endpoint": self.event_endpoint,
            "use_ipc": self.use_ipc,
            "ipc_path": self.ipc_path,
            "timeout_ms": self.timeout_ms,
            "reconnect_interval_ms": self.reconnect_interval_ms,
            "console_enabled": self.console_enabled,
            "console_log_queries": self.console_log_queries,
            "console_show_progress": self.console_show_progress,
            "max_workers": self.max_workers,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
        }


@dataclass
class ClientConfig:
    """Configuration for RETER ZeroMQ client.

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a value-object.
    ::: This is stateless.
    ::: This is serializable.

    Supports environment variables:
    - RETER_SERVER_HOST: Server host to connect to
    - RETER_SERVER_QUERY_PORT: Query socket port
    - RETER_SERVER_EVENT_PORT: Event socket port
    - RETER_PROJECT_ROOT: Project root for discovery-based connection

    Connection priority:
    1. Explicit endpoint (RETER_ENDPOINT env var)
    2. Discovery file in project root
    3. Environment variables for host/port
    4. Default localhost:5555
    """

    # Project for discovery-based connection
    project_root: Optional[Path] = field(default=None)

    # Server location (fallback if no discovery)
    host: str = field(default_factory=lambda: os.environ.get("RETER_SERVER_HOST", DEFAULT_HOST))
    query_port: int = field(default_factory=lambda: int(os.environ.get("RETER_SERVER_QUERY_PORT", DEFAULT_QUERY_PORT)))
    event_port: int = field(default_factory=lambda: int(os.environ.get("RETER_SERVER_EVENT_PORT", DEFAULT_EVENT_PORT)))

    # Connection settings
    timeout_ms: int = field(default_factory=lambda: int(os.environ.get("RETER_CLIENT_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)))
    reconnect_interval_ms: int = field(default_factory=lambda: int(os.environ.get(
        "RETER_CLIENT_RECONNECT_MS", DEFAULT_RECONNECT_INTERVAL_MS
    )))
    max_reconnect_attempts: int = field(default_factory=lambda: int(os.environ.get("RETER_MAX_RECONNECT", "5")))

    # IPC configuration
    use_ipc: bool = field(default_factory=lambda: os.environ.get("RETER_CLIENT_USE_IPC", "false").lower() == "true")
    ipc_path: str = field(default_factory=lambda: os.environ.get("RETER_CLIENT_IPC_PATH", get_default_ipc_path()))

    # Client identification
    client_id: Optional[str] = None

    @property
    def query_endpoint(self) -> str:
        """Get the query socket endpoint."""
        if self.use_ipc and sys.platform != "win32":
            return f"ipc://{self.ipc_path}-query"
        return f"tcp://{self.host}:{self.query_port}"

    @property
    def event_endpoint(self) -> str:
        """Get the event socket endpoint."""
        if self.use_ipc and sys.platform != "win32":
            return f"ipc://{self.ipc_path}-events"
        return f"tcp://{self.host}:{self.event_port}"

    @classmethod
    def from_env(cls) -> "ClientConfig":
        """Create configuration from environment variables."""
        project_root = os.environ.get("RETER_PROJECT_ROOT")
        return cls(
            project_root=Path(project_root) if project_root else None
        )

    @classmethod
    def for_project(cls, project_root: Path) -> "ClientConfig":
        """Create client config for a specific project using discovery."""
        # Ensure project_root is a Path, not a string
        if isinstance(project_root, str):
            project_root = Path(project_root)
        return cls(project_root=project_root)

    @classmethod
    def for_server_config(cls, server_config: ServerConfig) -> "ClientConfig":
        """Create client config that matches a server config."""
        return cls(
            project_root=server_config.project_root,
            host=server_config.host,
            query_port=server_config.query_port,
            event_port=server_config.event_port,
            timeout_ms=server_config.timeout_ms,
            use_ipc=server_config.use_ipc,
            ipc_path=server_config.ipc_path
        )

    def discover_server(self) -> Optional[ServerDiscovery]:
        """Try to discover server from project discovery file."""
        if not self.project_root:
            return None
        return ServerDiscovery.load(self.project_root)

    def get_connection_info(self) -> tuple[str, str]:
        """Get query and event endpoints, using discovery if available.

        Returns:
            Tuple of (query_endpoint, event_endpoint)
        """
        # First try discovery file
        discovery = self.discover_server()
        if discovery and discovery.is_server_alive():
            return discovery.query_endpoint, discovery.event_endpoint

        # Fall back to configured endpoints
        return self.query_endpoint, self.event_endpoint

    def wait_for_server(self, timeout_seconds: float = 30.0, poll_interval: float = 0.5) -> Optional[ServerDiscovery]:
        """Wait for server to start and write discovery file.

        Args:
            timeout_seconds: Maximum time to wait
            poll_interval: Time between checks

        Returns:
            ServerDiscovery if found, None if timeout
        """
        if not self.project_root:
            return None

        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            discovery = self.discover_server()
            if discovery and discovery.is_server_alive():
                return discovery
            time.sleep(poll_interval)

        return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ServerConfig",
    "ClientConfig",
    "ServerDiscovery",
    "get_discovery_file_path",
    "DEFAULT_HOST",
    "DEFAULT_QUERY_PORT",
    "DEFAULT_EVENT_PORT",
    "DEFAULT_TIMEOUT_MS",
    "DEFAULT_RECONNECT_INTERVAL_MS",
    "DISCOVERY_DIR",
    "DISCOVERY_FILE",
]
