"""
RETER ZeroMQ Client.

Client that mirrors ReterWrapper interface but communicates via ZeroMQ.

::: This is-in-layer Client-Layer.
::: This is-in-component ZeroMQ-Client.
::: This depends-on pyzmq.
::: This mirrors-interface ReterWrapper.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import zmq

from .protocol import (
    ReterMessage,
    ReterError,
    serialize,
    deserialize,
    TIMEOUT_ERROR,
    CONNECTION_ERROR,
    # Method constants
    METHOD_REQL,
    METHOD_DL,
    METHOD_PATTERN,
    METHOD_NATURAL_LANGUAGE,
    METHOD_EXECUTE_CADSL,
    METHOD_GENERATE_CADSL,
    METHOD_ADD_KNOWLEDGE,
    METHOD_ADD_DIRECTORY,
    METHOD_FORGET,
    METHOD_RELOAD,
    METHOD_VALIDATE_CNL,
    METHOD_RAG_SEARCH,
    METHOD_RAG_REINDEX,
    METHOD_CODE_INSPECTION,
    METHOD_DIAGRAM,
    METHOD_RECOMMENDER,
    METHOD_SYSTEM,
    METHOD_FILE_SCAN,
    METHOD_SESSION,
    METHOD_THINKING,
    METHOD_ITEMS,
    METHOD_SIMILAR_CADSL_TOOLS,
)
from .config import ClientConfig

logger = logging.getLogger(__name__)


class ReterClientError(Exception):
    """Base exception for ReterClient errors.

    ::: This is-in-layer Core-Layer.
    ::: This is a value-object.
    ::: This is stateless.
    """

    def __init__(self, message: str, error: Optional[ReterError] = None):
        super().__init__(message)
        self.error = error


class ReterClientTimeoutError(ReterClientError):
    """Request timed out.

    ::: This is-in-layer Core-Layer.
    ::: This is a value-object.
    ::: This is stateless.
    """
    pass


class ReterClientConnectionError(ReterClientError):
    """Connection to server failed.

    ::: This is-in-layer Core-Layer.
    ::: This is a value-object.
    ::: This is stateless.
    """
    pass


class ReterClient:
    """ZeroMQ client with same interface as ReterWrapper.

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a client.
    ::: This is stateful.
    ::: This depends-on `zmq.Context`.

    This client provides the same interface as ReterWrapper but
    communicates with a RETER server process via ZeroMQ.

    Usage:
        # Using discovery (recommended for multi-project)
        client = ReterClient.for_project(Path("/path/to/project"))

        # Using explicit endpoint
        client = ReterClient(endpoint="tcp://127.0.0.1:5555")

        # Execute queries
        result = client.reql("SELECT ?c WHERE {?c type class}")
    """

    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        endpoint: Optional[str] = None
    ):
        """Initialize RETER client.

        Args:
            config: Client configuration
            endpoint: Explicit endpoint (overrides config/discovery)
        """
        self.config = config or ClientConfig()
        self._explicit_endpoint = endpoint
        self._client_id = str(uuid.uuid4())[:8]

        # ZeroMQ context and socket
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._connected = False

        # Connection state
        self._reconnect_attempts = 0
        self._last_error: Optional[str] = None

    @classmethod
    def for_project(cls, project_root: Path) -> "ReterClient":
        """Create client for a specific project using discovery.

        Args:
            project_root: Project directory

        Returns:
            Configured ReterClient
        """
        config = ClientConfig.for_project(project_root)
        return cls(config=config)

    @classmethod
    def for_endpoint(cls, endpoint: str, timeout_ms: int = 30000) -> "ReterClient":
        """Create client for explicit endpoint.

        Args:
            endpoint: ZeroMQ endpoint (e.g., "tcp://127.0.0.1:5555")
            timeout_ms: Request timeout

        Returns:
            Configured ReterClient
        """
        config = ClientConfig(timeout_ms=timeout_ms)
        return cls(config=config, endpoint=endpoint)

    def _get_endpoint(self) -> str:
        """Get the endpoint to connect to."""
        if self._explicit_endpoint:
            return self._explicit_endpoint

        # Try discovery
        query_endpoint, _ = self.config.get_connection_info()
        return query_endpoint

    def _ensure_connected(self) -> None:
        """Ensure socket is connected, reconnecting if needed."""
        if self._connected and self._socket:
            return

        try:
            if not self._context:
                self._context = zmq.Context()

            if self._socket:
                self._socket.close()

            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.RCVTIMEO, self.config.timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, self.config.timeout_ms)
            self._socket.setsockopt(zmq.LINGER, 0)

            endpoint = self._get_endpoint()
            self._socket.connect(endpoint)
            self._connected = True
            self._reconnect_attempts = 0

            logger.debug(f"Connected to {endpoint}")

        except zmq.ZMQError as e:
            self._connected = False
            raise ReterClientConnectionError(f"Failed to connect: {e}")

    def _send_request(
        self,
        method: str,
        params: Dict[str, Any],
        timeout_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """Send request and wait for response.

        Args:
            method: Method name
            params: Method parameters
            timeout_ms: Optional per-request timeout override

        Returns:
            Response result dictionary

        Raises:
            ReterClientTimeoutError: Request timed out
            ReterClientConnectionError: Connection failed
            ReterClientError: Server returned error
        """
        self._ensure_connected()

        # Apply per-request timeout if specified
        effective_timeout = timeout_ms if timeout_ms is not None else self.config.timeout_ms
        if timeout_ms is not None and self._socket:
            self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

        request = ReterMessage.request(method, params, client_id=self._client_id)

        try:
            # Send request
            self._socket.send(serialize(request))

            # Wait for response
            raw_response = self._socket.recv()
            response = deserialize(raw_response)

            # Check for error
            if response.error:
                raise ReterClientError(
                    response.error.message,
                    error=response.error
                )

            return response.result or {}

        except zmq.Again:
            # Timeout
            self._handle_timeout()
            raise ReterClientTimeoutError(
                f"Request timed out after {effective_timeout}ms"
            )

        except zmq.ZMQError as e:
            self._connected = False
            raise ReterClientConnectionError(f"ZMQ error: {e}")

    def _handle_timeout(self) -> None:
        """Handle request timeout by resetting socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
        self._connected = False

    def close(self) -> None:
        """Close the client connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None
        self._connected = False

    def __enter__(self) -> "ReterClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # =========================================================================
    # Query Operations (mirror ReterWrapper interface)
    # =========================================================================

    def reql(self, query: str, timeout_ms: Optional[int] = None) -> Any:
        """Execute REQL query.

        Args:
            query: REQL query string
            timeout_ms: Optional timeout override (also sets socket timeout)

        Returns:
            Query result (as dict with columns/rows)
        """
        params = {"query": query}
        if timeout_ms is not None:
            params["timeout_ms"] = timeout_ms

        return self._send_request(METHOD_REQL, params, timeout_ms=timeout_ms)

    def dl(self, query: str) -> List[str]:
        """Execute DL query.

        Args:
            query: Description Logic query

        Returns:
            List of matching instances
        """
        result = self._send_request(METHOD_DL, {"query": query})
        return result.get("instances", [])

    def pattern(self, query: str) -> Any:
        """Execute pattern query.

        Args:
            query: Pattern query string

        Returns:
            Query result
        """
        return self._send_request(METHOD_PATTERN, {"query": query})

    # =========================================================================
    # CADSL Operations
    # =========================================================================

    def execute_cadsl(
        self,
        script: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 300000
    ) -> Dict[str, Any]:
        """Execute CADSL script.

        Args:
            script: CADSL script or file path
            params: Script parameters
            timeout_ms: Execution timeout (also sets socket timeout)

        Returns:
            Execution result
        """
        return self._send_request(
            METHOD_EXECUTE_CADSL,
            {
                "script": script,
                "params": params or {},
                "timeout_ms": timeout_ms
            },
            timeout_ms=timeout_ms
        )

    def generate_cadsl(
        self,
        question: str,
        max_retries: int = 5
    ) -> Dict[str, Any]:
        """Generate CADSL from natural language.

        Args:
            question: Natural language question
            max_retries: Maximum retry attempts

        Returns:
            Generated CADSL query
        """
        return self._send_request(METHOD_GENERATE_CADSL, {
            "question": question,
            "max_retries": max_retries
        })

    def natural_language_query(
        self,
        question: str,
        max_retries: int = 5,
        timeout_ms: int = 300000
    ) -> Dict[str, Any]:
        """Execute natural language query.

        Args:
            question: Natural language question
            max_retries: Maximum retry attempts
            timeout_ms: Execution timeout (also sets socket timeout)

        Returns:
            Query result with generated CADSL
        """
        return self._send_request(
            METHOD_NATURAL_LANGUAGE,
            {
                "question": question,
                "max_retries": max_retries,
                "timeout_ms": timeout_ms
            },
            timeout_ms=timeout_ms
        )

    # =========================================================================
    # Knowledge Operations
    # =========================================================================

    def add_knowledge(
        self,
        source: str,
        source_type: str = "ontology",
        source_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add knowledge from source.

        Args:
            source: Ontology content, file path, or code
            source_type: Type (ontology, python, javascript, etc.)
            source_id: Optional identifier

        Returns:
            Load result with items_added
        """
        return self._send_request(METHOD_ADD_KNOWLEDGE, {
            "source": source,
            "type": source_type,
            "source_id": source_id
        })

    def add_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Add all code files from directory.

        Args:
            directory: Directory path
            recursive: Search subdirectories
            exclude_patterns: Patterns to exclude

        Returns:
            Scan result
        """
        return self._send_request(METHOD_ADD_DIRECTORY, {
            "directory": directory,
            "recursive": recursive,
            "exclude_patterns": exclude_patterns or []
        })

    def forget_source(self, source: str) -> Tuple[str, float]:
        """Forget knowledge from source.

        Args:
            source: Source identifier

        Returns:
            Tuple of (source_id, time_ms)
        """
        result = self._send_request(METHOD_FORGET, {"source": source})
        return result.get("source", source), result.get("execution_time_ms", 0)

    def reload(self) -> Dict[str, Any]:
        """Reload modified sources.

        Returns:
            Reload statistics
        """
        return self._send_request(METHOD_RELOAD, {})

    def validate_cnl(
        self,
        statement: str,
        context_entity: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate CNL statement.

        Args:
            statement: CNL statement
            context_entity: Entity for "This" resolution

        Returns:
            Validation result
        """
        return self._send_request(METHOD_VALIDATE_CNL, {
            "statement": statement,
            "context_entity": context_entity
        })

    # =========================================================================
    # RAG Operations
    # =========================================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        entity_types: Optional[List[str]] = None,
        file_filter: Optional[str] = None,
        include_content: bool = False,
        search_scope: str = "all"
    ) -> Dict[str, Any]:
        """Semantic search over code and docs.

        Args:
            query: Natural language search query
            top_k: Maximum results
            entity_types: Filter by type
            file_filter: Glob pattern filter
            include_content: Include source code
            search_scope: "all", "code", or "docs"

        Returns:
            Search results
        """
        return self._send_request(METHOD_RAG_SEARCH, {
            "query": query,
            "top_k": top_k,
            "entity_types": entity_types,
            "file_filter": file_filter,
            "include_content": include_content,
            "search_scope": search_scope
        })

    def rag_reindex(self, force: bool = False) -> Dict[str, Any]:
        """Rebuild RAG index.

        Args:
            force: Force full rebuild

        Returns:
            Reindex statistics
        """
        return self._send_request(METHOD_RAG_REINDEX, {"force": force})

    def rag_duplicates(
        self,
        threshold: float = 0.85,
        entity_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Find duplicate code.

        Args:
            threshold: Similarity threshold
            entity_types: Filter by type
            limit: Maximum pairs

        Returns:
            Duplicate pairs
        """
        return self._send_request("rag_duplicates", {
            "threshold": threshold,
            "entity_types": entity_types,
            "limit": limit
        })

    def rag_clusters(
        self,
        n_clusters: int = 50,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Cluster code by similarity.

        Args:
            n_clusters: Number of clusters
            entity_types: Filter by type

        Returns:
            Cluster assignments
        """
        return self._send_request("rag_clusters", {
            "n_clusters": n_clusters,
            "entity_types": entity_types
        })

    # =========================================================================
    # File Scan Operations
    # =========================================================================

    def file_scan(
        self,
        glob: str = "*",
        contains: Optional[str] = None,
        exclude: Optional[str] = None,
        case_sensitive: bool = False,
        include_matches: bool = True,
        context_lines: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Scan RETER-tracked files.

        Args:
            glob: Glob pattern
            contains: Content pattern
            exclude: Exclude pattern
            case_sensitive: Case sensitivity
            include_matches: Include matching lines
            context_lines: Context lines
            limit: Maximum files

        Returns:
            Matching files
        """
        return self._send_request(METHOD_FILE_SCAN, {
            "glob": glob,
            "contains": contains,
            "exclude": exclude,
            "case_sensitive": case_sensitive,
            "include_matches": include_matches,
            "context_lines": context_lines,
            "limit": limit
        })

    # =========================================================================
    # Code Inspection Operations
    # =========================================================================

    def code_inspection(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute code inspection action.

        Args:
            action: Inspection action
            **kwargs: Action parameters

        Returns:
            Inspection result
        """
        return self._send_request(METHOD_CODE_INSPECTION, {
            "action": action,
            **kwargs
        })

    def diagram(self, diagram_type: str, **kwargs) -> Dict[str, Any]:
        """Generate diagram.

        Args:
            diagram_type: Type of diagram
            **kwargs: Diagram parameters

        Returns:
            Diagram content
        """
        return self._send_request(METHOD_DIAGRAM, {
            "diagram_type": diagram_type,
            **kwargs
        })

    def recommender(
        self,
        recommender_type: Optional[str] = None,
        detector_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run recommender/detector.

        Args:
            recommender_type: Recommender type
            detector_name: Specific detector
            **kwargs: Additional parameters

        Returns:
            Recommendations
        """
        return self._send_request(METHOD_RECOMMENDER, {
            "recommender_type": recommender_type,
            "detector_name": detector_name,
            **kwargs
        })

    # =========================================================================
    # System Operations
    # =========================================================================

    def system(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute system action.

        Args:
            action: System action (status, info, sources, etc.)
            **kwargs: Action parameters

        Returns:
            Action result
        """
        return self._send_request(METHOD_SYSTEM, {
            "action": action,
            **kwargs
        })

    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        return self.system("status")

    def get_sources(self) -> List[str]:
        """Get all loaded sources."""
        result = self.system("sources")
        return result.get("sources", [])

    # =========================================================================
    # Session Operations
    # =========================================================================

    def session(self, action: str, **kwargs) -> Dict[str, Any]:
        """Session lifecycle operation.

        Args:
            action: Session action (start, context, end, clear)
            **kwargs: Action parameters

        Returns:
            Session state
        """
        return self._send_request(METHOD_SESSION, {
            "action": action,
            **kwargs
        })

    def thinking(self, **kwargs) -> Dict[str, Any]:
        """Create thinking entry.

        Args:
            **kwargs: Thinking parameters

        Returns:
            Thought creation result
        """
        return self._send_request(METHOD_THINKING, kwargs)

    def items(self, action: str = "list", **kwargs) -> Dict[str, Any]:
        """Item operations.

        Args:
            action: Item action (list, get, update, delete)
            **kwargs: Action parameters

        Returns:
            Item operation result
        """
        return self._send_request(METHOD_ITEMS, {
            "action": action,
            **kwargs
        })

    # =========================================================================
    # CADSL Tool Operations
    # =========================================================================

    def similar_cadsl_tools(
        self,
        question: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Find CADSL tools similar to a question.

        Args:
            question: Natural language question
            max_results: Maximum results to return

        Returns:
            Dictionary with similar_tools list
        """
        return self._send_request(METHOD_SIMILAR_CADSL_TOOLS, {
            "question": question,
            "max_results": max_results
        })


__all__ = [
    "ReterClient",
    "ReterClientError",
    "ReterClientTimeoutError",
    "ReterClientConnectionError",
]
