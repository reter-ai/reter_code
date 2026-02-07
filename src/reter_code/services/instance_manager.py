"""
Instance Manager Service

Manages RETER instances and synchronization locks.
Extracted from LogicalThinkingServer as part of God Class refactoring.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
from ..reter_wrapper import ReterWrapper
from ..logging_config import configure_logger_for_debug_trace

logger = configure_logger_for_debug_trace(__name__)

# Avoid circular import
if TYPE_CHECKING:
    from .state_persistence import StatePersistenceService
    from .default_instance_manager import DefaultInstanceManager


class InstanceManager:
    """
    Manages RETER instances for the logical-thinking MCP server.

    This class serves as the central dependency hub for all service classes,
    providing access to RETER instances and synchronization locks.

    Responsibilities:
    - Create and retrieve RETER instances by name
    - Manage per-instance locks for thread safety
    - Support lazy loading of instances from snapshots

    ::: This is-in-layer Service-Layer.
    ::: This is a core-service.
    ::: This depends-on `reter_code.reter_wrapper.ReterWrapper`.
    ::: This is-in-process Main-Process.
    ::: This is stateful.
    ::: This owns-resource `ReterWrapper`.
    ::: This has-singleton-scope.
    """

    def __init__(self):
        """Initialize the instance manager with empty dictionaries for instances."""
        self._instances: Dict[str, ReterWrapper] = {}
        self._persistence: Optional["StatePersistenceService"] = None
        self._default_manager: Optional["DefaultInstanceManager"] = None

        # NOTE: Removed lock management - RETER is now thread-safe at C++ level
        # No need for Python-side synchronization

    def set_persistence_service(self, persistence: "StatePersistenceService") -> None:
        """
        Set the persistence service for lazy loading support.

        Args:
            persistence: StatePersistenceService instance
        """
        self._persistence = persistence

    def set_default_manager(self, default_manager: "DefaultInstanceManager") -> None:
        """
        Set the default instance manager for auto-sync support.

        Args:
            default_manager: DefaultInstanceManager instance
        """
        self._default_manager = default_manager

    def get_default_instance_manager(self) -> Optional["DefaultInstanceManager"]:
        """
        Get the default instance manager.

        Returns:
            DefaultInstanceManager instance or None if not set
        """
        return self._default_manager

    def ensure_instance_loaded(self, instance_name: str) -> None:
        """
        Ensure instance is loaded (lazy load from snapshot if available).
        Call this before using get_or_create_instance() to enable lazy loading.

        For "default" instance: also performs file synchronization.

        Args:
            instance_name: Name of the RETER instance
        """
        # Get progress callback if available
        progress = getattr(self, '_current_progress_callback', None)

        # Skip if instance already exists
        if instance_name in self._instances:
            # For default instance, sync files even if already loaded
            if instance_name == "default" and self._default_manager and self._default_manager.is_configured():
                if not progress:
                    logger.debug(f"Instance '{instance_name}' already loaded, syncing...")
                rebuilt = self._default_manager.ensure_default_instance_synced(self._instances[instance_name], progress_callback=progress)
                if rebuilt is not None:
                    # Instance was rebuilt from scratch to compact RETE network
                    if not progress:
                        logger.debug("Default instance rebuilt, replacing in cache")
                    self._instances[instance_name] = rebuilt
            return

        # Try lazy loading from snapshot if persistence service is available
        if self._persistence:
            import time
            if not progress:
                logger.debug(f"Trying to load '{instance_name}' from snapshot...")
            start = time.time()
            self._persistence.load_snapshot_if_available(instance_name)
            if instance_name in self._instances:
                if not progress:
                    logger.debug(f"Successfully loaded '{instance_name}' from snapshot in {time.time()-start:.2f}s")
                # For default instance: sync files after loading from snapshot (handles changed files + RAG init)
                if instance_name == "default" and self._default_manager and self._default_manager.is_configured():
                    if not progress:
                        logger.debug("Syncing default instance after snapshot load...")
                    rebuilt = self._default_manager.ensure_default_instance_synced(self._instances[instance_name], progress_callback=progress)
                    if rebuilt is not None:
                        # Instance was rebuilt from scratch to compact RETE network
                        if not progress:
                            logger.debug("Default instance rebuilt, replacing in cache")
                        self._instances[instance_name] = rebuilt
                    if not progress:
                        logger.debug(f"Sync complete in {time.time()-start:.2f}s total")

    def get_or_create_instance(self, instance_name: str, progress_callback: Optional[Any] = None) -> ReterWrapper:
        """
        Get an existing RETER instance or create a new one.

        Automatically attempts lazy loading from snapshot if available.
        For "default" instance: creates from project files if snapshot doesn't exist.

        Thread-safe: RETER is thread-safe at C++ level, no Python locks needed.

        Args:
            instance_name: Name of the RETER instance
            progress_callback: Optional progress callback for UI updates

        Returns:
            ReterWrapper instance
        """
        logger.debug(f"get_or_create_instance ENTER: {instance_name}")

        # Store progress callback for use in ensure_instance_loaded
        self._current_progress_callback = progress_callback

        # Always ensure instance is loaded/synced (critical for default instance auto-sync)
        logger.debug(f"get_or_create_instance: calling ensure_instance_loaded({instance_name})")
        self.ensure_instance_loaded(instance_name)
        logger.debug(f"get_or_create_instance: ensure_instance_loaded returned")

        # Clear progress callback
        self._current_progress_callback = None

        # Create new instance if lazy loading didn't work
        if instance_name not in self._instances:
            logger.debug(f"get_or_create_instance: creating new ReterWrapper for {instance_name}")
            self._instances[instance_name] = ReterWrapper()
            logger.debug(f"get_or_create_instance: ReterWrapper created for {instance_name}")

            # For default instance: load from project files
            if instance_name == "default" and self._default_manager and self._default_manager.is_configured():
                logger.debug("[default] Creating default instance from project files...")
                rebuilt = self._default_manager.ensure_default_instance_synced(
                    self._instances[instance_name], progress_callback=progress_callback
                )
                if rebuilt is not None:
                    # Instance was rebuilt from scratch to compact RETE network
                    logger.debug("[default] Default instance rebuilt, replacing in cache")
                    self._instances[instance_name] = rebuilt
                logger.debug("[default] ensure_default_instance_synced completed")

        logger.debug(f"get_or_create_instance EXIT: {instance_name}")
        return self._instances[instance_name]

    def get_all_instances(self) -> Dict[str, ReterWrapper]:
        """
        Get all RETER instances.

        Returns:
            Dictionary mapping instance names to ReterWrapper instances
        """
        return self._instances
