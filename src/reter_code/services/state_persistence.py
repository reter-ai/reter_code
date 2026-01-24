"""
State Persistence Service

Handles all RETER state persistence operations including:
- Saving/loading RETER state to/from binary files
- Managing knowledge sources (listing, querying)
- Automatic snapshot management on server startup/shutdown
- Immediate save after knowledge additions (save-after-add pattern)

Extracted from LogicalThinkingServer as part of God Class refactoring.
"""

import sys
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .instance_manager import InstanceManager

logger = logging.getLogger(__name__)


class StatePersistenceService:
    """
    Service for RETER state persistence operations.

    This service handles:
    - Saving RETER instance state to disk
    - Loading RETER instance state from disk
    - Listing knowledge sources
    - Querying source facts
    - Automatic snapshot management

    Responsibilities:
    - Persist RETER state for durability
    - Restore RETER state on startup
    - Track and query knowledge sources
    - Manage automatic snapshots directory
    """

    def __init__(self, instance_manager: InstanceManager):
        """
        Initialize state persistence service.

        Args:
            instance_manager: InstanceManager for accessing RETER instances
        """
        self.instance_manager = instance_manager
        # Snapshot directory priority:
        # 1. RETER_SNAPSHOTS_DIR env var (explicit override)
        # 2. RETER_PROJECT_ROOT/.reter_code (if project root is explicitly set)
        # 3. CWD/.reter_code (auto-detection - Claude Code sets CWD to project root)
        snapshots_dir = os.getenv("RETER_SNAPSHOTS_DIR")
        if not snapshots_dir:
            project_root = os.getenv("RETER_PROJECT_ROOT")
            if project_root:
                snapshots_dir = str(Path(project_root) / ".reter_code")
            else:
                snapshots_dir = str(Path.cwd() / ".reter_code")
        self.snapshots_dir = Path(snapshots_dir)
        # Track available snapshots (discovered but not yet loaded)
        self._available_snapshots: Dict[str, Path] = {}

    def save_all_instances(self) -> None:
        """
        Save all RETER instances to .reter_code/ directory as snapshots.
        Called automatically on server shutdown.
        Only saves instances that have unsaved changes (dirty flag).
        Also shuts down instance resources (executors, queues).
        """
        try:
            # Create snapshots directory if it doesn't exist
            self.snapshots_dir.mkdir(exist_ok=True)

            instances = self.instance_manager.get_all_instances()
            if not instances:
                print("  ℹ️  No instances to save", file=sys.stderr)
                return

            # Save each instance to a file (atomic write: .tmp then rename)
            # Uses .{instance_name}.reter format (with leading dot)
            saved_count = 0
            skipped_count = 0
            for instance_name, reter in instances.items():
                snapshot_path = self.snapshots_dir / f".{instance_name}.reter"
                temp_path = self.snapshots_dir / f".{instance_name}.reter.tmp"

                try:
                    # Check if instance needs saving
                    if not reter.is_dirty():
                        skipped_count += 1
                    else:
                        # Write to temporary file first
                        success, filename, time_ms = reter.save_network(str(temp_path))

                        if success:
                            # Atomic rename: .tmp → .reter
                            temp_path.replace(snapshot_path)
                            print(f"  ✅ Saved '{instance_name}' → {snapshot_path} ({time_ms:.2f}ms)", file=sys.stderr)
                            saved_count += 1
                        else:
                            print(f"  ⚠️  Failed to save '{instance_name}': Unknown error", file=sys.stderr)
                            # Clean up temp file on failure
                            if temp_path.exists():
                                temp_path.unlink()

                    # Shutdown instance resources (executor, queue)
                    try:
                        reter.shutdown()
                    except Exception as shutdown_err:
                        print(f"  ⚠️  Error shutting down '{instance_name}': {shutdown_err}", file=sys.stderr)

                except Exception as e:
                    print(f"  ❌ Error saving '{instance_name}': {e}", file=sys.stderr)
                    # Clean up temp file on error
                    if temp_path.exists():
                        temp_path.unlink()

            if skipped_count > 0:
                print(f"  📊 Saved {saved_count}/{len(instances)} instances ({skipped_count} unchanged, skipped)", file=sys.stderr)
            else:
                print(f"  📊 Saved {saved_count}/{len(instances)} instances", file=sys.stderr)

        except Exception as e:
            print(f"  ❌ Error during save_all_instances: {e}", file=sys.stderr)

    def discover_snapshots(self) -> None:
        """
        Discover available snapshot files without loading them.
        Called on server startup for lazy loading support.
        """
        try:
            # Scan the directory
            self._scan_snapshot_directory()

            if not self.snapshots_dir.exists():
                print(f"  ℹ️  No snapshots directory found (will be created on first save)", file=sys.stderr)
                return

            if not self._available_snapshots:
                print(f"  ℹ️  No snapshots found in {self.snapshots_dir}", file=sys.stderr)
                return

            print(f"  📊 Discovered {len(self._available_snapshots)} snapshot(s) (will load on first use)", file=sys.stderr)

        except Exception as e:
            print(f"  ❌ Error during discover_snapshots: {e}", file=sys.stderr)

    def _scan_snapshot_directory(self) -> None:
        """
        Scan the snapshots directory for .reter files and update available snapshots.
        Internal method called before listing instances to ensure fresh data.

        Looks for files named .{instance_name}.reter (with leading dot).
        """
        # Clear the current available snapshots
        self._available_snapshots.clear()

        # Check if snapshots directory exists
        if not self.snapshots_dir.exists():
            return

        # Find all .reter snapshot files (with leading dot: .{instance_name}.reter)
        snapshot_files = list(self.snapshots_dir.glob(".*.reter"))

        # Register available snapshots
        for snapshot_path in snapshot_files:
            # Filename is .{instance_name}.reter, stem gives .{instance_name}
            # We need to strip the leading dot to get instance_name
            stem = snapshot_path.stem  # e.g., ".current" from ".current.reter"
            if stem.startswith("."):
                instance_name = stem[1:]  # Strip leading dot
            else:
                instance_name = stem
            self._available_snapshots[instance_name] = snapshot_path

    def get_all_instance_names(self) -> Dict[str, str]:
        """
        Get all instance names (both loaded and persisted but not yet loaded).

        Always rescans the snapshots directory to ensure fresh data.

        Returns:
            Dictionary mapping instance names to their status:
            - "loaded": Instance is currently in memory
            - "available": Instance exists as snapshot but not loaded
        """
        # Rescan directory to get latest snapshots
        self._scan_snapshot_directory()

        result = {}

        # Add loaded instances
        loaded_instances = self.instance_manager.get_all_instances()
        for instance_name in loaded_instances.keys():
            result[instance_name] = "loaded"

        # Add available snapshots (not yet loaded)
        for instance_name in self._available_snapshots.keys():
            if instance_name not in result:  # Don't overwrite loaded instances
                result[instance_name] = "available"

        return result

    def get_available_snapshot_names(self) -> list[str]:
        """
        Get names of instances that exist as snapshots but are not yet loaded.

        Always rescans the snapshots directory to ensure fresh data.

        Returns:
            List of instance names with available snapshots
        """
        # Rescan directory to get latest snapshots
        self._scan_snapshot_directory()
        return list(self._available_snapshots.keys())

    def load_snapshot_if_available(self, instance_name: str) -> bool:
        """
        Load a snapshot for the given instance if available (lazy loading).

        Rescans the snapshots directory before attempting to load.

        Args:
            instance_name: Name of the instance to load

        Returns:
            True if snapshot was loaded, False otherwise
        """
        from reter_code.reter_wrapper import debug_log

        # Rescan directory to ensure we have the latest snapshots
        debug_log(f"[persistence] load_snapshot_if_available({instance_name}): scanning directory...")
        self._scan_snapshot_directory()
        debug_log(f"[persistence] Found snapshots: {list(self._available_snapshots.keys())}")

        # Check if snapshot is available
        if instance_name not in self._available_snapshots:
            debug_log(f"[persistence] Snapshot '{instance_name}' NOT in available snapshots, returning False")
            return False

        snapshot_path = self._available_snapshots[instance_name]

        try:
            import traceback

            # Create instance directly to avoid recursion (don't call get_or_create_instance)
            try:
                # Check if instance already exists
                if instance_name in self.instance_manager._instances:
                    reter = self.instance_manager._instances[instance_name]
                else:
                    # Create new instance directly without triggering lazy loading
                    # Use load_ontology=False since snapshot already contains ontology
                    from reter_code.reter_wrapper import ReterWrapper
                    reter = ReterWrapper(load_ontology=False)
                    self.instance_manager._instances[instance_name] = reter
            except Exception as inst_error:
                print(f"  ❌ Error creating instance '{instance_name}': {inst_error}", file=sys.stderr, flush=True)
                print(f"  Traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                return False

            # Load the snapshot
            try:
                debug_log(f"[persistence] Loading snapshot from {snapshot_path}...")
                success, filename, time_ms = reter.load_network(str(snapshot_path))

                if success:
                    # Check what sources are in the loaded snapshot
                    sources, _ = reter.get_all_sources()
                    debug_log(f"[persistence] Snapshot loaded: {len(sources)} sources found")
                    if sources:
                        debug_log(f"[persistence] First 3 sources: {sources[:3]}")

                    print(f"  ✅ Lazy-loaded '{instance_name}' ← {snapshot_path} ({time_ms:.2f}ms)", file=sys.stderr, flush=True)
                    # Remove from available snapshots (now loaded)
                    del self._available_snapshots[instance_name]
                    return True
                else:
                    print(f"  ⚠️  Failed to lazy-load '{instance_name}': load_network returned False", file=sys.stderr, flush=True)
                    return False

            except Exception as load_error:
                print(f"  ❌ Error loading network for '{instance_name}': {load_error}", file=sys.stderr, flush=True)
                print(f"  Traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                return False

        except Exception as e:
            import traceback
            print(f"  ❌ Unexpected error lazy-loading '{instance_name}': {e}", file=sys.stderr, flush=True)
            print(f"  Traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            return False

    def load_all_instances(self) -> None:
        """
        Load all RETER instances from .reter/ directory snapshots.
        DEPRECATED: Use discover_snapshots() for lazy loading instead.
        Kept for backward compatibility.
        """
        try:
            # Check if snapshots directory exists
            if not self.snapshots_dir.exists():
                print(f"  ℹ️  No snapshots directory found (will be created on first save)", file=sys.stderr)
                return

            # Find all .reter snapshot files (with leading dot: .{instance_name}.reter)
            snapshot_files = list(self.snapshots_dir.glob(".*.reter"))

            if not snapshot_files:
                print(f"  ℹ️  No snapshots found in {self.snapshots_dir}", file=sys.stderr)
                return

            # Load each snapshot
            loaded_count = 0
            for snapshot_path in snapshot_files:
                # Filename is .{instance_name}.reter, stem gives .{instance_name}
                stem = snapshot_path.stem
                instance_name = stem[1:] if stem.startswith(".") else stem

                try:
                    # Create instance
                    reter = self.instance_manager.get_or_create_instance(instance_name)

                    # Load snapshot
                    success, filename, time_ms = reter.load_network(str(snapshot_path))

                    if success:
                        print(f"  ✅ Loaded '{instance_name}' ← {snapshot_path}", file=sys.stderr)
                        loaded_count += 1
                    else:
                        print(f"  ⚠️  Failed to load '{instance_name}': Unknown error", file=sys.stderr)

                except Exception as e:
                    print(f"  ❌ Error loading '{instance_name}': {e}", file=sys.stderr)

            print(f"  📊 Loaded {loaded_count}/{len(snapshot_files)} instances", file=sys.stderr)

        except Exception as e:
            print(f"  ❌ Error during load_all_instances: {e}", file=sys.stderr)

    def save_state(self, instance_name: str, filename: str) -> Dict[str, Any]:
        """
        Save entire RETER network state to binary file using atomic write.
        Writes to .tmp file first, then renames to final filename.

        Args:
            instance_name: RETER instance name
            filename: Path to save file (typically .reter extension)

        Returns:
            success: Whether save succeeded
            filename: Path where state was saved
        """
        try:
            # Create parent directory if it doesn't exist
            file_path = Path(filename)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Temporary file path for atomic write
            temp_path = Path(str(filename) + ".tmp")

            # Get or create the RETER instance
            reter = self.instance_manager.get_or_create_instance(instance_name)

            # Write to temporary file first
            success, returned_filename, time_ms = reter.save_network(str(temp_path))

            if success:
                # Atomic rename: .tmp → final filename
                temp_path.replace(file_path)
                return {
                    "success": True,
                    "filename": filename,
                    "message": "Network state saved successfully (atomic write)",
                    "execution_time_ms": time_ms
                }
            else:
                # Clean up temp file on failure
                if temp_path.exists():
                    temp_path.unlink()
                return {
                    "success": False,
                    "filename": filename,
                    "message": "Failed to save network state",
                    "execution_time_ms": time_ms
                }

        except Exception as e:
            # Clean up temp file on error
            temp_path = Path(str(filename) + ".tmp")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError as cleanup_error:
                    logger.warning(f"Failed to clean up temp file {temp_path}: {cleanup_error}")

            return {
                "success": False,
                "filename": filename,
                "message": f"Failed to save state: {str(e)}",
                "execution_time_ms": 0
            }

    def load_state(self, instance_name: str, filename: str) -> Dict[str, Any]:
        """
        Load RETER network state from binary file.

        Args:
            instance_name: RETER instance name
            filename: Path to saved state file

        Returns:
            success: Whether load succeeded
            filename: Path that was loaded
        """
        try:
            import traceback

            # Lazy load instance if snapshot is available
            self.instance_manager.ensure_instance_loaded(instance_name)

            # Get or create the RETER instance
            reter = self.instance_manager.get_or_create_instance(instance_name)

            # Load the network
            try:
                success, returned_filename, time_ms = reter.load_network(filename)

                return {
                    "success": success,
                    "filename": returned_filename,
                    "message": "Network state loaded successfully",
                    "execution_time_ms": time_ms
                }
            except Exception as load_error:
                return {
                    "success": False,
                    "filename": filename,
                    "message": f"Network load failed: {str(load_error)}",
                    "traceback": traceback.format_exc(),
                    "execution_time_ms": 0
                }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "filename": filename,
                "message": f"Failed to load state: {str(e)}",
                "traceback": traceback.format_exc(),
                "execution_time_ms": 0
            }

    def list_sources(self, instance_name: str) -> Dict[str, Any]:
        """
        List all source identifiers currently loaded in the RETER instance.

        Returns all source IDs that have been used with add_knowledge().
        Useful for:
        - Tracking what knowledge fragments are loaded
        - Deciding which sources to forget
        - Understanding knowledge base composition

        Args:
            instance_name: RETER instance name

        Returns:
            success: Whether operation succeeded
            sources: List of source identifiers
            count: Number of sources
        """
        try:
            # Lazy load instance if snapshot is available
            self.instance_manager.ensure_instance_loaded(instance_name)

            # Get or create the RETER instance
            reter = self.instance_manager.get_or_create_instance(instance_name)

            # Get all sources
            sources, time_ms = reter.get_all_sources()

            return {
                "success": True,
                "sources": sources,
                "count": len(sources),
                "execution_time_ms": time_ms
            }
        except Exception as e:
            return {
                "success": False,
                "sources": [],
                "count": 0,
                "error": str(e),
                "execution_time_ms": 0
            }

    def get_source_facts(self, instance_name: str, source: str) -> Dict[str, Any]:
        """
        Get all fact IDs associated with a specific source.

        Intelligently handles both exact source IDs and file paths:
        - If source contains "@", treats as exact source ID
        - If source is a file path, finds all versions and returns combined facts

        Returns the internal fact IDs (WME identifiers) that were added
        from a particular source. Useful for debugging and understanding
        what knowledge came from which source.

        Args:
            instance_name: RETER instance name
            source: Source ID or file path to query

        Returns:
            success: Whether operation succeeded
            sources_found: List of source IDs that were queried
            facts_by_source: Dict mapping source IDs to their fact IDs
            total_facts: Total number of facts across all matching sources
            count: Number of matching sources found
        """
        try:
            # Lazy load instance if snapshot is available
            self.instance_manager.ensure_instance_loaded(instance_name)

            # Get or create the RETER instance
            reter = self.instance_manager.get_or_create_instance(instance_name)


            sources_to_query = []

            # Check if this is an exact source ID (contains @) or a file path
            if "@" in source:
                # Exact source ID provided
                sources_to_query = [source]
            else:
                # Treat as file path - find all sources with this path
                source_path = Path(source)

                # Get all sources to find matches
                all_sources, _ = reter.get_all_sources()

                if all_sources:
                    # Find all sources that match this file path
                    for source_id in all_sources:
                        if "@" in source_id:
                            # Parse file path from source ID
                            file_part = source_id.rsplit("@", 1)[0]
                            # Compare paths (handle both absolute and relative)
                            if (Path(file_part).resolve() == source_path.resolve() or
                                file_part == source or
                                Path(file_part) == source_path):
                                sources_to_query.append(source_id)

                    # If no timestamped sources found, try exact match
                    if not sources_to_query and source in all_sources:
                        sources_to_query = [source]

                # If still no matches, try as exact source ID
                if not sources_to_query:
                    sources_to_query = [source]

            # Get facts from all matching sources
            facts_by_source = {}
            total_facts = 0
            all_success = True

            for source_id in sources_to_query:
                try:
                    fact_ids, _, _ = reter.get_facts_from_source(source_id)
                    facts_by_source[source_id] = fact_ids
                    total_facts += len(fact_ids)
                except (RuntimeError, KeyError, ValueError):
                    # RuntimeError: RETER operation failed
                    # KeyError: Source not found
                    # ValueError: Invalid source ID
                    facts_by_source[source_id] = []
                    all_success = False

            # Build response
            if len(sources_to_query) == 0:
                return {
                    "success": False,
                    "sources_found": [],
                    "facts_by_source": {},
                    "total_facts": 0,
                    "count": 0,
                    "error": f"No sources found matching: {source}",
                    "execution_time_ms": 0
                }

            return {
                "success": all_success,
                "sources_found": sources_to_query,
                "facts_by_source": facts_by_source,
                "total_facts": total_facts,
                "count": len(sources_to_query),
                "message": f"Found {total_facts} facts across {len(sources_to_query)} source(s)",
                "execution_time_ms": 0
            }
        except Exception as e:
            return {
                "success": False,
                "sources_found": [],
                "facts_by_source": {},
                "total_facts": 0,
                "count": 0,
                "error": str(e),
                "execution_time_ms": 0
            }
