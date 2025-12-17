"""Sequence diagram generator.

Generates sequence diagrams showing method calls between classes.
"""

import re
from typing import Dict, Any, List, Optional
from codeine.tools.dataclasses import SequenceDiagramOptions
from .base import UMLGeneratorBase


class SequenceDiagramGenerator(UMLGeneratorBase):
    """Generates sequence diagrams showing method call interactions."""

    def generate(
        self,
        classes: List[str],
        instance_name: str = "default",
        format: str = "markdown",
        entry_point: Optional[str] = None,
        max_depth: int = 10,
        exclude_patterns: Optional[List[str]] = None,
        include_only_classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate sequence diagram showing method calls.

        Args:
            classes: List of class names to include
            instance_name: RETER instance name
            format: Output format ('json' or 'markdown')
            entry_point: Optional entry point method name
            max_depth: Maximum call depth to traverse
            exclude_patterns: List of regex patterns to exclude methods
            include_only_classes: If specified, only show calls involving these classes

        Returns:
            Dictionary with sequence diagram data
        """
        reter = self.instance_manager.get_or_create_instance(instance_name)

        method_concept = self._concept('Method')
        # Query for method calls between classes
        query = f"""
        SELECT ?caller ?callerName ?callee ?calleeName ?callerClassName ?calleeClassName WHERE {{
            ?caller type {method_concept} .
            ?caller name ?callerName .
            ?caller calls ?callee .
            ?callee type {method_concept} .
            ?callee name ?calleeName .
            ?caller definedIn ?callerClass .
            ?callee definedIn ?calleeClass .
            ?callerClass name ?callerClassName .
            ?calleeClass name ?calleeClassName
        }}
        """

        result = reter.reql(query)

        # Build sequence data with filtering
        options = SequenceDiagramOptions(
            entry_point=entry_point,
            max_depth=max_depth,
            exclude_patterns=exclude_patterns,
            include_only_classes=include_only_classes
        )
        sequences = self._build_sequence_data(result, classes, options)

        if format == "json":
            return {
                "success": True,
                "format": "json",
                "sequences": sequences,
                "total_interactions": len(sequences)
            }
        else:
            # Generate Mermaid sequence diagram
            diagram = self._render_markdown(sequences, classes)
            return {
                "success": True,
                "format": "markdown",
                "diagram": diagram,
                "sequences": sequences,
                "total_interactions": len(sequences)
            }

    def _build_sequence_data(
        self,
        query_result: Any,
        classes: List[str],
        options: SequenceDiagramOptions
    ) -> List[Dict[str, Any]]:
        """Build sequence diagram data from query results.

        Args:
            query_result: PyArrow table with call data
            classes: List of classes to include
            options: SequenceDiagramOptions with filtering options

        Returns:
            List of interaction dictionaries
        """
        sequences = []
        exclude_regexes = []

        # Compile exclude patterns
        if options.exclude_patterns:
            exclude_regexes = [re.compile(pattern) for pattern in options.exclude_patterns]

        rows = self._result_to_pylist(query_result)

        for row in rows:
            caller_class = row.get('?callerClassName')
            callee_class = row.get('?calleeClassName')
            caller_method = row.get('?callerName')
            callee_method = row.get('?calleeName')

            # Apply exclude patterns
            if exclude_regexes:
                if any(regex.match(caller_method) for regex in exclude_regexes):
                    continue
                if any(regex.match(callee_method) for regex in exclude_regexes):
                    continue

            # Filter by include_only_classes if specified
            if options.include_only_classes:
                if caller_class not in options.include_only_classes and callee_class not in options.include_only_classes:
                    continue

            # Filter by specified classes
            if caller_class in classes or callee_class in classes:
                # Filter by entry point if specified
                if options.entry_point is None or caller_method == options.entry_point:
                    sequences.append({
                        'caller_class': caller_class,
                        'caller_method': caller_method,
                        'callee_class': callee_class,
                        'callee_method': callee_method,
                        'depth': 1  # TODO: Implement proper depth tracking
                    })

        return sequences

    def _render_markdown(
        self,
        sequences: List[Dict[str, Any]],
        classes: List[str]
    ) -> str:
        """Render sequence diagram as Mermaid markdown.

        Args:
            sequences: List of interaction dictionaries
            classes: List of classes involved

        Returns:
            Mermaid formatted sequence diagram
        """
        lines = ["```mermaid", "sequenceDiagram", ""]

        # Declare participants
        for cls in sorted(classes):
            lines.append(f"    participant {cls}")

        lines.append("")

        # Render interactions with activation boxes and returns
        for seq in sequences:
            caller = seq['caller_class']
            callee = seq['callee_class']
            method = seq['callee_method']

            # Check if this is a self-call
            is_self_call = (caller == callee)

            if is_self_call:
                # Self-call: activate without deactivating caller
                lines.append(f"    {caller}->>{caller}: {method}()")
            else:
                # Regular call: activate callee and return
                lines.append(f"    {caller}->>+{callee}: {method}()")
                lines.append(f"    {callee}-->>-{caller}: return")

        lines.append("```")

        return "\n".join(lines)
