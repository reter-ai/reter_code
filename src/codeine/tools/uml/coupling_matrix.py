"""Coupling matrix generator.

Generates coupling/cohesion matrix showing coupling strength between classes.
"""

from typing import Dict, Any, List, Optional
from codeine.tools.dataclasses import CouplingMatrixOptions
from .base import UMLGeneratorBase


class CouplingMatrixGenerator(UMLGeneratorBase):
    """Generates coupling matrices showing relationships between classes."""

    def generate(
        self,
        instance_name: str = "default",
        classes: Optional[List[str]] = None,
        max_classes: int = 20,
        threshold: int = 0,
        include_inheritance: bool = True,
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """Generate coupling/cohesion matrix showing coupling strength between classes.

        Args:
            instance_name: RETER instance name
            classes: Optional list of class names to analyze
            max_classes: Maximum classes to include in matrix
            threshold: Minimum coupling to display
            include_inheritance: Include inheritance in coupling calculation
            format: Output format ('json', 'markdown', 'heatmap')

        Returns:
            Dictionary with coupling matrix data
        """
        options = CouplingMatrixOptions(
            threshold=threshold,
            include_inheritance=include_inheritance,
            max_classes=max_classes
        )
        return self._generate_impl(instance_name, classes, format, options)

    def _generate_impl(
        self,
        instance_name: str,
        classes: Optional[List[str]],
        format: str,
        options: CouplingMatrixOptions
    ) -> Dict[str, Any]:
        """Implementation of coupling matrix generation."""
        reter = self.instance_manager.get_or_create_instance(instance_name)

        class_concept = self._concept('Class')
        # Get all classes if not specified
        if not classes:
            class_query = f"""
            SELECT ?className WHERE {{
                ?class type {class_concept} .
                ?class name ?className
            }}
            """
            result = reter.reql(class_query)

            if result and hasattr(result, 'to_pylist'):
                rows = result.to_pylist()
                classes = [row.get('?className') for row in rows if row.get('?className')]
            else:
                classes = []

        if not classes:
            return {
                "success": False,
                "matrix": {},
                "classes": [],
                "message": "No classes found",
                "format": format
            }

        # Limit number of classes
        if len(classes) > options.max_classes:
            # Get coupling for all classes to find most coupled ones
            all_coupling = self._calculate_coupling(reter, classes, options.include_inheritance)

            # Calculate total coupling for each class
            class_coupling_totals = {}
            for cls in classes:
                total = sum(all_coupling.get(cls, {}).values()) + sum(
                    all_coupling.get(other, {}).get(cls, 0) for other in classes if other != cls
                )
                class_coupling_totals[cls] = total

            # Keep top max_classes most coupled classes
            classes = sorted(classes, key=lambda c: class_coupling_totals.get(c, 0), reverse=True)[:options.max_classes]

        # Calculate coupling matrix
        coupling_matrix = self._calculate_coupling(reter, classes, options.include_inheritance)

        # Apply threshold filter
        if options.threshold > 0:
            filtered_matrix = {}
            for cls1 in coupling_matrix:
                filtered_matrix[cls1] = {
                    cls2: strength
                    for cls2, strength in coupling_matrix[cls1].items()
                    if strength >= options.threshold
                }
            coupling_matrix = filtered_matrix

        # Calculate statistics
        total_relationships = sum(len(deps) for deps in coupling_matrix.values())
        high_coupling_count = sum(
            1 for deps in coupling_matrix.values()
            for strength in deps.values()
            if strength >= 8
        )

        # Format output
        if format == "json":
            diagram = None
        elif format == "heatmap":
            diagram = self._render_heatmap(coupling_matrix, classes)
        else:  # markdown
            diagram = self._render_markdown(coupling_matrix, classes, options.threshold)

        return {
            "success": True,
            "matrix": coupling_matrix,
            "classes": classes,
            "total_relationships": total_relationships,
            "high_coupling_count": high_coupling_count,
            "threshold": options.threshold,
            "diagram": diagram,
            "format": format
        }

    def _calculate_coupling(
        self,
        reter: Any,
        classes: List[str],
        include_inheritance: bool
    ) -> Dict[str, Dict[str, int]]:
        """Calculate coupling strength between classes.

        Coupling is calculated based on:
        - Method calls from ClassA to ClassB methods
        - Inheritance relationships (if enabled)

        Args:
            reter: RETER instance
            classes: List of class names to analyze
            include_inheritance: Include inheritance in calculation

        Returns:
            Nested dict: coupling[ClassA][ClassB] = coupling_strength
        """
        coupling = {cls: {} for cls in classes}

        method_concept = self._concept('Method')
        # Query 1: Method calls between classes
        calls_query = f"""
        SELECT ?callerClassName ?calleeClassName WHERE {{
            ?caller type {method_concept} .
            ?caller calls ?callee .
            ?callee type {method_concept} .
            ?caller definedIn ?callerClass .
            ?callee definedIn ?calleeClass .
            ?callerClass name ?callerClassName .
            ?calleeClass name ?calleeClassName
        }}
        """

        result = reter.reql(calls_query)

        if result and hasattr(result, 'to_pylist'):
            for row in result.to_pylist():
                caller_class = row.get('?callerClassName')
                callee_class = row.get('?calleeClassName')

                if caller_class in classes and callee_class in classes and caller_class != callee_class:
                    coupling[caller_class][callee_class] = coupling[caller_class].get(callee_class, 0) + 1

        # Query 2: Inheritance relationships
        if include_inheritance:
            class_concept = self._concept('Class')
            inheritance_query = f"""
            SELECT ?className ?baseName WHERE {{
                ?class type {class_concept} .
                ?class name ?className .
                ?class inheritsFrom ?base .
                ?base name ?baseName
            }}
            """

            result = reter.reql(inheritance_query)

            if result and hasattr(result, 'to_pylist'):
                for row in result.to_pylist():
                    child_class = row.get('?className')
                    base_class = row.get('?baseName')

                    if child_class in classes and base_class in classes:
                        # Inheritance creates strong coupling
                        coupling[child_class][base_class] = coupling[child_class].get(base_class, 0) + 5

        return coupling

    def _get_coupling_level(self, strength: int) -> str:
        """Get coupling level indicator."""
        if strength >= 8:
            return "🔴"  # High
        elif strength >= 4:
            return "🟡"  # Medium
        else:
            return "🟢"  # Low

    def _render_markdown(
        self,
        coupling_matrix: Dict[str, Dict[str, int]],
        classes: List[str],
        threshold: int
    ) -> str:
        """Render coupling matrix as markdown table."""
        lines = ["# Coupling Matrix", ""]

        # Add legend
        lines.append("## Coupling Strength Legend")
        lines.append("- 🟢 **Low** (1-3): Loose coupling - Good")
        lines.append("- 🟡 **Medium** (4-7): Moderate coupling - Review")
        lines.append("- 🔴 **High** (8+): Tight coupling - Refactor!")
        lines.append("")

        if threshold > 0:
            lines.append(f"**Filter**: Showing only coupling >= {threshold}")
            lines.append("")

        # Build matrix table
        header = "| Class |" + "".join(f" {cls[:12]} |" for cls in classes)
        separator = "|-------|" + "|".join("-------" for _ in classes) + "|"

        lines.append(header)
        lines.append(separator)

        # Data rows
        for cls1 in classes:
            row_data = [f"| **{cls1[:12]}**"]
            for cls2 in classes:
                if cls1 == cls2:
                    row_data.append(" - ")
                else:
                    strength = coupling_matrix.get(cls1, {}).get(cls2, 0)
                    if strength == 0:
                        row_data.append(" · ")
                    else:
                        level = self._get_coupling_level(strength)
                        row_data.append(f" {level}{strength} ")

            lines.append("|".join(row_data) + "|")

        lines.append("")

        # Add summary
        lines.append("## Summary")
        total_pairs = len(classes) * (len(classes) - 1)
        coupled_pairs = sum(1 for deps in coupling_matrix.values() for s in deps.values() if s > 0)
        high_coupling = sum(1 for deps in coupling_matrix.values() for s in deps.values() if s >= 8)

        lines.append(f"- **Total class pairs**: {total_pairs}")
        lines.append(f"- **Coupled pairs**: {coupled_pairs} ({100*coupled_pairs//total_pairs if total_pairs > 0 else 0}%)")
        lines.append(f"- **High coupling pairs**: {high_coupling}")

        if high_coupling > 0:
            lines.append("")
            lines.append("### ⚠️ High Coupling Pairs (Refactor Priority)")
            for cls1 in classes:
                for cls2, strength in coupling_matrix.get(cls1, {}).items():
                    if strength >= 8:
                        lines.append(f"- `{cls1}` → `{cls2}`: {strength} (🔴 HIGH)")

        return "\n".join(lines)

    def _render_heatmap(
        self,
        coupling_matrix: Dict[str, Dict[str, int]],
        classes: List[str]
    ) -> str:
        """Render coupling matrix as ASCII heat map."""
        lines = ["# Coupling Heat Map", ""]
        lines.append("```")

        # Column headers
        header = "           " + "".join(f"{i:3}" for i in range(len(classes)))
        lines.append(header)
        lines.append("           " + "---" * len(classes))

        # Matrix rows
        for i, cls1 in enumerate(classes):
            row = f"{i:2} {cls1[:7]:<7} "
            for cls2 in classes:
                if cls1 == cls2:
                    row += "  ·"
                else:
                    strength = coupling_matrix.get(cls1, {}).get(cls2, 0)
                    if strength == 0:
                        row += "  ·"
                    elif strength <= 3:
                        row += "  ░"  # Low
                    elif strength <= 7:
                        row += "  ▒"  # Medium
                    else:
                        row += "  █"  # High
            lines.append(row)

        lines.append("```")
        lines.append("")
        lines.append("**Legend**: · = none, ░ = low (1-3), ▒ = medium (4-7), █ = high (8+)")
        lines.append("")

        # Add class index
        lines.append("## Class Index")
        for i, cls in enumerate(classes):
            lines.append(f"{i:2}. {cls}")

        return "\n".join(lines)
