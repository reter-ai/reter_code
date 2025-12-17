"""Class hierarchy diagram generator.

Generates class inheritance hierarchy diagrams from Python code.
"""

from typing import Dict, Any, Optional
from .base import UMLGeneratorBase


class ClassHierarchyGenerator(UMLGeneratorBase):
    """Generates class hierarchy diagrams showing inheritance relationships."""

    def generate(
        self,
        instance_name: str = "default",
        format: str = "markdown",
        root_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate class hierarchy diagram.

        Args:
            instance_name: RETER instance name
            format: Output format ('json' or 'markdown')
            root_class: Optional root class to start from

        Returns:
            Dictionary with hierarchy data and formatted diagram
        """
        reter = self.instance_manager.get_or_create_instance(instance_name)

        class_concept = self._concept('Class')
        # Query for all classes and their base classes
        query = f"""
        SELECT ?class ?className ?base WHERE {{
            ?class type {class_concept} .
            ?class name ?className .
            OPTIONAL {{
                ?class inheritsFrom ?base
            }}
        }}
        """

        result = reter.reql(query)

        # Build hierarchy structure
        hierarchy = self._build_hierarchy_tree(result, root_class)

        if format == "json":
            return {
                "success": True,
                "format": "json",
                "hierarchy": hierarchy,
                "total_classes": len(hierarchy.get("all_classes", []))
            }
        else:
            # Generate markdown diagram
            diagram = self._render_markdown(hierarchy)
            return {
                "success": True,
                "format": "markdown",
                "diagram": diagram,
                "hierarchy": hierarchy,
                "total_classes": len(hierarchy.get("all_classes", []))
            }

    def _build_hierarchy_tree(
        self,
        query_result: Any,
        root_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build hierarchy tree from query results.

        Args:
            query_result: PyArrow table with class and base class data
            root_class: Optional root class to filter by

        Returns:
            Hierarchy tree structure
        """
        # Convert PyArrow table to dict
        classes = {}
        all_classes = set()

        rows = self._result_to_pylist(query_result)

        for row in rows:
            class_name = row.get('?className')
            base_iri = row.get('?base')

            # Extract base class name from IRI (format: "path/file.py@timestamp.ClassName")
            base_name = None
            if base_iri and isinstance(base_iri, str):
                # Extract class name after the last dot
                base_name = base_iri.split('.')[-1] if '.' in base_iri else None

            if class_name:
                all_classes.add(class_name)
                if class_name not in classes:
                    classes[class_name] = {
                        'name': class_name,
                        'bases': [],
                        'children': []
                    }

                if base_name:
                    classes[class_name]['bases'].append(base_name)
                    all_classes.add(base_name)

                    # Ensure base class exists in dict
                    if base_name not in classes:
                        classes[base_name] = {
                            'name': base_name,
                            'bases': [],
                            'children': []
                        }
                    classes[base_name]['children'].append(class_name)

        # Find root classes (no bases or matching root_class)
        if root_class:
            roots = [classes[root_class]] if root_class in classes else []
        else:
            roots = [c for c in classes.values() if not c['bases']]

        return {
            'roots': roots,
            'all_classes': sorted(list(all_classes)),
            'class_details': classes
        }

    def _render_markdown(self, hierarchy: Dict[str, Any]) -> str:
        """Render hierarchy as markdown tree.

        Args:
            hierarchy: Hierarchy tree structure

        Returns:
            Markdown formatted tree
        """
        lines = ["# Class Hierarchy\n"]

        def render_tree(class_info: Dict[str, Any], indent: int = 0):
            """Recursively render class tree."""
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            lines.append(f"{prefix}{class_info['name']}")

            # Render children
            class_details = hierarchy['class_details']
            for child_name in sorted(class_info.get('children', [])):
                if child_name in class_details:
                    render_tree(class_details[child_name], indent + 1)

        # Render each root
        for root in hierarchy['roots']:
            render_tree(root)

        lines.append(f"\n**Total classes:** {len(hierarchy['all_classes'])}")

        return "\n".join(lines)
