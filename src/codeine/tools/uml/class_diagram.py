"""Class diagram generator.

Generates class diagrams showing attributes, methods, and relationships.
"""

from typing import Dict, Any, List, Optional
from .base import UMLGeneratorBase


class ClassDiagramGenerator(UMLGeneratorBase):
    """Generates class diagrams with attributes and methods."""

    def generate(
        self,
        classes: List[str],
        instance_name: str = "default",
        format: str = "markdown",
        include_methods: bool = True,
        include_attributes: bool = True
    ) -> Dict[str, Any]:
        """Generate class diagram for specified classes.

        Args:
            classes: List of class names to include
            instance_name: RETER instance name
            format: Output format ('json' or 'markdown')
            include_methods: Include methods in diagram
            include_attributes: Include attributes in diagram

        Returns:
            Dictionary with class diagram data
        """
        reter = self.instance_manager.get_or_create_instance(instance_name)

        class_data = []

        for class_name in classes:
            # Query class details
            class_info = self._get_class_details(
                reter, class_name, include_methods, include_attributes
            )
            if class_info:
                class_data.append(class_info)

        if format == "json":
            return {
                "success": True,
                "format": "json",
                "classes": class_data,
                "total_classes": len(class_data)
            }
        else:
            # Generate Mermaid-compatible markdown
            diagram = self._render_markdown(class_data)
            return {
                "success": True,
                "format": "markdown",
                "diagram": diagram,
                "classes": class_data,
                "total_classes": len(class_data)
            }

    def _get_class_details(
        self,
        reter: Any,
        class_name: str,
        include_methods: bool,
        include_attributes: bool
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information about a class.

        Args:
            reter: RETER instance
            class_name: Class name to query
            include_methods: Include methods
            include_attributes: Include attributes

        Returns:
            Class details dictionary or None
        """
        class_concept = self._concept('Class')
        # Query class
        class_query = f"""
        SELECT ?class ?base WHERE {{
            ?class type {class_concept} .
            ?class name "{class_name}" .
            OPTIONAL {{
                ?class inheritsFrom ?base
            }}
        }}
        """

        class_result = reter.reql(class_query)

        if not class_result or (hasattr(class_result, 'num_rows') and class_result.num_rows == 0):
            return None

        class_info = {
            'name': class_name,
            'bases': [],
            'methods': [],
            'attributes': []
        }

        # Extract bases (extract class name from IRI)
        if hasattr(class_result, 'to_pylist'):
            rows = class_result.to_pylist()
            for row in rows:
                base_iri = row.get('?base')
                if base_iri and isinstance(base_iri, str):
                    # Extract class name after the last dot
                    base_name = base_iri.split('.')[-1] if '.' in base_iri else None
                    if base_name and base_name not in class_info['bases']:
                        class_info['bases'].append(base_name)

        # Query methods if requested
        if include_methods:
            method_concept = self._concept('Method')
            methods_query = f"""
            SELECT ?method ?name WHERE {{
                ?method type {method_concept} .
                ?method definedIn ?class .
                ?class name "{class_name}" .
                ?method name ?name
            }}
            """
            methods_result = reter.reql(methods_query)

            if hasattr(methods_result, 'to_pylist'):
                for row in methods_result.to_pylist():
                    method_name = row.get('?name')
                    if method_name:
                        class_info['methods'].append(method_name)

        # Query attributes if requested
        if include_attributes:
            attr_concept = self._concept('Attribute')
            attrs_query = f"""
            SELECT ?attr ?name WHERE {{
                ?attr type {attr_concept} .
                ?attr definedIn ?class .
                ?class name "{class_name}" .
                ?attr name ?name
            }}
            """
            attrs_result = reter.reql(attrs_query)

            if hasattr(attrs_result, 'to_pylist'):
                for row in attrs_result.to_pylist():
                    attr_name = row.get('?name')
                    if attr_name:
                        class_info['attributes'].append(attr_name)

        return class_info

    def _render_markdown(self, class_data: List[Dict[str, Any]]) -> str:
        """Render class diagram as Mermaid markdown.

        Args:
            class_data: List of class information dictionaries

        Returns:
            Mermaid formatted diagram
        """
        lines = ["```mermaid", "classDiagram", ""]

        # Render inheritance relationships first (Mermaid convention)
        for cls in class_data:
            for base in cls.get('bases', []):
                lines.append(f"    {base} <|-- {cls['name']}")

        if any(cls.get('bases') for cls in class_data):
            lines.append("")

        # Render each class with attributes and methods
        for cls in class_data:
            class_name = cls['name']

            # Use class block syntax for classes with attributes/methods
            if cls.get('attributes') or cls.get('methods'):
                lines.append(f"    class {class_name} {{")

                # Attributes (Mermaid format: +type attributeName or just +attributeName)
                if cls.get('attributes'):
                    for attr in sorted(cls['attributes']):
                        # Assume public attributes (+), private would use (-)
                        visibility = '+' if not attr.startswith('_') else '-'
                        lines.append(f"        {visibility}{attr}")

                # Methods (Mermaid format: +methodName())
                if cls.get('methods'):
                    for method in sorted(cls['methods']):
                        # Determine visibility
                        if method.startswith('__') and not method.endswith('__'):
                            visibility = '-'  # private
                        elif method.startswith('_') and not method.startswith('__'):
                            visibility = '-'  # protected
                        else:
                            visibility = '+'  # public
                        lines.append(f"        {visibility}{method}()")

                lines.append("    }")
            else:
                # Empty class - just declare it
                lines.append(f"    class {class_name}")

            lines.append("")

        lines.append("```")

        return "\n".join(lines)
