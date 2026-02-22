"""
CADSL Mermaid Steps.

Contains dataclass configs and the render step for Mermaid diagram generation:
- FlowchartConfig, SequenceConfig, ClassDiagramConfig, PieChartConfig
- StateDiagramConfig, ERDiagramConfig, BlockBetaConfig
- MermaidConfig: Unified configuration grouping all diagram types
- RenderMermaidStep: Renders pipeline data to Mermaid diagrams
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FlowchartConfig:
    """Configuration for flowchart diagrams.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    nodes: Optional[str] = None
    edges_from: Optional[str] = None
    edges_to: Optional[str] = None
    direction: str = "TB"


@dataclass
class SequenceConfig:
    """Configuration for sequence diagrams.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    participants: Optional[str] = None
    messages_from: Optional[str] = None
    messages_to: Optional[str] = None
    messages_label: Optional[str] = None


@dataclass
class ClassDiagramConfig:
    """Configuration for class diagrams.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    classes: Optional[str] = None
    methods: Optional[str] = None
    attributes: Optional[str] = None
    inheritance_from: Optional[str] = None
    inheritance_to: Optional[str] = None
    composition_from: Optional[str] = None
    composition_to: Optional[str] = None
    association_from: Optional[str] = None
    association_to: Optional[str] = None


@dataclass
class PieChartConfig:
    """Configuration for pie charts.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    labels: Optional[str] = None
    values: Optional[str] = None


@dataclass
class StateDiagramConfig:
    """Configuration for state diagrams.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    states: Optional[str] = None
    transitions_from: Optional[str] = None
    transitions_to: Optional[str] = None


@dataclass
class ERDiagramConfig:
    """Configuration for ER diagrams.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    entities: Optional[str] = None
    relationships: Optional[str] = None


@dataclass
class BlockBetaConfig:
    """Configuration for block-beta diagrams.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    groups: Optional[str] = None
    nodes: Optional[str] = None
    columns: int = 4
    color: Optional[str] = None
    max_per_group: int = 20


@dataclass
class MermaidConfig:
    """Unified configuration for all Mermaid diagram types.

    Groups related parameters by diagram type, reducing the parameter count
    from 27 individual parameters to a single structured config object.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    mermaid_type: str = "flowchart"
    title: Optional[str] = None
    flowchart: FlowchartConfig = field(default_factory=FlowchartConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    class_diagram: ClassDiagramConfig = field(default_factory=ClassDiagramConfig)
    pie_chart: PieChartConfig = field(default_factory=PieChartConfig)
    state_diagram: StateDiagramConfig = field(default_factory=StateDiagramConfig)
    er_diagram: ERDiagramConfig = field(default_factory=ERDiagramConfig)
    block_beta: BlockBetaConfig = field(default_factory=BlockBetaConfig)

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "MermaidConfig":
        """Create MermaidConfig from a specification dict.

        This factory method provides backward compatibility with dict-based
        configuration, extracting and grouping parameters appropriately.

        Args:
            spec: Dictionary with mermaid configuration parameters

        Returns:
            MermaidConfig instance with grouped parameters
        """
        return cls(
            mermaid_type=spec.get("mermaid_type", "flowchart"),
            title=spec.get("title"),
            flowchart=FlowchartConfig(
                nodes=spec.get("nodes"),
                edges_from=spec.get("edges_from"),
                edges_to=spec.get("edges_to"),
                direction=spec.get("direction", "TB"),
            ),
            sequence=SequenceConfig(
                participants=spec.get("participants"),
                messages_from=spec.get("messages_from"),
                messages_to=spec.get("messages_to"),
                messages_label=spec.get("messages_label"),
            ),
            class_diagram=ClassDiagramConfig(
                classes=spec.get("classes"),
                methods=spec.get("methods"),
                attributes=spec.get("attributes"),
                inheritance_from=spec.get("inheritance_from"),
                inheritance_to=spec.get("inheritance_to"),
                composition_from=spec.get("composition_from"),
                composition_to=spec.get("composition_to"),
                association_from=spec.get("association_from"),
                association_to=spec.get("association_to"),
            ),
            pie_chart=PieChartConfig(
                labels=spec.get("labels"),
                values=spec.get("values"),
            ),
            state_diagram=StateDiagramConfig(
                states=spec.get("states"),
                transitions_from=spec.get("transitions_from"),
                transitions_to=spec.get("transitions_to"),
            ),
            er_diagram=ERDiagramConfig(
                entities=spec.get("entities"),
                relationships=spec.get("relationships"),
            ),
            block_beta=BlockBetaConfig(
                groups=spec.get("groups"),
                nodes=spec.get("nodes"),
                columns=spec.get("columns", 4),
                color=spec.get("color"),
                max_per_group=spec.get("max_per_group", 20),
            ),
        )


class RenderMermaidStep:
    """
    Render data to a Mermaid diagram.

    Syntax: render_mermaid { type: flowchart, nodes: name, edges: from -> to }
    Supports: flowchart, sequence, class, gantt, state, er, pie

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, config: MermaidConfig):
        """Initialize RenderMermaidStep with a MermaidConfig.

        Args:
            config: MermaidConfig instance containing all diagram parameters
        """
        self.config = config
        # Expose commonly used attributes for convenience
        self.mermaid_type = config.mermaid_type
        self.title = config.title

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "RenderMermaidStep":
        """Create RenderMermaidStep from a specification dict.

        This factory method provides backward compatibility with dict-based
        configuration used by the CADSL transformer and loader.

        Args:
            spec: Dictionary with mermaid configuration parameters

        Returns:
            RenderMermaidStep instance
        """
        return cls(MermaidConfig.from_spec(spec))

    def execute(self, data, ctx=None):
        """Render to Mermaid diagram."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if self.mermaid_type == "flowchart":
                diagram = self._render_flowchart(data)
            elif self.mermaid_type == "sequence":
                diagram = self._render_sequence(data)
            elif self.mermaid_type == "class":
                diagram = self._render_class(data)
            elif self.mermaid_type == "gantt":
                diagram = self._render_gantt(data)
            elif self.mermaid_type == "pie":
                diagram = self._render_pie(data)
            elif self.mermaid_type == "state":
                diagram = self._render_state(data)
            elif self.mermaid_type == "er":
                diagram = self._render_er(data)
            elif self.mermaid_type == "block_beta":
                diagram = self._render_block_beta(data)
            else:
                diagram = self._render_flowchart(data)

            return pipeline_ok({"diagram": diagram, "format": "mermaid", "type": self.mermaid_type})
        except Exception as e:
            return pipeline_err("render_mermaid", f"Mermaid rendering failed: {e}", e)

    def _render_flowchart(self, data):
        """Render a flowchart."""
        fc = self.config.flowchart
        lines = [f"flowchart {fc.direction}"]

        # Auto-detect node field if not specified
        nodes_field = fc.nodes
        if not nodes_field and not fc.edges_from and data:
            # Try to auto-detect from first row's fields
            first_row = data[0] if data else {}
            # Prefer fields named 'name', 'node', 'layer', or first string field
            # Also check for REQL-style ?name variants
            for candidate in ['name', 'node', 'layer', 'label', 'id',
                              '?name', '?node', '?layer', '?label', '?id']:
                if candidate in first_row:
                    nodes_field = candidate
                    break
            if not nodes_field and first_row:
                # Use first non-count field as nodes
                for key in first_row.keys():
                    if 'count' not in key.lower():
                        nodes_field = key
                        break
                if not nodes_field:
                    # Fallback to first field
                    nodes_field = list(first_row.keys())[0]

        # Collect unique nodes and edges
        nodes = set()
        edges = set()
        for row in data:
            if fc.edges_from and fc.edges_to:
                from_val = row.get(fc.edges_from)
                to_val = row.get(fc.edges_to)
                if from_val and to_val:
                    nodes.add(from_val)
                    nodes.add(to_val)
                    edges.add((from_val, to_val))
            elif nodes_field:
                nodes.add(row.get(nodes_field))

        def _sanitize_mermaid_id(name):
            """Sanitize a string for use as a Mermaid node ID."""
            import re
            return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

        # Render edges (nodes are implicit in edges)
        if edges:
            for from_val, to_val in sorted(edges):
                safe_from = _sanitize_mermaid_id(from_val)
                safe_to = _sanitize_mermaid_id(to_val)
                lines.append(f'    {safe_from}["{from_val}"] --> {safe_to}["{to_val}"]')
        elif nodes:
            # Render standalone nodes when no edges defined
            for node_val in sorted(n for n in nodes if n):
                safe_node = _sanitize_mermaid_id(node_val)
                lines.append(f'    {safe_node}["{node_val}"]')

        return "\n".join(lines)

    def _render_sequence(self, data):
        """Render a sequence diagram."""
        import re
        seq = self.config.sequence
        lines = ["sequenceDiagram"]

        def _get(row, field):
            """Get field, trying both plain and ?-prefixed keys."""
            if not field:
                return None
            v = row.get(field)
            if v is None:
                v = row.get("?" + field)
            return v

        def _safe_participant(name):
            return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

        # Collect participants
        participants = set()
        messages = []
        for row in data:
            if seq.messages_from and seq.messages_to:
                from_val = _get(row, seq.messages_from)
                to_val = _get(row, seq.messages_to)
                label = _get(row, seq.messages_label) or "" if seq.messages_label else ""
                if from_val and to_val:
                    participants.add(from_val)
                    participants.add(to_val)
                    messages.append((from_val, to_val, label))
            elif seq.participants:
                participants.add(_get(row, seq.participants))

        # Render participants
        for p in sorted(participants):
            if p:
                safe_p = _safe_participant(p)
                lines.append(f"    participant {safe_p} as {p}")

        # Render messages
        for from_val, to_val, label in messages:
            safe_from = _safe_participant(from_val)
            safe_to = _safe_participant(to_val)
            lines.append(f"    {safe_from}->>+{safe_to}: {label}")

        return "\n".join(lines)

    def _render_class(self, data):
        """Render a class diagram with methods, attributes, and relationships."""
        cd = self.config.class_diagram
        lines = ["classDiagram"]

        # Aggregate class data
        class_info = {}
        inheritance_rels = set()  # parent <|-- child
        composition_rels = set()  # owner *-- owned (strong ownership)
        association_rels = set()  # from --> to (uses/references)

        for row in data:
            class_name = row.get(cd.classes) if cd.classes else row.get("class_name") or row.get("name")
            if not class_name:
                continue

            if class_name not in class_info:
                class_info[class_name] = {"methods": set(), "attributes": set()}

            # Collect methods
            if cd.methods:
                method = row.get(cd.methods)
                if method:
                    class_info[class_name]["methods"].add(method)

            # Collect attributes
            if cd.attributes:
                attr = row.get(cd.attributes)
                if attr:
                    class_info[class_name]["attributes"].add(attr)

            # Collect inheritance
            if cd.inheritance_from and cd.inheritance_to:
                parent = row.get(cd.inheritance_from)
                child = row.get(cd.inheritance_to)
                if parent and child:
                    inheritance_rels.add((parent, child))
            elif row.get("parent_name") or row.get("inherits_from"):
                parent = row.get("parent_name") or row.get("inherits_from")
                if parent:
                    inheritance_rels.add((parent, class_name))

            # Collect composition (owner *-- owned)
            if cd.composition_from and cd.composition_to:
                owner = row.get(cd.composition_from)
                owned = row.get(cd.composition_to)
                if owner and owned:
                    composition_rels.add((owner, owned))
            elif row.get("composition_from") and row.get("composition_to"):
                owner = row.get("composition_from")
                owned = row.get("composition_to")
                if owner and owned:
                    composition_rels.add((owner, owned))

            # Collect associations (from --> to)
            if cd.association_from and cd.association_to:
                from_cls = row.get(cd.association_from)
                to_cls = row.get(cd.association_to)
                if from_cls and to_cls:
                    association_rels.add((from_cls, to_cls))
            elif row.get("assoc_from") and row.get("assoc_to"):
                from_cls = row.get("assoc_from")
                to_cls = row.get("assoc_to")
                if from_cls and to_cls:
                    association_rels.add((from_cls, to_cls))

        # Render classes with members
        LBRACE = "{"
        RBRACE = "}"
        for cls_name in sorted(class_info.keys()):
            info = class_info[cls_name]
            safe_name = str(cls_name).replace(" ", "_").replace("-", "_").replace(".", "_")
            lines.append(f"    class {safe_name} {LBRACE}")
            for attr in sorted(info["attributes"]):
                lines.append(f"        +{attr}")
            for method in sorted(info["methods"]):
                lines.append(f"        +{method}()")
            lines.append(f"    {RBRACE}")

        # Render inheritance relationships (parent <|-- child)
        for parent, child in sorted(inheritance_rels):
            safe_parent = str(parent).replace(" ", "_").replace("-", "_").replace(".", "_")
            safe_child = str(child).replace(" ", "_").replace("-", "_").replace(".", "_")
            lines.append(f"    {safe_parent} <|-- {safe_child}")

        # Render composition relationships (owner *-- owned)
        for owner, owned in sorted(composition_rels):
            safe_owner = str(owner).replace(" ", "_").replace("-", "_").replace(".", "_")
            safe_owned = str(owned).replace(" ", "_").replace("-", "_").replace(".", "_")
            lines.append(f"    {safe_owner} *-- {safe_owned}")

        # Render association relationships (from --> to)
        for from_cls, to_cls in sorted(association_rels):
            safe_from = str(from_cls).replace(" ", "_").replace("-", "_").replace(".", "_")
            safe_to = str(to_cls).replace(" ", "_").replace("-", "_").replace(".", "_")
            lines.append(f"    {safe_from} --> {safe_to}")

        return "\n".join(lines)

    def _render_gantt(self, data):
        """Render a gantt chart."""
        lines = ["gantt"]
        if self.title:
            lines.append(f"    title {self.title}")
        lines.append("    dateFormat YYYY-MM-DD")

        # Process task data
        for row in data:
            name = row.get("name") or row.get("task")
            start = row.get("start") or row.get("start_date")
            end = row.get("end") or row.get("end_date")
            if name and start:
                if end:
                    lines.append(f"    {name} : {start}, {end}")
                else:
                    lines.append(f"    {name} : {start}, 1d")

        return "\n".join(lines)

    def _render_pie(self, data):
        """Render a pie chart."""
        pie = self.config.pie_chart
        lines = ["pie showData"]
        if self.title:
            lines[0] = f'pie showData title {self.title}'

        for row in data:
            label = row.get(pie.labels) if pie.labels else row.get("label") or row.get("name")
            value = row.get(pie.values) if pie.values else row.get("value") or row.get("count")
            if label and value:
                lines.append(f'    "{label}" : {value}')

        return "\n".join(lines)

    def _render_state(self, data):
        """Render a state diagram."""
        sd = self.config.state_diagram
        lines = ["stateDiagram-v2"]

        transitions = set()
        states = set()

        for row in data:
            if sd.states:
                state = row.get(sd.states)
                if state:
                    states.add(state)

            if sd.transitions_from and sd.transitions_to:
                from_state = row.get(sd.transitions_from)
                to_state = row.get(sd.transitions_to)
                if from_state and to_state:
                    transitions.add((from_state, to_state))
                    states.add(from_state)
                    states.add(to_state)

        for from_s, to_s in sorted(transitions):
            lines.append(f"    {from_s} --> {to_s}")

        return "\n".join(lines)

    def _render_er(self, data):
        """Render an ER diagram."""
        er = self.config.er_diagram
        lines = ["erDiagram"]

        for row in data:
            entity = row.get(er.entities) if er.entities else row.get("entity")
            if entity:
                lines.append(f"    {entity}")

        return "\n".join(lines)

    def _render_block_beta(self, data):
        """Render a block-beta diagram with grouped grid layout."""
        from collections import OrderedDict

        bb = self.config.block_beta
        lines = ["block-beta", "    columns 1"]

        # Default color palette for groups without explicit colors
        PALETTE = [
            "#4CAF50", "#2196F3", "#9C27B0", "#FF9800",
            "#795548", "#607D8B", "#9E9E9E", "#E91E63",
        ]

        def _safe_id(name):
            import re
            return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

        def _get_field(row, field_name):
            """Get field value, handling ?-prefixed REQL columns."""
            if not field_name:
                return None
            v = row.get(field_name)
            if v is None:
                v = row.get("?" + field_name)
            return v

        # Auto-detect field names
        groups_field = bb.groups
        nodes_field = bb.nodes or (self.config.flowchart.nodes if self.config.flowchart.nodes else None)
        color_field = bb.color

        if data and not groups_field:
            first = data[0]
            for c in ["layer", "group", "category", "?layer", "?group", "?category"]:
                if c in first:
                    groups_field = c
                    break

        if data and not nodes_field:
            first = data[0]
            for c in ["name", "class_name", "node", "?name", "?class_name", "?node"]:
                if c in first:
                    nodes_field = c
                    break

        # Group rows
        groups = OrderedDict()
        group_totals = OrderedDict()
        for row in data:
            g = _get_field(row, groups_field)
            n = _get_field(row, nodes_field)
            if not g or not n:
                continue
            if g not in groups:
                groups[g] = []
                group_totals[g] = 0
            group_totals[g] += 1
            if len(groups[g]) < bb.max_per_group:
                groups[g].append(n)
            # Capture color from first row of each group
            if color_field and g not in {k: v for k, v in groups.items() if hasattr(v, '_color')}:
                c = _get_field(row, color_field)
                if c and not hasattr(groups[g], '_color'):
                    groups[g] = type('ColorList', (list,), {'_color': c})(groups[g])

        # Simpler color tracking
        group_colors = {}
        for row in data:
            if color_field:
                g = _get_field(row, groups_field)
                c = _get_field(row, color_field)
                if g and c and g not in group_colors:
                    group_colors[g] = c

        # Render each group as a block
        styled = []
        header_styles = []
        for gi, (group_name, nodes) in enumerate(groups.items()):
            if isinstance(nodes, list):
                node_list = nodes
            else:
                node_list = list(nodes)

            safe_gid = _safe_id(group_name)
            total = group_totals.get(group_name, len(node_list))
            overflow = total - len(node_list)
            color = group_colors.get(group_name, PALETTE[gi % len(PALETTE)])

            lines.append(f'    block:{safe_gid}')
            lines.append(f"        columns {bb.columns}")

            # Header row spanning all columns
            header_id = safe_gid + "_hdr"
            lines.append(f'        {header_id}["{group_name} ({total})"]:{bb.columns}')
            raw_color = color.split("fill:")[1].split(",")[0] if "fill:" in color else color
            header_styles.append((header_id, raw_color))

            seen = set()
            for node in node_list:
                safe_nid = _safe_id(node)
                if safe_nid in seen:
                    continue
                seen.add(safe_nid)
                lines.append(f'        {safe_nid}["{node}"]')

            if overflow > 0:
                lines.append(f'        {safe_gid}_more["... +{overflow} more"]')

            lines.append("    end")
            styled.append((safe_gid, color))

        # Apply styles
        for safe_gid, color in styled:
            if color.startswith("fill:"):
                lines.append(f"    style {safe_gid} {color}")
            else:
                lines.append(f"    style {safe_gid} fill:{color},color:#fff")

        # Style header rows to match block color
        for hdr_id, hdr_color in header_styles:
            lines.append(f"    style {hdr_id} fill:{hdr_color},color:#fff,font-weight:bold")

        return "\n".join(lines)
