# CADSL Native Multi-Output Design

## Executive Summary

This document proposes native CADSL support for multiple diagramming formats and comprehensive reporting capabilities, eliminating the need for Python blocks in 95%+ of visualization tools.

---

## 1. Current State Analysis

### Current Output Capabilities
| Feature | Status | Limitations |
|---------|--------|-------------|
| `render_mermaid` | Partial | Only flowchart/sequence, limited edge syntax |
| `pivot` | Supported | JSON output only |
| `emit` | Supported | Single key output |
| Python fallback | Supported | Breaks declarative model |

### Tools Still Requiring Python for Output
- `sequence_diagram.cadsl` - Complex participant/message logic
- `call_graph.cadsl` - BFS traversal + Mermaid generation
- `class_diagram.cadsl` - Aggregation + class member formatting

---

## 2. Proposed Grammar Extensions

### 2.1 Unified Render Step

```lark
// Unified render step with format negotiation
render_step: "render" "{" render_spec "}"

render_spec: render_target ("," render_option)*

render_target: "diagram" ":" diagram_spec     -> render_diagram
             | "table" ":" table_spec         -> render_table
             | "chart" ":" chart_spec         -> render_chart
             | "report" ":" report_spec       -> render_report
             | "format" ":" STRING            -> render_raw
```

### 2.2 Multi-Format Diagram Rendering

```lark
// Diagram specification with format negotiation
diagram_spec: diagram_format "(" diagram_params ")"

diagram_format: "mermaid"     -> fmt_mermaid
              | "plantuml"    -> fmt_plantuml
              | "graphviz"    -> fmt_graphviz
              | "d2"          -> fmt_d2
              | "ascii"       -> fmt_ascii

// Each format supports multiple diagram types
diagram_params: diagram_param ("," diagram_param)*

diagram_param: "type" ":" diagram_type
             | "nodes" ":" node_spec
             | "edges" ":" edge_spec
             | "groups" ":" group_spec
             | "direction" ":" direction
             | "title" ":" expr
             | "style" ":" style_spec
```

### 2.3 Enhanced Mermaid Support

```lark
// Extended Mermaid types
mermaid_type: "flowchart"    -> mermaid_flowchart
            | "sequence"     -> mermaid_sequence
            | "class"        -> mermaid_class
            | "state"        -> mermaid_state
            | "er"           -> mermaid_er
            | "gantt"        -> mermaid_gantt
            | "pie"          -> mermaid_pie
            | "mindmap"      -> mermaid_mindmap
            | "timeline"     -> mermaid_timeline
            | "quadrant"     -> mermaid_quadrant
            | "requirement"  -> mermaid_requirement
            | "git"          -> mermaid_git

// Class diagram members
class_members: "members" ":" "{" member_spec "}"
member_spec: "methods" ":" NAME ("," "attributes" ":" NAME)?

// Sequence diagram messages
sequence_spec: "participants" ":" NAME
             | "messages" ":" message_edge ("," message_edge)*
message_edge: NAME arrow_type NAME ":" NAME
arrow_type: "->>"   -> sync_call
          | "-->"   -> async_call
          | "->"    -> simple
          | "--x"   -> lost
```

### 2.4 Table Rendering

```lark
// Table rendering for reports
table_spec: "table" "(" table_params ")"

table_params: table_param ("," table_param)*

table_param: "columns" ":" column_list
           | "format" ":" table_format
           | "title" ":" expr
           | "summary" ":" summary_spec
           | "sort" ":" sort_spec
           | "group_by" ":" NAME
           | "totals" ":" bool_val

column_list: "[" column_def ("," column_def)* "]"

column_def: NAME                           -> col_simple
          | NAME "as" STRING               -> col_alias
          | NAME ":" column_options        -> col_with_options

column_options: "{" col_option ("," col_option)* "}"
col_option: "align" ":" align_val
          | "width" ":" INT
          | "format" ":" STRING

table_format: "markdown"   -> tbl_markdown
            | "html"       -> tbl_html
            | "csv"        -> tbl_csv
            | "json"       -> tbl_json
            | "ascii"      -> tbl_ascii
```

### 2.5 Chart Rendering

```lark
// Chart types for data visualization
chart_spec: "chart" "(" chart_params ")"

chart_params: chart_param ("," chart_param)*

chart_param: "type" ":" chart_type
           | "x" ":" NAME
           | "y" ":" NAME
           | "series" ":" NAME
           | "title" ":" expr
           | "format" ":" chart_format
           | "colors" ":" color_list

chart_type: "bar"      -> chart_bar
          | "line"     -> chart_line
          | "pie"      -> chart_pie
          | "scatter"  -> chart_scatter
          | "heatmap"  -> chart_heatmap
          | "treemap"  -> chart_treemap
          | "radar"    -> chart_radar

chart_format: "mermaid"  -> chart_mermaid
            | "ascii"    -> chart_ascii
            | "svg"      -> chart_svg
            | "vega"     -> chart_vegalite
```

### 2.6 Report Composition

```lark
// Compose multiple outputs into a report
report_spec: "report" "(" report_params ")"

report_params: report_param ("," report_param)*

report_param: "title" ":" expr
            | "sections" ":" section_list
            | "format" ":" report_format
            | "template" ":" STRING

section_list: "[" section ("," section)* "]"

section: "{" section_content "}"
section_content: section_field ("," section_field)*
section_field: "type" ":" section_type
             | "title" ":" expr
             | "data" ":" NAME
             | "options" ":" section_options

section_type: "heading"   -> sec_heading
            | "text"      -> sec_text
            | "table"     -> sec_table
            | "chart"     -> sec_chart
            | "diagram"   -> sec_diagram
            | "metrics"   -> sec_metrics
            | "list"      -> sec_list

report_format: "markdown" -> rpt_markdown
             | "html"     -> rpt_html
             | "json"     -> rpt_json
```

### 2.7 Multi-Output Emit

```lark
// Emit multiple named outputs
emit_step: "emit" "{" emit_spec "}"

emit_spec: NAME                                    -> emit_single
         | emit_field ("," emit_field)*            -> emit_multiple

emit_field: NAME                                   -> emit_key
          | NAME ":" NAME                          -> emit_named
          | NAME ":" render_inline                 -> emit_rendered

// Inline rendering for emit
render_inline: "diagram" "(" diagram_params ")"
             | "table" "(" table_params ")"
             | "chart" "(" chart_params ")"
```

---

## 3. CADSL Examples

### 3.1 Class Diagram (Native)

```cadsl
diagram class_diagram() {
    param classes: list = [];
    param include_methods: bool = true;
    param include_attributes: bool = true;

    reql {
        SELECT ?class ?class_name ?method_name ?attr_name ?parent_name
        WHERE {
            ?class type {Class} .
            ?class name ?class_name .
            OPTIONAL { ?method definedIn ?class . ?method name ?method_name }
            OPTIONAL { ?attr definedIn ?class . ?attr name ?attr_name }
            OPTIONAL { ?class inheritsFrom ?parent . ?parent name ?parent_name }
        }
    }
    | when { {classes} is not null } filter { class_name in {classes} }
    | render {
        diagram: mermaid(
            type: class,
            classes: class_name,
            methods: method_name,
            attributes: attr_name,
            inheritance: parent_name -> class_name
        )
    }
    | emit { diagram }
}
```

### 3.2 Sequence Diagram (Native)

```cadsl
diagram sequence_diagram() {
    param classes: list = [];
    param max_depth: int = 10;

    reql {
        SELECT ?caller_class ?callee_class ?method_name
        WHERE {
            ?caller type {Method} .
            ?caller definedIn ?cc . ?cc name ?caller_class .
            ?caller calls ?callee .
            ?callee name ?method_name .
            ?callee definedIn ?tc . ?tc name ?callee_class
        }
    }
    | when { {classes} is not null } filter { caller_class in {classes} or callee_class in {classes} }
    | limit { {max_depth} * 10 }
    | render {
        diagram: mermaid(
            type: sequence,
            participants: [caller_class, callee_class],
            messages: caller_class ->> callee_class : method_name
        )
    }
    | emit { diagram }
}
```

### 3.3 Multi-Format Table Report

```cadsl
detector code_quality_report(category="reporting", severity="info") {
    param format: str = "markdown";

    reql {
        SELECT ?class ?name ?method_count ?complexity ?file
        WHERE {
            ?class type {Class} . ?class name ?name .
            ?class inFile ?file .
            ?m type {Method} . ?m definedIn ?class
        }
        GROUP BY ?class ?name ?file
    }
    | compute {
        method_count: count(methods),
        status: method_count > 20 ? "Large" : method_count > 10 ? "Medium" : "Small"
    }
    | render {
        table: markdown(
            columns: [
                name as "Class Name",
                method_count as "Methods" : { align: right },
                status as "Size",
                file as "Location"
            ],
            title: "Code Quality Summary",
            totals: true,
            summary: { total_classes: count, avg_methods: avg(method_count) }
        )
    }
    | emit { report }
}
```

### 3.4 Dashboard with Multiple Outputs

```cadsl
diagram code_dashboard() {
    param format: str = "markdown";

    merge {
        reql { SELECT (COUNT(?c) AS ?class_count) WHERE { ?c type {Class} } },
        reql { SELECT (COUNT(?m) AS ?method_count) WHERE { ?m type {Method} } },
        reql { SELECT (COUNT(?f) AS ?function_count) WHERE { ?f type {Function} } }
    }
    | render {
        report: markdown(
            title: "Codebase Dashboard",
            sections: [
                { type: metrics, data: counts },
                { type: chart, title: "Entity Distribution",
                  data: counts, options: { type: pie } },
                { type: table, title: "Details",
                  data: breakdown, options: { columns: [name, count] } }
            ]
        )
    }
    | emit { dashboard, metrics: counts, chart: distribution }
}
```

### 3.5 Coupling Heatmap

```cadsl
diagram coupling_heatmap() {
    reql {
        SELECT ?from_class ?to_class
        WHERE {
            ?m type {Method} . ?m definedIn ?fc . ?fc name ?from_class .
            ?m calls ?c . ?c definedIn ?tc . ?tc name ?to_class
        }
    }
    | filter { from_class != to_class }
    | compute { coupling: 1 }
    | pivot { rows: from_class, cols: to_class, value: coupling, aggregate: sum }
    | render {
        chart: mermaid(
            type: heatmap,
            x: from_class,
            y: to_class,
            value: coupling,
            title: "Class Coupling Matrix"
        )
    }
    | emit { heatmap, matrix: pivot_data }
}
```

---

## 4. Implementation Strategy

### Phase 1: Enhanced Mermaid (Week 1)
- Extend `render_mermaid` to support all diagram types
- Add `class` diagram with member aggregation
- Add `sequence` diagram with message syntax
- Add `state`, `er`, `pie`, `mindmap` types

### Phase 2: Table Rendering (Week 2)
- Implement `render_table` step
- Support markdown, HTML, CSV, ASCII formats
- Add column formatting options
- Add totals and summary rows

### Phase 3: Chart Support (Week 3)
- Implement `render_chart` step
- Mermaid pie charts
- ASCII bar/line charts
- Optional Vega-Lite output

### Phase 4: Report Composition (Week 4)
- Implement `render_report` step
- Multi-section composition
- Template support
- Multi-output `emit`

---

## 5. Grammar Changes Summary

```lark
// New/enhanced steps
step: ...existing...
    | render_diagram_step
    | render_table_step
    | render_chart_step
    | render_report_step

// Enhanced emit
emit_step: "emit" "{" emit_spec "}"
emit_spec: emit_field ("," emit_field)*
emit_field: NAME (":" (NAME | render_inline))?

// Diagram formats
render_diagram_step: "render_diagram" "{" diagram_spec "}"
diagram_spec: diagram_format "(" diagram_params ")"
diagram_format: "mermaid" | "plantuml" | "graphviz" | "d2" | "ascii"

// Table rendering
render_table_step: "render_table" "{" table_spec "}"
table_spec: table_format "(" table_params ")"
table_format: "markdown" | "html" | "csv" | "ascii" | "json"

// Chart rendering
render_chart_step: "render_chart" "{" chart_spec "}"
chart_spec: chart_format "(" chart_params ")"
chart_format: "mermaid" | "ascii" | "svg" | "vega"

// Report composition
render_report_step: "render_report" "{" report_spec "}"
report_spec: report_format "(" report_params ")"
```

---

## 6. Aggregation Primitives

Key insight: Many diagram tools need aggregation before rendering.

### 6.1 Collect Step

```lark
// Collect fields into groups
collect_step: "collect" "{" collect_spec "}"

collect_spec: "by" ":" NAME "," "fields" ":" collect_fields

collect_fields: "{" collect_field ("," collect_field)* "}"
collect_field: NAME ":" collect_op
collect_op: "list" | "set" | "first" | "last" | "count"
```

**Example:**
```cadsl
# Aggregate methods/attributes per class
| collect {
    by: class_name,
    fields: {
        methods: set(method_name),
        attributes: set(attr_name),
        parent: first(parent_name)
    }
}
| render_diagram { mermaid(type: class, ...) }
```

### 6.2 Nest Step

```lark
// Create nested structure
nest_step: "nest" "{" nest_spec "}"

nest_spec: "key" ":" NAME "," "children" ":" NAME ("," "depth" ":" INT)?
```

**Example:**
```cadsl
# Build call hierarchy
| nest { key: caller, children: callee, depth: 5 }
| render_diagram { mermaid(type: flowchart, ...) }
```

---

## 7. Step Implementations

### 7.1 RenderDiagramStep

```python
class RenderDiagramStep(PipelineStep):
    """Render data as diagram."""

    def __init__(self, format: str, params: dict):
        self.format = format  # mermaid, plantuml, graphviz, d2, ascii
        self.params = params

    def process(self, data: List[Dict], ctx: Context) -> Dict:
        diagram_type = self.params.get('type', 'flowchart')

        if self.format == 'mermaid':
            return self._render_mermaid(data, diagram_type)
        elif self.format == 'plantuml':
            return self._render_plantuml(data, diagram_type)
        # ... etc

    def _render_mermaid(self, data, diagram_type):
        if diagram_type == 'class':
            return self._mermaid_class_diagram(data)
        elif diagram_type == 'sequence':
            return self._mermaid_sequence_diagram(data)
        # ... etc
```

### 7.2 RenderTableStep

```python
class RenderTableStep(PipelineStep):
    """Render data as formatted table."""

    def __init__(self, format: str, params: dict):
        self.format = format  # markdown, html, csv, ascii
        self.params = params

    def process(self, data: List[Dict], ctx: Context) -> Dict:
        columns = self.params.get('columns', list(data[0].keys()) if data else [])
        title = self.params.get('title', '')

        if self.format == 'markdown':
            return self._render_markdown_table(data, columns, title)
        elif self.format == 'html':
            return self._render_html_table(data, columns, title)
        # ... etc
```

### 7.3 CollectStep

```python
class CollectStep(PipelineStep):
    """Aggregate rows by key, collecting fields."""

    def __init__(self, by: str, fields: Dict[str, str]):
        self.by = by
        self.fields = fields  # {output_name: (source_field, op)}

    def process(self, data: List[Dict], ctx: Context) -> List[Dict]:
        groups = {}
        for row in data:
            key = row.get(self.by)
            if key not in groups:
                groups[key] = {self.by: key}
                for name, (field, op) in self.fields.items():
                    groups[key][name] = [] if op in ('list', 'set') else None

            for name, (field, op) in self.fields.items():
                value = row.get(field)
                if op == 'list':
                    groups[key][name].append(value)
                elif op == 'set':
                    if value not in groups[key][name]:
                        groups[key][name].append(value)
                elif op == 'first' and groups[key][name] is None:
                    groups[key][name] = value
                # ... etc

        return list(groups.values())
```

---

## 8. Benefits

| Aspect | Current | Proposed |
|--------|---------|----------|
| Diagram tools with Python | 5+ | 0 |
| Output formats | Mermaid only | 5+ (Mermaid, PlantUML, Graphviz, D2, ASCII) |
| Table rendering | None | Markdown, HTML, CSV, ASCII |
| Chart support | None | Pie, Bar, Line (via Mermaid/ASCII) |
| Multi-output | No | Yes (emit multiple named outputs) |
| Report composition | No | Yes (sections, templates) |
| Declarative coverage | 95% | 99%+ |

---

## 9. Migration Path

1. **Existing tools continue to work** - `render_mermaid` remains supported
2. **Gradual adoption** - New syntax coexists with old
3. **Tool updates** - Convert Python-heavy diagram tools one by one
4. **Deprecation** - Mark old syntax as deprecated after migration

---

## 10. Priority Implementation Order

### Must Have (P0)
1. `collect` step (aggregation primitive)
2. Enhanced `render_mermaid` (class, sequence)
3. `render_table` (markdown format)
4. Multi-output `emit`

### Should Have (P1)
5. `render_table` (HTML, CSV, ASCII)
6. `render_chart` (pie, bar via Mermaid)
7. `nest` step

### Nice to Have (P2)
8. PlantUML output
9. Graphviz output
10. `render_report` composition
11. Template support
