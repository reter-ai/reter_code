"""
CADSL Render Steps.

Contains step classes for rendering data as tables and charts:
- RenderTableStep: Render data as formatted table (markdown, html, csv, ascii, json)
- RenderChartStep: Render data as chart (mermaid pie/bar/line, ascii)
"""

from typing import Any, Dict, List, Optional


class RenderTableStep:
    """
    Render data as formatted table.

    Syntax: render_table { format: markdown, columns: [name, count], title: "Summary" }

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, format="markdown", columns=None, title=None, totals=False,
                 sort=None, group_by=None, max_rows=None):
        self.format = format
        self.columns = columns or []
        self.title = title
        self.totals = totals
        self.sort = sort
        self.group_by = group_by
        self.max_rows = max_rows

    def execute(self, data, ctx=None):
        """Render as table."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok({"table": "", "format": self.format, "row_count": 0})

            # Determine columns
            if self.columns:
                cols = self.columns
            else:
                # Auto-detect from first row
                cols = [{"name": k, "alias": k} for k in data[0].keys()]

            # Sort if specified
            if self.sort:
                reverse = self.sort.startswith('-')
                sort_key = self.sort.lstrip('-+')
                data = sorted(data, key=lambda r: r.get(sort_key, ''), reverse=reverse)

            # Limit rows
            if self.max_rows:
                data = data[:self.max_rows]

            # Render based on format
            if self.format == "markdown":
                table = self._render_markdown(data, cols)
            elif self.format == "html":
                table = self._render_html(data, cols)
            elif self.format == "csv":
                table = self._render_csv(data, cols)
            elif self.format == "ascii":
                table = self._render_ascii(data, cols)
            elif self.format == "json":
                import json
                table = json.dumps(data, indent=2, default=str)
            else:
                table = self._render_markdown(data, cols)

            return pipeline_ok({"table": table, "format": self.format, "row_count": len(data)})
        except Exception as e:
            return pipeline_err("render_table", f"Table rendering failed: {e}", e)

    def _render_markdown(self, data, cols):
        """Render as Markdown table."""
        lines = []
        if self.title:
            lines.append(f"## {self.title}\n")

        # Header
        headers = [c.get('alias', c.get('name', '')) for c in cols]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

        # Rows
        for row in data:
            cells = [str(row.get(c.get('name', ''), '')) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")

        # Totals row
        if self.totals:
            totals = []
            for c in cols:
                name = c.get('name', '')
                values = [r.get(name) for r in data if isinstance(r.get(name), (int, float))]
                if values:
                    totals.append(str(sum(values)))
                else:
                    totals.append('')
            lines.append("| " + " | ".join(totals) + " |")

        return "\n".join(lines)

    def _render_html(self, data, cols):
        """Render as HTML table."""
        lines = ['<table>']
        if self.title:
            lines.append(f'<caption>{self.title}</caption>')

        # Header
        lines.append('<thead><tr>')
        for c in cols:
            lines.append(f'<th>{c.get("alias", c.get("name", ""))}</th>')
        lines.append('</tr></thead>')

        # Body
        lines.append('<tbody>')
        for row in data:
            lines.append('<tr>')
            for c in cols:
                lines.append(f'<td>{row.get(c.get("name", ""), "")}</td>')
            lines.append('</tr>')
        lines.append('</tbody>')

        lines.append('</table>')
        return "\n".join(lines)

    def _render_csv(self, data, cols):
        """Render as CSV."""
        lines = []
        headers = [c.get('alias', c.get('name', '')) for c in cols]
        lines.append(",".join(f'"{h}"' for h in headers))

        for row in data:
            cells = [str(row.get(c.get('name', ''), '')).replace('"', '""') for c in cols]
            lines.append(",".join(f'"{cell}"' for cell in cells))

        return "\n".join(lines)

    def _render_ascii(self, data, cols):
        """Render as ASCII table."""
        headers = [c.get('alias', c.get('name', '')) for c in cols]

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in data:
            for i, c in enumerate(cols):
                val = str(row.get(c.get('name', ''), ''))
                widths[i] = max(widths[i], len(val))

        # Build table
        lines = []
        sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

        if self.title:
            lines.append(self.title)
            lines.append('=' * len(sep))

        lines.append(sep)
        lines.append('|' + '|'.join(f' {h:<{w}} ' for h, w in zip(headers, widths)) + '|')
        lines.append(sep)

        for row in data:
            cells = [str(row.get(c.get('name', ''), '')) for c in cols]
            lines.append('|' + '|'.join(f' {c:<{w}} ' for c, w in zip(cells, widths)) + '|')

        lines.append(sep)
        return "\n".join(lines)


class RenderChartStep:
    """
    Render data as chart.

    Syntax: render_chart { type: bar, x: category, y: count, format: mermaid }

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, chart_type="bar", x=None, y=None, series=None, title=None,
                 format="mermaid", colors=None, stacked=False, horizontal=False):
        self.chart_type = chart_type
        self.x = x
        self.y = y
        self.series = series
        self.title = title
        self.format = format
        self.colors = colors
        self.stacked = stacked
        self.horizontal = horizontal

    def execute(self, data, ctx=None):
        """Render as chart."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok({"chart": "", "format": self.format, "type": self.chart_type})

            if self.format == "mermaid":
                chart = self._render_mermaid(data)
            elif self.format == "ascii":
                chart = self._render_ascii(data)
            else:
                chart = self._render_mermaid(data)

            return pipeline_ok({"chart": chart, "format": self.format, "type": self.chart_type})
        except Exception as e:
            return pipeline_err("render_chart", f"Chart rendering failed: {e}", e)

    def _render_mermaid(self, data):
        """Render as Mermaid chart."""
        if self.chart_type == "pie":
            return self._mermaid_pie(data)
        elif self.chart_type in ("bar", "line"):
            return self._mermaid_xychart(data)
        else:
            return self._mermaid_pie(data)

    def _mermaid_pie(self, data):
        """Render Mermaid pie chart."""
        lines = ["pie showData"]
        if self.title:
            lines[0] = f'pie showData title {self.title}'

        for row in data:
            label = row.get(self.x, "Unknown")
            value = row.get(self.y, 0)
            if value:
                lines.append(f'    "{label}" : {value}')

        return "\n".join(lines)

    def _mermaid_xychart(self, data):
        """Render Mermaid xychart (bar/line)."""
        lines = ["xychart-beta"]
        if self.horizontal:
            lines[0] += " horizontal"
        if self.title:
            lines.append(f'    title "{self.title}"')

        # Extract x-axis categories
        categories = [str(row.get(self.x, '')) for row in data]
        lines.append(f'    x-axis [{", ".join(f"{c}" for c in categories)}]')

        # Extract y values
        values = [row.get(self.y, 0) for row in data]
        max_val = max(values) if values else 100
        lines.append(f'    y-axis "Count" 0 --> {max_val}')

        chart_type = "bar" if self.chart_type == "bar" else "line"
        lines.append(f'    {chart_type} [{", ".join(str(v) for v in values)}]')

        return "\n".join(lines)

    def _render_ascii(self, data):
        """Render as ASCII chart."""
        if self.chart_type == "pie":
            return self._ascii_pie(data)
        else:
            return self._ascii_bar(data)

    def _ascii_pie(self, data):
        """Simple ASCII representation of pie data."""
        lines = []
        if self.title:
            lines.append(f"  {self.title}")
            lines.append("  " + "=" * len(self.title))

        total = sum(row.get(self.y, 0) for row in data)
        for row in data:
            label = row.get(self.x, "Unknown")
            value = row.get(self.y, 0)
            pct = (value / total * 100) if total else 0
            bar = "#" * int(pct / 2)
            lines.append(f"  {label:<20} {bar} {pct:.1f}%")

        return "\n".join(lines)

    def _ascii_bar(self, data):
        """Simple ASCII bar chart."""
        lines = []
        if self.title:
            lines.append(f"  {self.title}")
            lines.append("  " + "=" * len(self.title))

        max_val = max(row.get(self.y, 0) for row in data) if data else 1
        for row in data:
            label = row.get(self.x, "Unknown")[:15]
            value = row.get(self.y, 0)
            bar_len = int(value / max_val * 40) if max_val else 0
            bar = "#" * bar_len
            lines.append(f"  {label:<15} |{bar} {value}")

        return "\n".join(lines)
