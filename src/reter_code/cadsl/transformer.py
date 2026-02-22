"""
CADSL Transformer - AST to Pipeline Transformer.

This module transforms CADSL Lark parse trees into executable Pipeline objects
that can be run against a RETER instance.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import textwrap

from lark import Tree, Token

from .compiler import (
    compile_condition,
    compile_expression,
    compile_object_expr,
    ExpressionCompiler,
    ConditionCompiler,
    ObjectExprCompiler,
    unquote,
)
from .utils import get_tool_name, get_tool_type

# Import step classes from steps package (re-export for backward compatibility)
from .steps import (
    WhenStep, UnlessStep, BranchStep, CatchStep,
    GraphCyclesStep, GraphClosureStep, GraphTraverseStep, ParallelStep,
    CollectStep, NestStep,
)


# ============================================================
# TOOL SPECIFICATION
# ============================================================

@dataclass
class ParamSpec:
    """Specification for a tool parameter.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    name: str
    type: str
    required: bool = False
    default: Any = None
    choices: Optional[List[Any]] = None


@dataclass
class ToolSpec:
    """
    Specification for a CADSL tool.

    Contains all the information needed to create and execute a pipeline.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    name: str
    tool_type: str  # "query", "detector", "diagram"
    description: str = ""
    params: List[ParamSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Pipeline components
    source_type: str = ""  # "reql", "rag_search", "rag_duplicates", "rag_clusters", "rag_dbscan", "value", "merge"
    source_content: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    emit_key: Optional[str] = None

    # For RAG sources
    rag_params: Dict[str, Any] = field(default_factory=dict)

    # For merge sources - list of sub-sources
    merge_sources: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# TRANSFORMER
# ============================================================

class CADSLTransformer:
    """
    Transforms CADSL parse trees into ToolSpec objects.

    The ToolSpec can then be converted to executable Pipeline objects.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a abstract-syntax-tree-transformer.
    ::: This depends-on `reter_code.cadsl.ExpressionCompiler`.
    ::: This is-part-of `reter_code.cadsl`.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self):
        self.expr_compiler = ExpressionCompiler()
        self.cond_compiler = ConditionCompiler()
        self.obj_compiler = ObjectExprCompiler()

    def transform(self, tree: Tree) -> List[ToolSpec]:
        """
        Transform a parse tree into a list of ToolSpec objects.

        Args:
            tree: Lark parse tree from CADSLParser

        Returns:
            List of ToolSpec objects
        """
        tools = []

        for child in tree.children:
            if isinstance(child, Tree) and child.data == "tool_def":
                spec = self._transform_tool_def(child)
                if spec:
                    tools.append(spec)

        return tools

    def _transform_tool_def(self, node: Tree) -> Optional[ToolSpec]:
        """Transform a tool_def node into a ToolSpec."""
        # Extract tool type
        tool_type = get_tool_type(node)
        tool_name = get_tool_name(node)

        spec = ToolSpec(
            name=tool_name,
            tool_type=tool_type,
        )

        # Extract metadata
        metadata_node = self._find_child(node, "metadata")
        if metadata_node:
            spec.metadata = self._transform_metadata(metadata_node)

        # Extract tool body
        tool_body = self._find_child(node, "tool_body")
        if tool_body:
            # Extract docstring
            docstring_node = self._find_child(tool_body, "docstring")
            if docstring_node:
                spec.description = self._extract_docstring(docstring_node)

            # Extract parameters
            for child in tool_body.children:
                if isinstance(child, Tree) and child.data == "param_def":
                    param = self._transform_param_def(child)
                    if param:
                        spec.params.append(param)

            # Extract pipeline
            pipeline = self._find_child(tool_body, "pipeline")
            if pipeline:
                self._transform_pipeline(pipeline, spec)

        return spec

    def _find_child(self, node: Tree, data: str) -> Optional[Tree]:
        """Find first child with matching data."""
        for child in node.children:
            if isinstance(child, Tree) and child.data == data:
                return child
        return None

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    def _transform_metadata(self, node: Tree) -> Dict[str, Any]:
        """Transform metadata node to dict."""
        metadata = {}

        for child in node.children:
            if isinstance(child, Tree) and child.data == "meta_item":
                key = None
                value = None

                for item in child.children:
                    if isinstance(item, Token) and item.type == "NAME":
                        key = str(item)
                    elif isinstance(item, Tree) and item.data == "meta_value":
                        value = self._extract_meta_value(item)

                if key:
                    metadata[key] = value

        return metadata

    def _extract_meta_value(self, node: Tree) -> Any:
        """Extract value from meta_value node."""
        for child in node.children:
            if isinstance(child, Token):
                if child.type == "STRING":
                    return unquote(str(child))
                elif child.type == "NAME":
                    return str(child)
            elif isinstance(child, Tree):
                if child.data == "capability_array":
                    return self._extract_capability_list(child)
        return None

    def _extract_capability_list(self, node: Tree) -> List[str]:
        """Extract capability list."""
        caps = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "capability_list":
                for item in child.children:
                    if isinstance(item, Token) and item.type == "STRING":
                        caps.append(unquote(str(item)))
        return caps

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    def _transform_param_def(self, node: Tree) -> Optional[ParamSpec]:
        """Transform param_def node to ParamSpec."""
        name = None
        param_type = "str"
        default = None
        required = False
        choices = None

        for child in node.children:
            if isinstance(child, Token) and child.type == "NAME":
                name = str(child)
            elif isinstance(child, Tree):
                if child.data.startswith("type_"):
                    param_type = self._extract_type(child)
                elif child.data == "param_modifiers":
                    for mod in child.children:
                        if isinstance(mod, Tree):
                            if mod.data == "param_default":
                                default = self._extract_value(mod)
                            elif mod.data == "param_required":
                                required = True
                            elif mod.data == "param_choices":
                                choices = self._extract_choices(mod)

        if name:
            return ParamSpec(
                name=name,
                type=param_type,
                required=required,
                default=default,
                choices=choices,
            )
        return None

    def _extract_type(self, node: Tree) -> str:
        """Extract type name from type_spec node."""
        type_map = {
            "type_int": "int",
            "type_str": "str",
            "type_float": "float",
            "type_bool": "bool",
            "type_list": "list",
        }
        if node.data in type_map:
            return type_map[node.data]
        if node.data == "type_list_of":
            inner = None
            for child in node.children:
                if isinstance(child, Tree) and child.data.startswith("type_"):
                    inner = self._extract_type(child)
            return f"list<{inner}>" if inner else "list"
        return "str"

    def _extract_choices(self, node: Tree) -> List[Any]:
        """Extract choices from param_choices node."""
        choices = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "value_list":
                for item in child.children:
                    val = self._extract_value(item)
                    if val is not None:
                        choices.append(val)
        return choices

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    def _transform_pipeline(self, node: Tree, spec: ToolSpec) -> None:
        """Transform pipeline node and update spec."""
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "source":
                    # Source wrapper
                    for source_child in child.children:
                        if isinstance(source_child, Tree):
                            self._transform_source(source_child, spec)

                elif child.data in ("reql_source", "rag_source", "value_source"):
                    self._transform_source(child, spec)

                elif child.data == "step":
                    # Step wrapper
                    for step_child in child.children:
                        if isinstance(step_child, Tree):
                            step = self._transform_step(step_child)
                            if step:
                                spec.steps.append(step)

                elif child.data.endswith("_step"):
                    step = self._transform_step(child)
                    if step:
                        spec.steps.append(step)

    def _transform_source(self, node: Tree, spec: ToolSpec) -> None:
        """Transform source node and update spec."""
        if node.data == "reql_source":
            spec.source_type = "reql"
            # Extract REQL content from REQL_BLOCK token
            for child in node.children:
                if isinstance(child, Token) and child.type == "REQL_BLOCK":
                    content = str(child)
                    # Remove outer braces
                    if content.startswith("{") and content.endswith("}"):
                        content = content[1:-1].strip()
                    spec.source_content = content

        elif node.data == "rag_source":
            # Parse RAG operation type and parameters
            for child in node.children:
                if isinstance(child, Tree) and child.data == "rag_args":
                    self._transform_rag_args(child, spec)

        elif node.data == "value_source":
            spec.source_type = "value"
            # Extract value expression
            for child in node.children:
                if isinstance(child, Tree):
                    spec.source_content = self._extract_expr_as_string(child)

        elif node.data == "merge_source":
            spec.source_type = "merge"
            # Extract list of sub-sources
            for child in node.children:
                if isinstance(child, Tree) and child.data == "source_list":
                    spec.merge_sources = self._extract_source_list(child)

        elif node.data == "file_scan_source":
            spec.source_type = "file_scan"
            # Extract file_scan parameters
            for child in node.children:
                if isinstance(child, Tree) and child.data == "file_scan_spec":
                    spec.rag_params = self._transform_file_scan_spec(child)

        elif node.data == "parse_file_source":
            spec.source_type = "parse_file"
            for child in node.children:
                if isinstance(child, Tree) and child.data == "parse_file_spec":
                    spec.rag_params = self._transform_parse_file_spec(child)

    def _transform_file_scan_spec(self, node: Tree) -> Dict[str, Any]:
        """Transform file_scan specification into parameters dict."""
        params = {}

        for child in node.children:
            if isinstance(child, Tree):
                data = child.data

                if data == "fs_glob":
                    params["glob"] = self._extract_fs_string(child)
                elif data == "fs_glob_param":
                    params["glob"] = self._extract_fs_param_ref(child)
                elif data == "fs_exclude":
                    params["exclude"] = self._extract_fs_string_list(child)
                elif data == "fs_contains":
                    params["contains"] = self._extract_fs_string(child)
                elif data == "fs_contains_param":
                    params["contains"] = self._extract_fs_param_ref(child)
                elif data == "fs_not_contains":
                    params["not_contains"] = self._extract_fs_string(child)
                elif data == "fs_not_contains_param":
                    params["not_contains"] = self._extract_fs_param_ref(child)
                elif data == "fs_case_sensitive":
                    params["case_sensitive"] = self._extract_fs_bool(child)
                elif data == "fs_include_matches":
                    params["include_matches"] = self._extract_fs_bool(child)
                elif data == "fs_context_lines":
                    params["context_lines"] = self._extract_fs_int(child)
                elif data == "fs_context_lines_param":
                    params["context_lines"] = self._extract_fs_param_ref(child)
                elif data == "fs_max_matches":
                    params["max_matches_per_file"] = self._extract_fs_int(child)
                elif data == "fs_max_matches_param":
                    params["max_matches_per_file"] = self._extract_fs_param_ref(child)
                elif data == "fs_include_stats":
                    params["include_stats"] = self._extract_fs_bool(child)

        return params

    def _extract_fs_string(self, node: Tree) -> str:
        """Extract a string value from a file_scan param node."""
        for child in node.children:
            if isinstance(child, Token) and child.type == "STRING":
                s = str(child)
                return s[1:-1] if s.startswith('"') or s.startswith("'") else s
        return ""

    def _extract_fs_int(self, node: Tree) -> int:
        """Extract an integer value from a file_scan param node."""
        for child in node.children:
            if isinstance(child, Token) and child.type == "INT":
                return int(str(child))
        return 0

    def _extract_fs_bool(self, node: Tree) -> bool:
        """Extract a boolean value from a file_scan param node."""
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "bool_true":
                    return True
                elif child.data == "bool_false":
                    return False
        return False

    def _extract_fs_param_ref(self, node: Tree) -> str:
        """Extract a parameter reference from a file_scan param node."""
        for child in node.children:
            if isinstance(child, Tree) and child.data == "param_ref":
                for item in child.children:
                    if isinstance(item, Token) and item.type == "NAME":
                        return "{" + str(item) + "}"
        return ""

    def _extract_fs_string_list(self, node: Tree) -> List[str]:
        """Extract list of strings from fs_exclude node."""
        strings = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "fs_string_list":
                for item in child.children:
                    if isinstance(item, Token) and item.type == "STRING":
                        s = str(item)
                        strings.append(s[1:-1] if s.startswith('"') or s.startswith("'") else s)
            elif isinstance(child, Token) and child.type == "STRING":
                s = str(child)
                strings.append(s[1:-1] if s.startswith('"') or s.startswith("'") else s)
        return strings

    def _transform_parse_file_spec(self, node: Tree) -> Dict[str, Any]:
        """Transform parse_file specification into parameters dict."""
        params = {}

        for child in node.children:
            if isinstance(child, Tree):
                data = child.data

                if data == "pf_path":
                    params["path"] = self._extract_fs_string(child)
                elif data == "pf_path_param":
                    params["path"] = self._extract_fs_param_ref(child)
                elif data == "pf_format":
                    # Extract format from pf_format_type subtree
                    for fmt_child in child.children:
                        if isinstance(fmt_child, Tree):
                            params["format"] = fmt_child.data.replace("pf_", "")
                elif data == "pf_format_param":
                    params["format"] = self._extract_fs_param_ref(child)
                elif data == "pf_encoding":
                    params["encoding"] = self._extract_fs_string(child)
                elif data == "pf_encoding_param":
                    params["encoding"] = self._extract_fs_param_ref(child)
                elif data == "pf_separator":
                    params["separator"] = self._extract_fs_string(child)
                elif data == "pf_separator_param":
                    params["separator"] = self._extract_fs_param_ref(child)
                elif data == "pf_sheet":
                    params["sheet"] = self._extract_fs_string(child)
                elif data == "pf_sheet_param":
                    params["sheet"] = self._extract_fs_param_ref(child)
                elif data == "pf_columns":
                    params["columns"] = self._extract_pf_string_list(child)
                elif data == "pf_limit":
                    params["limit"] = self._extract_fs_int(child)
                elif data == "pf_limit_param":
                    params["limit"] = self._extract_fs_param_ref(child)

        return params

    def _extract_pf_string_list(self, node: Tree) -> List[str]:
        """Extract list of strings from pf_columns node."""
        strings = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "pf_string_list":
                for item in child.children:
                    if isinstance(item, Token) and item.type == "STRING":
                        s = str(item)
                        strings.append(s[1:-1] if s.startswith('"') or s.startswith("'") else s)
            elif isinstance(child, Token) and child.type == "STRING":
                s = str(child)
                strings.append(s[1:-1] if s.startswith('"') or s.startswith("'") else s)
        return strings

    def _extract_source_list(self, node: Tree) -> List[Dict[str, Any]]:
        """Extract list of sources from source_list node.

        Each source_item contains:
        - A source (reql_source, rag_source, or value_source)
        - Optional source_steps (pipeline steps to apply)
        """
        sources = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "source_item":
                source_spec = None
                steps = []

                for source_node in child.children:
                    if isinstance(source_node, Tree):
                        if source_node.data in ("reql_source", "rag_source", "value_source"):
                            # Extract the source
                            source_spec = self._extract_single_source(source_node)
                        elif source_node.data == "source_steps":
                            # Extract pipeline steps
                            for step_node in source_node.children:
                                if isinstance(step_node, Tree):
                                    step = self._transform_step(step_node)
                                    if step:
                                        steps.append(step)

                if source_spec:
                    source_spec["steps"] = steps
                    sources.append(source_spec)
        return sources

    def _extract_single_source(self, node: Tree) -> Optional[Dict[str, Any]]:
        """Extract a single source specification."""
        if node.data == "reql_source":
            for child in node.children:
                if isinstance(child, Token) and child.type == "REQL_BLOCK":
                    content = str(child)
                    if content.startswith("{") and content.endswith("}"):
                        content = content[1:-1].strip()
                    return {"type": "reql", "content": content}

        elif node.data == "rag_source":
            # Create a temporary spec to extract RAG params
            temp_spec = ToolSpec(name="", tool_type="")
            for child in node.children:
                if isinstance(child, Tree) and child.data == "rag_args":
                    self._transform_rag_args(child, temp_spec)
            return {
                "type": temp_spec.source_type,
                "params": temp_spec.rag_params
            }

        elif node.data == "value_source":
            for child in node.children:
                if isinstance(child, Tree):
                    content = self._extract_expr_as_string(child)
                    return {"type": "value", "content": content}

        return None

    def _transform_rag_args(self, node: Tree, spec: ToolSpec) -> None:
        """Transform RAG arguments (search, duplicates, or clusters)."""
        # Find the operation type (rag_search, rag_duplicates, rag_clusters)
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "rag_search":
                    spec.source_type = "rag_search"
                    spec.rag_params = self._extract_rag_params(child)
                elif child.data == "rag_duplicates":
                    spec.source_type = "rag_duplicates"
                    spec.rag_params = self._extract_rag_params(child)
                elif child.data == "rag_clusters":
                    spec.source_type = "rag_clusters"
                    spec.rag_params = self._extract_rag_params(child)
                elif child.data == "rag_dbscan":
                    spec.source_type = "rag_dbscan"
                    spec.rag_params = self._extract_rag_params(child)

    def _extract_rag_params(self, node: Tree) -> Dict[str, Any]:
        """Extract parameters from RAG operation node."""
        params = {}
        for child in node.children:
            if isinstance(child, Tree) and child.data == "rag_param":
                name = None
                value = None
                for param_child in child.children:
                    if isinstance(param_child, Token) and param_child.type == "NAME":
                        name = str(param_child)
                    elif isinstance(param_child, Tree):
                        value = self._extract_rag_param_value(param_child)
                if name is not None:
                    params[name] = value
        return params

    def _extract_rag_param_value(self, node: Tree) -> Any:
        """Extract value from rag_param_value node."""
        if node.data == "rag_float_val":
            return float(str(node.children[0]))
        elif node.data == "rag_int_val":
            return int(str(node.children[0]))
        elif node.data == "rag_true_val":
            return True
        elif node.data == "rag_false_val":
            return False
        elif node.data == "rag_string_val":
            s = str(node.children[0])
            return s[1:-1] if s.startswith('"') or s.startswith("'") else s
        elif node.data == "rag_param_ref":
            # Return as placeholder string to be resolved at runtime
            param_node = node.children[0]
            if isinstance(param_node, Tree) and param_node.data == "param_ref":
                return "{" + str(param_node.children[0]) + "}"
            return None
        elif node.data == "rag_list_val":
            # Extract list values
            items = []
            for child in node.children:
                if isinstance(child, Tree) and child.data == "val_list":
                    for item in child.children:
                        if isinstance(item, Tree) and item.data == "value_list":
                            for val in item.children:
                                items.append(self._extract_value(val))
            return items
        return None

    def _extract_value(self, node: Tree) -> Any:
        """Extract a simple value from a value node."""
        if node.data == "val_string":
            s = str(node.children[0])
            return s[1:-1] if s.startswith('"') or s.startswith("'") else s
        elif node.data == "val_int":
            return int(str(node.children[0]))
        elif node.data == "val_float":
            return float(str(node.children[0]))
        elif node.data == "val_true":
            return True
        elif node.data == "val_false":
            return False
        elif node.data == "val_null":
            return None
        return str(node.children[0]) if node.children else None

    def _transform_step(self, node: Tree) -> Optional[Dict[str, Any]]:
        """Transform a step node to a step specification dict."""
        step_type = node.data.replace("_step", "")

        if step_type == "filter":
            return self._transform_filter_step(node)
        elif step_type == "select":
            return self._transform_select_step(node)
        elif step_type == "map":
            return self._transform_map_step(node)
        elif step_type == "flat_map":
            return self._transform_flat_map_step(node)
        elif step_type == "order_by":
            return self._transform_order_by_step(node)
        elif step_type == "limit":
            return self._transform_limit_step(node)
        elif step_type == "offset":
            return self._transform_offset_step(node)
        elif step_type == "group_by":
            return self._transform_group_by_step(node)
        elif step_type == "aggregate":
            return self._transform_aggregate_step(node)
        elif step_type == "unique":
            return self._transform_unique_step(node)
        elif step_type == "flatten":
            return {"type": "flatten"}
        elif step_type == "tap":
            return self._transform_tap_step(node)
        elif step_type == "python":
            return self._transform_python_step(node)
        elif step_type == "render":
            return self._transform_render_step(node)
        elif step_type == "emit":
            return self._transform_emit_step(node)
        elif step_type == "when":
            return self._transform_when_step(node)
        elif step_type == "unless":
            return self._transform_unless_step(node)
        elif step_type == "branch":
            return self._transform_branch_step(node)
        elif step_type == "catch":
            return self._transform_catch_step(node)
        elif step_type == "parallel":
            return self._transform_parallel_step(node)
        elif step_type == "join":
            return self._transform_join_step(node)
        elif step_type == "graph_cycles":
            return self._transform_graph_cycles_step(node)
        elif step_type == "graph_closure":
            return self._transform_graph_closure_step(node)
        elif step_type == "graph_traverse":
            return self._transform_graph_traverse_step(node)
        elif step_type == "render_mermaid":
            return self._transform_render_mermaid_step(node)
        elif step_type == "pivot":
            return self._transform_pivot_step(node)
        elif step_type == "compute":
            return self._transform_compute_step(node)
        elif step_type == "collect":
            return self._transform_collect_step(node)
        elif step_type == "nest":
            return self._transform_nest_step(node)
        elif step_type == "render_table":
            return self._transform_render_table_step(node)
        elif step_type == "render_chart":
            return self._transform_render_chart_step(node)
        elif step_type == "cross_join":
            return self._transform_cross_join_step(node)
        elif step_type == "set_similarity":
            return self._transform_set_similarity_step(node)
        elif step_type == "string_match":
            return self._transform_string_match_step(node)
        elif step_type == "rag_enrich":
            return self._transform_rag_enrich_step(node)
        elif step_type == "create_task":
            return self._transform_create_task_step(node)
        elif step_type == "fetch_content":
            return self._transform_fetch_content_step(node)
        elif step_type == "view":
            return self._transform_view_step(node)
        elif step_type == "write_file":
            return self._transform_write_file_step(node)

        return None

    def _transform_parallel_step(self, node: Tree) -> Dict[str, Any]:
        """Transform parallel step: parallel { step1, step2, ... }"""
        inner_steps = []

        for child in node.children:
            if isinstance(child, Tree) and child.data == "step_list":
                for step_item in child.children:
                    if isinstance(step_item, Tree) and step_item.data == "step_item":
                        for inner_node in step_item.children:
                            if isinstance(inner_node, Tree):
                                step = self._transform_step(inner_node)
                                if step:
                                    inner_steps.append(step)

        return {
            "type": "parallel",
            "steps": inner_steps,
        }

    def _transform_join_step(self, node: Tree) -> Dict[str, Any]:
        """Transform join step: join { left: key, right: source, right_key: key, type: inner }"""
        result = {
            "type": "join",
            "left_key": None,
            "right_source": None,
            "right_key": None,
            "join_type": "inner",
        }

        # Find join_spec
        for child in node.children:
            if isinstance(child, Tree) and child.data == "join_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "join_left_key":
                            result["left_key"] = str(param.children[0])
                        elif param.data == "join_right_source":
                            # Extract the source item
                            for src in param.children:
                                if isinstance(src, Tree) and src.data == "source_item":
                                    for src_node in src.children:
                                        if isinstance(src_node, Tree):
                                            result["right_source"] = self._extract_single_source(src_node)
                        elif param.data == "join_right_key":
                            result["right_key"] = str(param.children[0])
                        elif param.data == "join_on_key":
                            # on: key sets both left and right keys
                            key = str(param.children[0])
                            result["left_key"] = key
                            if result["right_key"] is None:
                                result["right_key"] = key
                        elif param.data == "join_type_spec":
                            # Extract join type from the nested node
                            for type_node in param.children:
                                if isinstance(type_node, Tree):
                                    jt = type_node.data.replace("join_", "")
                                    result["join_type"] = jt

        return result

    def _transform_when_step(self, node: Tree) -> Dict[str, Any]:
        """Transform when step: when { condition } step"""
        condition = None
        inner_step = None

        for child in node.children:
            if isinstance(child, Tree):
                if child.data in ("or_cond", "and_cond", "not_cond", "comparison",
                                  "or_expr", "and_expr", "not_expr", "binary_comp",
                                  "paren_cond"):
                    condition = compile_condition(child)
                elif child.data.endswith("_step") or child.data == "step":
                    # Handle nested step
                    if child.data == "step":
                        for step_child in child.children:
                            if isinstance(step_child, Tree):
                                inner_step = self._transform_step(step_child)
                                break
                    else:
                        inner_step = self._transform_step(child)

        return {
            "type": "when",
            "condition": condition,
            "inner_step": inner_step,
        }

    def _transform_unless_step(self, node: Tree) -> Dict[str, Any]:
        """Transform unless step: unless { condition } step"""
        condition = None
        inner_step = None

        for child in node.children:
            if isinstance(child, Tree):
                if child.data in ("or_cond", "and_cond", "not_cond", "comparison",
                                  "or_expr", "and_expr", "not_expr", "binary_comp",
                                  "paren_cond"):
                    condition = compile_condition(child)
                elif child.data.endswith("_step") or child.data == "step":
                    if child.data == "step":
                        for step_child in child.children:
                            if isinstance(step_child, Tree):
                                inner_step = self._transform_step(step_child)
                                break
                    else:
                        inner_step = self._transform_step(child)

        return {
            "type": "unless",
            "condition": condition,
            "inner_step": inner_step,
        }

    def _transform_branch_step(self, node: Tree) -> Dict[str, Any]:
        """Transform branch step: branch { condition } then step [else step]"""
        condition = None
        then_step = None
        else_step = None
        found_then = False

        for child in node.children:
            if isinstance(child, Token) and str(child) == "then":
                found_then = True
                continue
            if isinstance(child, Token) and str(child) == "else":
                found_then = False  # Next step is else
                continue

            if isinstance(child, Tree):
                if child.data in ("or_cond", "and_cond", "not_cond", "comparison",
                                  "or_expr", "and_expr", "not_expr", "binary_comp",
                                  "paren_cond"):
                    condition = compile_condition(child)
                elif child.data.endswith("_step") or child.data == "step":
                    step = None
                    if child.data == "step":
                        for step_child in child.children:
                            if isinstance(step_child, Tree):
                                step = self._transform_step(step_child)
                                break
                    else:
                        step = self._transform_step(child)

                    if then_step is None:
                        then_step = step
                    else:
                        else_step = step

        return {
            "type": "branch",
            "condition": condition,
            "then_step": then_step,
            "else_step": else_step,
        }

    def _transform_catch_step(self, node: Tree) -> Dict[str, Any]:
        """Transform catch step: catch { default_value }"""
        default_expr = None

        for child in node.children:
            if isinstance(child, Tree):
                default_expr = compile_expression(child)

        return {
            "type": "catch",
            "default": default_expr,
        }

    def _transform_filter_step(self, node: Tree) -> Dict[str, Any]:
        """Transform filter step."""
        # Find condition
        for child in node.children:
            if isinstance(child, Tree):
                predicate = compile_condition(child)
                return {
                    "type": "filter",
                    "predicate": predicate,
                    "_condition_node": child,  # For debugging
                }
        return {"type": "filter", "predicate": lambda r, ctx=None: True}

    def _transform_select_step(self, node: Tree) -> Dict[str, Any]:
        """Transform select step.

        Handles:
        - field_simple: NAME -> select field as-is
        - field_alias: NAME as NAME -> select field with alias
        - field_expr: NAME: expr -> computed field (may convert to map step)
        """
        fields = {}  # output_name -> source_name
        expressions = {}  # output_name -> compiled expression
        has_expressions = False

        field_list = self._find_child(node, "field_list")
        if field_list:
            for item in field_list.children:
                if isinstance(item, Tree):
                    if item.data == "field_simple":
                        name = str(item.children[0])
                        fields[name] = name
                    elif item.data == "field_alias":
                        source = str(item.children[0])
                        alias = str(item.children[1])
                        fields[alias] = source
                    elif item.data == "field_expr":
                        # NAME: expr - could be rename or computed value
                        alias = str(item.children[0])
                        expr_node = item.children[1]

                        # Check if expr is a simple field reference (NAME)
                        if isinstance(expr_node, Tree) and expr_node.data == "field_ref":
                            # Simple rename: new_name: old_name
                            source = str(expr_node.children[0])
                            fields[alias] = source
                        else:
                            # Computed expression: type: "class" or complex expr
                            expr_func = compile_expression(expr_node)
                            expressions[alias] = expr_func
                            has_expressions = True

        # If we have computed expressions, convert to map step
        if has_expressions:
            # Build a transform function that combines field selection and expressions
            def make_transform(fields_map, expr_map):
                def transform(row, ctx=None):
                    result = {}
                    # Copy selected/renamed fields (handle ?-prefixed REQL keys)
                    for out_name, src_name in fields_map.items():
                        clean_out = out_name.lstrip("?")
                        if src_name in row:
                            result[clean_out] = row[src_name]
                        elif not src_name.startswith("?") and f"?{src_name}" in row:
                            result[clean_out] = row[f"?{src_name}"]
                        elif out_name in row:
                            result[clean_out] = row[out_name]
                        elif not out_name.startswith("?") and f"?{out_name}" in row:
                            result[clean_out] = row[f"?{out_name}"]
                    # Add computed expressions
                    for out_name, expr_func in expr_map.items():
                        try:
                            result[out_name] = expr_func(row, ctx)
                        except Exception:
                            result[out_name] = None
                    return result
                return transform

            return {"type": "map", "transform": make_transform(fields, expressions)}

        return {"type": "select", "fields": fields}

    def _transform_map_step(self, node: Tree) -> Dict[str, Any]:
        """Transform map step."""
        # Find object expression
        obj_expr = self._find_child(node, "object_expr")
        if obj_expr:
            transform = compile_object_expr(obj_expr)
            return {"type": "map", "transform": transform}

        return {"type": "map", "transform": lambda r, ctx=None: r}

    def _transform_flat_map_step(self, node: Tree) -> Dict[str, Any]:
        """Transform flat_map step."""
        for child in node.children:
            if isinstance(child, Tree):
                expr = compile_expression(child)
                return {"type": "flat_map", "transform": expr}

        return {"type": "flat_map", "transform": lambda r, ctx=None: [r]}

    def _transform_order_by_step(self, node: Tree) -> Dict[str, Any]:
        """Transform order_by step."""
        orders = []

        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "order_desc":
                    field = str(child.children[0])
                    orders.append(("-" + field, True))
                elif child.data == "order_asc":
                    field = str(child.children[0])
                    orders.append((field, False))
                elif child.data == "order_asc_default":
                    field = str(child.children[0])
                    orders.append((field, False))

        return {"type": "order_by", "orders": orders}

    def _transform_limit_step(self, node: Tree) -> Dict[str, Any]:
        """Transform limit step."""
        for child in node.children:
            if isinstance(child, Tree) and child.data == "limit_value":
                return self._extract_limit_value(child, "limit")
            elif isinstance(child, Token) and child.type == "INT":
                return {"type": "limit", "count": int(str(child))}

        return {"type": "limit", "count": 100}

    def _transform_offset_step(self, node: Tree) -> Dict[str, Any]:
        """Transform offset step."""
        for child in node.children:
            if isinstance(child, Tree) and child.data == "offset_value":
                return self._extract_limit_value(child, "offset")
            elif isinstance(child, Token) and child.type == "INT":
                return {"type": "offset", "count": int(str(child))}

        return {"type": "offset", "count": 0}

    def _extract_limit_value(self, node: Tree, step_type: str) -> Dict[str, Any]:
        """Extract limit/offset value (may be param ref or int)."""
        for child in node.children:
            if isinstance(child, Token) and child.type == "INT":
                return {"type": step_type, "count": int(str(child))}
            elif isinstance(child, Tree) and child.data == "param_ref":
                param = str(child.children[0])
                return {"type": step_type, "param": param}

        return {"type": step_type, "count": 100 if step_type == "limit" else 0}

    def _transform_group_by_step(self, node: Tree) -> Dict[str, Any]:
        """Transform group_by step."""
        group_spec = self._find_child(node, "group_spec")
        if not group_spec:
            return {"type": "group_by", "field": None}

        result = {"type": "group_by", "field": None, "aggregate": None}

        for child in group_spec.children:
            if isinstance(child, Tree):
                if child.data == "group_field":
                    result["field"] = str(child.children[0])
                elif child.data == "group_lambda":
                    # key: row => expr
                    lambda_expr = self._find_child(child, "lambda_expr")
                    if lambda_expr:
                        expr = lambda_expr.children[-1]  # Expression after =>
                        result["key_fn"] = compile_expression(expr)
                elif child.data == "group_all":
                    result["field"] = "_all"
                elif child.data in ("agg_func_ref", "agg_inline"):
                    result["aggregate"] = self._extract_aggregate(child)

        return result

    def _extract_aggregate(self, node: Tree) -> Dict[str, Tuple[str, str]]:
        """Extract aggregate specification."""
        aggs = {}

        if node.data == "agg_func_ref":
            # Just a function name reference
            return {"_ref": str(node.children[0])}

        # Inline aggregation: {field: op(source), ...}
        for child in node.children:
            if isinstance(child, Tree) and child.data == "agg_field_list":
                for agg_field in child.children:
                    if isinstance(agg_field, Tree) and agg_field.data == "agg_field":
                        out_name = str(agg_field.children[0])
                        op_node = agg_field.children[1]
                        source = str(agg_field.children[2])
                        op = op_node.data.replace("agg_", "")
                        aggs[out_name] = (source, op)

        return aggs

    def _transform_aggregate_step(self, node: Tree) -> Dict[str, Any]:
        """Transform aggregate step."""
        aggs = {}

        for child in node.children:
            if isinstance(child, Tree) and child.data == "agg_field":
                out_name = str(child.children[0])
                op_node = child.children[1]
                source = str(child.children[2])
                op = op_node.data.replace("agg_", "")
                aggs[out_name] = (source, op)

        return {"type": "aggregate", "aggregations": aggs}

    def _transform_unique_step(self, node: Tree) -> Dict[str, Any]:
        """Transform unique step."""
        key = None
        for child in node.children:
            if isinstance(child, Token) and child.type == "NAME":
                key = str(child)

        if key:
            return {"type": "unique", "key": lambda r, k=key: r.get(k) if isinstance(r, dict) else r}
        return {"type": "unique", "key": None}

    def _transform_tap_step(self, node: Tree) -> Dict[str, Any]:
        """Transform tap step."""
        func_name = None
        for child in node.children:
            if isinstance(child, Token) and child.type == "NAME":
                func_name = str(child)

        return {"type": "tap", "func_name": func_name}

    def _transform_python_step(self, node: Tree) -> Dict[str, Any]:
        """Transform python step."""
        code = ""
        for child in node.children:
            if isinstance(child, Token) and child.type == "PYTHON_BLOCK":
                content = str(child)
                # Remove outer braces
                if content.startswith("{") and content.endswith("}"):
                    code = content[1:-1]
        # Dedent the code to remove common leading whitespace
        code = textwrap.dedent(code)
        return {"type": "python", "code": code}

    def _transform_render_step(self, node: Tree) -> Dict[str, Any]:
        """Transform render step."""
        render_spec = self._find_child(node, "render_spec")
        if not render_spec:
            return {"type": "render", "format": "text", "renderer": None}

        result = {"type": "render", "format": "text", "renderer": None}

        for child in render_spec.children:
            if isinstance(child, Tree):
                if child.data == "render_with_format":
                    # format: "x", renderer: name
                    for item in child.children:
                        if isinstance(item, Tree) and item.data == "format_value":
                            for fmt in item.children:
                                if isinstance(fmt, Token) and fmt.type == "STRING":
                                    result["format"] = unquote(str(fmt))
                                elif isinstance(fmt, Tree) and fmt.data == "param_ref":
                                    result["format_param"] = str(fmt.children[0])
                        elif isinstance(item, Token) and item.type == "NAME":
                            result["renderer"] = str(item)

                elif child.data == "render_func":
                    result["renderer"] = str(child.children[0])

        return result

    def _transform_emit_step(self, node: Tree) -> Dict[str, Any]:
        """Transform emit step - supports single or multiple outputs."""
        outputs = {}

        # Find emit_spec
        for child in node.children:
            if isinstance(child, Tree) and child.data == "emit_spec":
                for emit_field in child.children:
                    if isinstance(emit_field, Tree):
                        if emit_field.data == "emit_simple":
                            # Simple: emit { key }
                            name = str(emit_field.children[0])
                            outputs[name] = name
                        elif emit_field.data == "emit_named":
                            # Named: emit { key: source }
                            name = str(emit_field.children[0])
                            source = str(emit_field.children[1])
                            outputs[name] = source
            elif isinstance(child, Token) and child.type == "NAME":
                # Legacy single key format
                key = str(child)
                outputs[key] = key

        if len(outputs) == 1:
            # Single output - use legacy format for compatibility
            key = list(outputs.keys())[0]
            return {"type": "emit", "key": key}
        else:
            # Multiple outputs
            return {"type": "emit", "outputs": outputs}

    def _transform_graph_cycles_step(self, node: Tree) -> Dict[str, Any]:
        """Transform graph_cycles step: graph_cycles { from: x, to: y }"""
        result = {"type": "graph_cycles", "from_field": None, "to_field": None}

        for child in node.children:
            if isinstance(child, Tree) and child.data == "graph_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "graph_from":
                            result["from_field"] = str(param.children[0])
                        elif param.data == "graph_to":
                            result["to_field"] = str(param.children[0])

        return result

    def _transform_graph_closure_step(self, node: Tree) -> Dict[str, Any]:
        """Transform graph_closure step: graph_closure { from: x, to: y, max_depth: 10 }"""
        result = {"type": "graph_closure", "from_field": None, "to_field": None, "max_depth": 10}

        for child in node.children:
            if isinstance(child, Tree) and child.data == "graph_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "graph_from":
                            result["from_field"] = str(param.children[0])
                        elif param.data == "graph_to":
                            result["to_field"] = str(param.children[0])
                        elif param.data == "graph_max_depth":
                            result["max_depth"] = int(str(param.children[0]))
                        elif param.data == "graph_max_depth_param":
                            result["max_depth_param"] = str(param.children[0].children[0])

        return result

    def _transform_graph_traverse_step(self, node: Tree) -> Dict[str, Any]:
        """Transform graph_traverse step: graph_traverse { from: x, to: y, algorithm: bfs }"""
        result = {"type": "graph_traverse", "from_field": None, "to_field": None,
                  "algorithm": "bfs", "max_depth": 10, "root": None}

        for child in node.children:
            if isinstance(child, Tree) and child.data == "graph_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "graph_from":
                            result["from_field"] = str(param.children[0])
                        elif param.data == "graph_to":
                            result["to_field"] = str(param.children[0])
                        elif param.data == "graph_max_depth":
                            result["max_depth"] = int(str(param.children[0]))
                        elif param.data == "graph_algorithm":
                            algo_node = param.children[0]
                            if isinstance(algo_node, Tree):
                                result["algorithm"] = "bfs" if algo_node.data == "algo_bfs" else "dfs"
                        elif param.data == "graph_root":
                            result["root"] = compile_expression(param.children[0])

        return result

    def _transform_render_mermaid_step(self, node: Tree) -> Dict[str, Any]:
        """Transform render_mermaid step.

        Uses dispatch table pattern to handle different mermaid parameter types.
        """
        result = {
            "type": "render_mermaid",
            "mermaid_type": "flowchart",
            "nodes": None,
            "edges_from": None,
            "edges_to": None,
            "direction": "TB",
            "title": None,
            "participants": None,
            "messages_from": None,
            "messages_to": None,
            "messages_label": None,
            # Class diagram
            "classes": None,
            "methods": None,
            "attributes": None,
            "inheritance_from": None,
            "inheritance_to": None,
            "composition_from": None,
            "composition_to": None,
            "association_from": None,
            "association_to": None,
            # Pie chart
            "labels": None,
            "values": None,
            # State diagram
            "states": None,
            "transitions_from": None,
            "transitions_to": None,
            # ER diagram
            "entities": None,
            "relationships": None,
            # Block diagram
            "groups": None,
            "columns": 4,
            "color": None,
            "max_per_group": 20,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "mermaid_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        self._apply_mermaid_param(param, result)

        return result

    def _apply_mermaid_param(self, param: Tree, result: Dict[str, Any]) -> None:
        """Apply a single mermaid parameter to the result dict.

        Dispatch table pattern replaces long if-elif chain.
        """
        # Single-value parameters: param.data -> result key
        single_value_params = {
            "mermaid_nodes": "nodes",
            "mermaid_participants": "participants",
            "mermaid_classes": "classes",
            "mermaid_methods": "methods",
            "mermaid_attributes": "attributes",
            "mermaid_labels": "labels",
            "mermaid_values": "values",
            "mermaid_states": "states",
            "mermaid_entities": "entities",
            "mermaid_relationships": "relationships",
            "mermaid_groups": "groups",
            "mermaid_color": "color",
        }

        # Dual-value parameters: param.data -> (from_key, to_key)
        dual_value_params = {
            "mermaid_edges": ("edges_from", "edges_to"),
            "mermaid_inheritance": ("inheritance_from", "inheritance_to"),
            "mermaid_composition": ("composition_from", "composition_to"),
            "mermaid_association": ("association_from", "association_to"),
            "mermaid_transitions": ("transitions_from", "transitions_to"),
        }

        param_type = param.data

        # Handle single-value parameters
        if param_type in single_value_params:
            result[single_value_params[param_type]] = str(param.children[0])

        # Handle dual-value parameters
        elif param_type in dual_value_params:
            from_key, to_key = dual_value_params[param_type]
            result[from_key] = str(param.children[0])
            result[to_key] = str(param.children[1])

        # Handle special cases
        elif param_type == "mermaid_type_spec":
            type_node = param.children[0]
            if isinstance(type_node, Tree):
                result["mermaid_type"] = type_node.data.replace("mermaid_", "")

        elif param_type == "mermaid_direction":
            dir_node = param.children[0]
            if isinstance(dir_node, Tree):
                result["direction"] = dir_node.data.replace("dir_", "").upper()

        elif param_type == "mermaid_title":
            result["title"] = unquote(str(param.children[0]))

        elif param_type == "mermaid_messages":
            result["messages_from"] = str(param.children[0])
            result["messages_to"] = str(param.children[1])
            result["messages_label"] = str(param.children[2])

        elif param_type == "mermaid_columns":
            result["columns"] = int(str(param.children[0]))

        elif param_type == "mermaid_columns_param":
            result["columns_param"] = str(param.children[0].children[0])

        elif param_type == "mermaid_max_per_group":
            result["max_per_group"] = int(str(param.children[0]))

        elif param_type == "mermaid_max_per_group_param":
            result["max_per_group_param"] = str(param.children[0].children[0])

    def _transform_pivot_step(self, node: Tree) -> Dict[str, Any]:
        """Transform pivot step: pivot { rows: x, cols: y, value: z, aggregate: sum }"""
        result = {"type": "pivot", "rows": None, "cols": None, "value": None, "aggregate": "sum"}

        for child in node.children:
            if isinstance(child, Tree) and child.data == "pivot_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "pivot_rows":
                            result["rows"] = str(param.children[0])
                        elif param.data == "pivot_cols":
                            result["cols"] = str(param.children[0])
                        elif param.data == "pivot_value":
                            result["value"] = str(param.children[0])
                        elif param.data == "pivot_aggregate":
                            agg_node = param.children[0]
                            if isinstance(agg_node, Tree):
                                result["aggregate"] = agg_node.data.replace("agg_", "")

        return result

    def _transform_compute_step(self, node: Tree) -> Dict[str, Any]:
        """Transform compute step: compute { ratio: a / b, pct: ratio * 100 }"""
        computations = {}

        for child in node.children:
            if isinstance(child, Tree) and child.data == "compute_field":
                name = str(child.children[0])
                expr = compile_expression(child.children[1])
                computations[name] = expr

        return {"type": "compute", "computations": computations}

    def _transform_collect_step(self, node: Tree) -> Dict[str, Any]:
        """Transform collect step: collect { by: field, name: op(field) }"""
        result = {"type": "collect", "by": None, "fields": {}}

        for child in node.children:
            if isinstance(child, Tree) and child.data == "collect_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "collect_by":
                            result["by"] = str(param.children[0])
                        elif param.data == "collect_field":
                            # name: op(field)
                            name = str(param.children[0])
                            op_node = param.children[1]
                            source = str(param.children[2])
                            if isinstance(op_node, Tree):
                                op = op_node.data.replace("collect_", "")
                            else:
                                op = "list"
                            result["fields"][name] = (source, op)

        return result

    def _transform_nest_step(self, node: Tree) -> Dict[str, Any]:
        """Transform nest step: nest { parent: field, child: field, root: expr }"""
        result = {
            "type": "nest",
            "parent": None,
            "child": None,
            "root": None,
            "max_depth": 10,
            "children_key": "children",
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "nest_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "nest_parent":
                            result["parent"] = str(param.children[0])
                        elif param.data == "nest_child":
                            result["child"] = str(param.children[0])
                        elif param.data == "nest_root":
                            result["root"] = compile_expression(param.children[0])
                        elif param.data == "nest_max_depth":
                            result["max_depth"] = int(str(param.children[0]))
                        elif param.data == "nest_max_depth_param":
                            result["max_depth_param"] = str(param.children[0].children[0])
                        elif param.data == "nest_children_key":
                            result["children_key"] = unquote(str(param.children[0]))

        return result

    def _transform_render_table_step(self, node: Tree) -> Dict[str, Any]:
        """Transform render_table step."""
        result = {
            "type": "render_table",
            "format": "markdown",
            "columns": [],
            "title": None,
            "totals": False,
            "sort": None,
            "group_by": None,
            "max_rows": None,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "table_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "table_format_spec":
                            fmt_node = param.children[0]
                            if isinstance(fmt_node, Tree):
                                result["format"] = fmt_node.data.replace("tbl_", "")
                        elif param.data == "table_columns":
                            result["columns"] = self._extract_column_list(param)
                        elif param.data == "table_title":
                            result["title"] = unquote(str(param.children[0]))
                        elif param.data == "table_title_param":
                            result["title_param"] = str(param.children[0].children[0])
                        elif param.data == "table_totals":
                            val_node = param.children[0]
                            if isinstance(val_node, Tree):
                                result["totals"] = val_node.data == "bool_true"
                        elif param.data == "table_sort":
                            result["sort"] = str(param.children[0])
                        elif param.data == "table_group":
                            result["group_by"] = str(param.children[0])
                        elif param.data == "table_max_rows":
                            result["max_rows"] = int(str(param.children[0]))
                        elif param.data == "table_max_rows_param":
                            result["max_rows_param"] = str(param.children[0].children[0])

        return result

    def _extract_column_list(self, node: Tree) -> List[Dict[str, str]]:
        """Extract column definitions from column_list."""
        columns = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "column_list":
                for col in child.children:
                    if isinstance(col, Tree):
                        if col.data == "col_simple":
                            name = str(col.children[0])
                            columns.append({"name": name, "alias": name})
                        elif col.data == "col_alias":
                            name = str(col.children[0])
                            alias = unquote(str(col.children[1]))
                            columns.append({"name": name, "alias": alias})
        return columns

    def _transform_render_chart_step(self, node: Tree) -> Dict[str, Any]:
        """Transform render_chart step."""
        result = {
            "type": "render_chart",
            "chart_type": "bar",
            "x": None,
            "y": None,
            "series": None,
            "title": None,
            "format": "mermaid",
            "colors": None,
            "stacked": False,
            "horizontal": False,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "chart_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "chart_type_spec":
                            type_node = param.children[0]
                            if isinstance(type_node, Tree):
                                result["chart_type"] = type_node.data.replace("chart_", "")
                        elif param.data == "chart_x":
                            result["x"] = str(param.children[0])
                        elif param.data == "chart_y":
                            result["y"] = str(param.children[0])
                        elif param.data == "chart_series":
                            result["series"] = str(param.children[0])
                        elif param.data == "chart_title":
                            result["title"] = unquote(str(param.children[0]))
                        elif param.data == "chart_title_param":
                            result["title_param"] = str(param.children[0].children[0])
                        elif param.data == "chart_format_spec":
                            fmt_node = param.children[0]
                            if isinstance(fmt_node, Tree):
                                result["format"] = fmt_node.data.replace("chart_fmt_", "")
                        elif param.data == "chart_colors":
                            result["colors"] = self._extract_color_list(param)
                        elif param.data == "chart_stacked":
                            val_node = param.children[0]
                            if isinstance(val_node, Tree):
                                result["stacked"] = val_node.data == "bool_true"
                        elif param.data == "chart_horizontal":
                            val_node = param.children[0]
                            if isinstance(val_node, Tree):
                                result["horizontal"] = val_node.data == "bool_true"

        return result

    def _extract_color_list(self, node: Tree) -> List[str]:
        """Extract color list from color_list."""
        colors = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "color_list":
                for item in child.children:
                    if isinstance(item, Token) and item.type == "STRING":
                        colors.append(unquote(str(item)))
        return colors

    def _transform_cross_join_step(self, node: Tree) -> Dict[str, Any]:
        """Transform cross_join step: cross_join { unique_pairs: true, left_prefix: "left_" }"""
        result = {
            "type": "cross_join",
            "unique_pairs": True,
            "exclude_self": True,
            "left_prefix": "left_",
            "right_prefix": "right_",
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "cross_join_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "cj_unique":
                            val_node = param.children[0]
                            if isinstance(val_node, Tree):
                                result["unique_pairs"] = val_node.data == "bool_true"
                        elif param.data == "cj_exclude_self":
                            val_node = param.children[0]
                            if isinstance(val_node, Tree):
                                result["exclude_self"] = val_node.data == "bool_true"
                        elif param.data == "cj_left_prefix":
                            result["left_prefix"] = unquote(str(param.children[0]))
                        elif param.data == "cj_right_prefix":
                            result["right_prefix"] = unquote(str(param.children[0]))

        return result

    def _transform_set_similarity_step(self, node: Tree) -> Dict[str, Any]:
        """Transform set_similarity step: set_similarity { left: col1, right: col2, type: jaccard }"""
        result = {
            "type": "set_similarity",
            "left": None,
            "right": None,
            "sim_type": "jaccard",
            "output": "similarity",
            "intersection_output": None,
            "union_output": None,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "set_sim_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "ss_left":
                            result["left"] = str(param.children[0])
                        elif param.data == "ss_right":
                            result["right"] = str(param.children[0])
                        elif param.data == "ss_type":
                            type_node = param.children[0]
                            if isinstance(type_node, Tree):
                                result["sim_type"] = type_node.data
                            else:
                                result["sim_type"] = str(type_node)
                        elif param.data == "ss_output":
                            result["output"] = str(param.children[0])
                        elif param.data == "ss_intersection":
                            result["intersection_output"] = str(param.children[0])
                        elif param.data == "ss_union":
                            result["union_output"] = str(param.children[0])

        return result

    def _transform_string_match_step(self, node: Tree) -> Dict[str, Any]:
        """Transform string_match step: string_match { left: col1, right: col2, type: common_affix }"""
        result = {
            "type": "string_match",
            "left": None,
            "right": None,
            "match_type": "common_affix",
            "min_length": 3,
            "output": "has_match",
            "match_output": None,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "string_match_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "sm_left":
                            result["left"] = str(param.children[0])
                        elif param.data == "sm_right":
                            result["right"] = str(param.children[0])
                        elif param.data == "sm_type":
                            type_node = param.children[0]
                            if isinstance(type_node, Tree):
                                result["match_type"] = type_node.data
                            else:
                                result["match_type"] = str(type_node)
                        elif param.data == "sm_min_len":
                            result["min_length"] = int(str(param.children[0]))
                        elif param.data == "sm_min_len_param":
                            result["min_length_param"] = str(param.children[0].children[0])
                        elif param.data == "sm_output":
                            result["output"] = str(param.children[0])
                        elif param.data == "sm_match_output":
                            result["match_output"] = str(param.children[0])

        return result

    def _transform_rag_enrich_step(self, node: Tree) -> Dict[str, Any]:
        """Transform rag_enrich step: rag_enrich { query: "template {field}", top_k: 3, mode: "best" }"""
        result = {
            "type": "rag_enrich",
            "query_template": "",
            "top_k": 1,
            "threshold": None,
            "mode": "best",
            "batch_size": 50,
            "max_rows": 1000,
            "entity_types": None,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "rag_enrich_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "re_query":
                            result["query_template"] = unquote(str(param.children[0]))
                        elif param.data == "re_query_param":
                            result["query_template_param"] = str(param.children[0].children[0])
                        elif param.data == "re_top_k":
                            result["top_k"] = int(str(param.children[0]))
                        elif param.data == "re_top_k_param":
                            result["top_k_param"] = str(param.children[0].children[0])
                        elif param.data == "re_threshold":
                            result["threshold"] = float(str(param.children[0]))
                        elif param.data == "re_threshold_param":
                            result["threshold_param"] = str(param.children[0].children[0])
                        elif param.data == "re_mode":
                            mode_node = param.children[0]
                            if isinstance(mode_node, Tree):
                                if mode_node.data == "re_mode_best":
                                    result["mode"] = "best"
                                elif mode_node.data == "re_mode_all":
                                    result["mode"] = "all"
                        elif param.data == "re_batch_size":
                            result["batch_size"] = int(str(param.children[0]))
                        elif param.data == "re_batch_size_param":
                            result["batch_size_param"] = str(param.children[0].children[0])
                        elif param.data == "re_max_rows":
                            result["max_rows"] = int(str(param.children[0]))
                        elif param.data == "re_max_rows_param":
                            result["max_rows_param"] = str(param.children[0].children[0])
                        elif param.data == "re_entity_types":
                            # Extract entity types list
                            entity_list = param.children[0]
                            if isinstance(entity_list, Tree) and entity_list.data == "rag_entity_list":
                                result["entity_types"] = [
                                    unquote(str(t))
                                    for t in entity_list.children
                                    if isinstance(t, Token) and t.type == "STRING"
                                ]

        return result

    def _transform_create_task_step(self, node: Tree) -> Dict[str, Any]:
        """Transform create_task step: create_task { name: "template {field}", category: "annotation", priority: "medium" }"""
        result = {
            "type": "create_task",
            "name_template": "",
            "category": "annotation",
            "priority": "medium",
            "description_template": None,
            "prompt_template": None,
            "affects_field": None,
            "batch_size": 50,
            "dry_run": False,
            "filter_predicates": [],
            "metadata_template": {},
            "group_id": None,
            "source_tool": None,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "create_task_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "ct_name":
                            result["name_template"] = unquote(str(param.children[0]))
                        elif param.data == "ct_name_param":
                            result["name_template_param"] = str(param.children[0].children[0])
                        elif param.data == "ct_category":
                            result["category"] = unquote(str(param.children[0]))
                        elif param.data == "ct_category_param":
                            result["category_param"] = str(param.children[0].children[0])
                        elif param.data == "ct_priority":
                            prio_node = param.children[0]
                            if isinstance(prio_node, Tree):
                                if prio_node.data == "ct_prio_critical":
                                    result["priority"] = "critical"
                                elif prio_node.data == "ct_prio_high":
                                    result["priority"] = "high"
                                elif prio_node.data == "ct_prio_medium":
                                    result["priority"] = "medium"
                                elif prio_node.data == "ct_prio_low":
                                    result["priority"] = "low"
                        elif param.data == "ct_priority_param":
                            result["priority_param"] = str(param.children[0].children[0])
                        elif param.data == "ct_description":
                            result["description_template"] = unquote(str(param.children[0]))
                        elif param.data == "ct_description_param":
                            result["description_template_param"] = str(param.children[0].children[0])
                        elif param.data == "ct_prompt":
                            result["prompt_template"] = unquote(str(param.children[0]))
                        elif param.data == "ct_prompt_param":
                            result["prompt_template_param"] = str(param.children[0].children[0])
                        elif param.data == "ct_affects":
                            result["affects_field"] = str(param.children[0])
                        elif param.data == "ct_batch_size":
                            result["batch_size"] = int(str(param.children[0]))
                        elif param.data == "ct_batch_size_param":
                            result["batch_size_param"] = str(param.children[0].children[0])
                        elif param.data == "ct_dry_run":
                            dry_run_node = param.children[0]
                            if isinstance(dry_run_node, Tree):
                                result["dry_run"] = dry_run_node.data == "bool_true"
                        elif param.data == "ct_dry_run_param":
                            result["dry_run_param"] = str(param.children[0].children[0])
                        elif param.data == "ct_filter_predicates":
                            result["filter_predicates"] = self._extract_ct_predicates(param)
                        elif param.data == "ct_metadata":
                            result["metadata_template"] = self._extract_ct_metadata(param)
                        elif param.data == "ct_group_id":
                            result["group_id"] = unquote(str(param.children[0]))
                        elif param.data == "ct_group_id_param":
                            result["group_id_param"] = str(param.children[0].children[0])
                        elif param.data == "ct_source_tool":
                            result["source_tool"] = unquote(str(param.children[0]))
                        elif param.data == "ct_source_tool_param":
                            result["source_tool_param"] = str(param.children[0].children[0])

        return result

    def _extract_ct_predicates(self, node: Tree) -> List[str]:
        """Extract filter predicates list from ct_filter_predicates node."""
        predicates = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "ct_predicate_list":
                for pred in child.children:
                    if isinstance(pred, Tree) and pred.data == "ct_predicate":
                        for token in pred.children:
                            if isinstance(token, Token) and token.type == "STRING":
                                predicates.append(unquote(str(token)))
        return predicates

    def _extract_ct_metadata(self, node: Tree) -> Dict[str, Any]:
        """Extract metadata template from ct_metadata node."""
        metadata = {}
        for child in node.children:
            if isinstance(child, Tree) and child.data == "ct_metadata_block":
                for pairs_node in child.children:
                    if isinstance(pairs_node, Tree) and pairs_node.data == "ct_metadata_pairs":
                        for pair in pairs_node.children:
                            if isinstance(pair, Tree) and pair.data == "ct_metadata_pair":
                                key = None
                                value = None
                                for item in pair.children:
                                    if isinstance(item, Token) and item.type == "NAME":
                                        key = str(item)
                                    elif isinstance(item, Tree):
                                        value = self._extract_ct_metadata_value(item)
                                if key is not None:
                                    metadata[key] = value
        return metadata

    def _extract_ct_metadata_value(self, node: Tree) -> Any:
        """Extract value from ct_metadata_value node."""
        if node.data == "ct_meta_string":
            return unquote(str(node.children[0]))
        elif node.data == "ct_meta_int":
            return int(str(node.children[0]))
        elif node.data == "ct_meta_float":
            return float(str(node.children[0]))
        elif node.data == "ct_meta_true":
            return True
        elif node.data == "ct_meta_false":
            return False
        elif node.data == "ct_meta_param":
            # Parameter reference like {score} - return as template string
            for child in node.children:
                if isinstance(child, Tree) and child.data == "param_ref":
                    for item in child.children:
                        if isinstance(item, Token) and item.type == "NAME":
                            return "{" + str(item) + "}"
        elif node.data == "ct_meta_field":
            # Field reference like score - return as template string
            return "{" + str(node.children[0]) + "}"
        return None

    def _transform_view_step(self, node: Tree) -> Dict[str, Any]:
        """Transform view step: view { skip: false, content: diagram, type: mermaid }"""
        result = {
            "type": "view",
            "skip": False,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "view_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "view_skip":
                            # bool_val node
                            bool_node = param.children[0]
                            if isinstance(bool_node, Tree):
                                result["skip"] = bool_node.data == "bool_true"
                            else:
                                result["skip"] = str(bool_node).lower() == "true"
                        elif param.data == "view_skip_param":
                            # param_ref node
                            ref_node = param.children[0]
                            if isinstance(ref_node, Tree) and ref_node.data == "param_ref":
                                result["skip_param"] = str(ref_node.children[0])
                            else:
                                result["skip_param"] = str(ref_node)
                        elif param.data == "view_content":
                            result["content_key"] = str(param.children[0])
                        elif param.data == "view_type_param":
                            # view_content_type node
                            type_node = param.children[0]
                            if isinstance(type_node, Tree):
                                result["content_type"] = type_node.data.replace("view_", "")
                            else:
                                result["content_type"] = str(type_node)
                        elif param.data == "view_description":
                            result["description"] = unquote(str(param.children[0]))
                        elif param.data == "view_description_param":
                            ref_node = param.children[0]
                            if isinstance(ref_node, Tree) and ref_node.data == "param_ref":
                                result["description_param"] = str(ref_node.children[0])
                            else:
                                result["description_param"] = str(ref_node)

        return result

    def _transform_write_file_step(self, node: Tree) -> Dict[str, Any]:
        """Transform write_file step: write_file { path: "out.csv", format: csv }"""
        result = {
            "type": "write_file",
            "path": "",
            "format": "json",
            "encoding": "utf-8",
            "separator": ",",
            "indent": 2,
            "overwrite": True,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "write_file_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "wf_path":
                            result["path"] = self._extract_fs_string(param)
                        elif param.data == "wf_path_param":
                            result["path"] = self._extract_fs_param_ref(param)
                        elif param.data == "wf_format":
                            for fmt_child in param.children:
                                if isinstance(fmt_child, Tree):
                                    result["format"] = fmt_child.data.replace("wf_", "")
                        elif param.data == "wf_format_param":
                            result["format"] = self._extract_fs_param_ref(param)
                        elif param.data == "wf_encoding":
                            result["encoding"] = self._extract_fs_string(param)
                        elif param.data == "wf_encoding_param":
                            result["encoding"] = self._extract_fs_param_ref(param)
                        elif param.data == "wf_separator":
                            result["separator"] = self._extract_fs_string(param)
                        elif param.data == "wf_separator_param":
                            result["separator"] = self._extract_fs_param_ref(param)
                        elif param.data == "wf_indent":
                            result["indent"] = self._extract_fs_int(param)
                        elif param.data == "wf_indent_param":
                            result["indent"] = self._extract_fs_param_ref(param)
                        elif param.data == "wf_overwrite":
                            result["overwrite"] = self._extract_fs_bool(param)

        return result

    def _transform_fetch_content_step(self, node: Tree) -> Dict[str, Any]:
        """Transform fetch_content step: fetch_content { file: file_field, start_line: line, end_line: end, output: body }"""
        result = {
            "type": "fetch_content",
            "file_field": "file",
            "start_line_field": "line",
            "end_line_field": None,
            "output_field": "body",
            "max_lines": 50,
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "fetch_content_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "fc_file":
                            result["file_field"] = str(param.children[0])
                        elif param.data == "fc_start_line":
                            result["start_line_field"] = str(param.children[0])
                        elif param.data == "fc_end_line":
                            result["end_line_field"] = str(param.children[0])
                        elif param.data == "fc_output":
                            result["output_field"] = str(param.children[0])
                        elif param.data == "fc_max_lines":
                            result["max_lines"] = int(str(param.children[0]))
                        elif param.data == "fc_max_lines_param":
                            result["max_lines_param"] = str(param.children[0].children[0])

        return result

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _extract_docstring(self, node: Tree) -> str:
        """Extract docstring text."""
        for child in node.children:
            if isinstance(child, Token) and child.type == "TRIPLE_STRING":
                text = str(child)
                # Remove triple quotes
                if text.startswith('"""') and text.endswith('"""'):
                    return text[3:-3].strip()
        return ""

    def _extract_value(self, node: Tree) -> Any:
        """Extract a value from a value node."""
        if isinstance(node, Token):
            return self._token_to_value(node)

        for child in node.children:
            if isinstance(child, Token):
                return self._token_to_value(child)
            elif isinstance(child, Tree):
                if child.data == "val_string":
                    return unquote(str(child.children[0]))
                elif child.data == "val_int":
                    return int(str(child.children[0]))
                elif child.data == "val_float":
                    return float(str(child.children[0]))
                elif child.data == "val_true":
                    return True
                elif child.data == "val_false":
                    return False
                elif child.data == "val_null":
                    return None
                else:
                    return self._extract_value(child)

        return None

    def _token_to_value(self, token: Token) -> Any:
        """Convert token to Python value."""
        if token.type == "STRING":
            return unquote(str(token))
        elif token.type in ("SIGNED_INT", "INT"):
            return int(str(token))
        elif token.type in ("SIGNED_FLOAT", "FLOAT"):
            return float(str(token))
        return str(token)

    def _extract_expr_as_string(self, node: Tree) -> str:
        """Extract expression as string representation."""
        if isinstance(node, Token):
            if node.type == "STRING":
                return unquote(str(node))
            return str(node)

        # For complex expressions, return a placeholder
        return "{expr}"


# ============================================================
# PIPELINE BUILDER
# ============================================================

class PipelineBuilder:
    """
    Builds executable Pipeline objects from ToolSpec.

    This separates the AST transformation from Pipeline construction,
    allowing for different target Pipeline implementations.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a builder.
    """

    def __init__(self):
        pass

    def build(self, spec: ToolSpec) -> Callable:
        """
        Build a pipeline factory from a ToolSpec.

        Args:
            spec: ToolSpec from CADSLTransformer

        Returns:
            A callable that creates a Pipeline when given a Context
        """
        # Import here to avoid circular imports
        from reter_code.dsl.core import (
            Pipeline, REQLSource, ValueSource,
            RAGSearchSource, RAGDuplicatesSource, RAGClustersSource, RAGDBScanSource,
            FilterStep, SelectStep, MapStep, FlatMapStep,
            OrderByStep, LimitStep, OffsetStep,
            GroupByStep, AggregateStep, FlattenStep, UniqueStep,
            TapStep, RenderStep,
        )

        def pipeline_factory(ctx):
            # Create source
            if spec.source_type == "reql":
                pipeline = Pipeline(_source=REQLSource(spec.source_content))
            elif spec.source_type == "rag_search":
                params = spec.rag_params
                pipeline = Pipeline(_source=RAGSearchSource(
                    query=params.get("query", ""),
                    top_k=params.get("top_k", 10),
                    entity_types=params.get("entity_types"),
                ))
            elif spec.source_type == "rag_duplicates":
                params = spec.rag_params
                pipeline = Pipeline(_source=RAGDuplicatesSource(
                    similarity=params.get("similarity", 0.85),
                    limit=params.get("limit", 50),
                    exclude_same_file=params.get("exclude_same_file", True),
                    exclude_same_class=params.get("exclude_same_class", True),
                    entity_types=params.get("entity_types"),
                ))
            elif spec.source_type == "rag_clusters":
                params = spec.rag_params
                pipeline = Pipeline(_source=RAGClustersSource(
                    n_clusters=params.get("n_clusters", 50),
                    min_size=params.get("min_size", 2),
                    exclude_same_file=params.get("exclude_same_file", True),
                    exclude_same_class=params.get("exclude_same_class", True),
                    entity_types=params.get("entity_types"),
                ))
            elif spec.source_type == "rag_dbscan":
                params = spec.rag_params
                pipeline = Pipeline(_source=RAGDBScanSource(
                    eps=params.get("eps", 0.5),
                    min_samples=params.get("min_samples", 3),
                    min_size=params.get("min_size", 2),
                    exclude_same_file=params.get("exclude_same_file", True),
                    exclude_same_class=params.get("exclude_same_class", True),
                    entity_types=params.get("entity_types"),
                ))
            elif spec.source_type == "value":
                pipeline = Pipeline(_source=ValueSource(spec.source_content))
            elif spec.source_type == "merge":
                # Create MergeSource from multiple sub-sources
                pipeline = Pipeline(_source=MergeSource(spec.merge_sources))
            else:
                pipeline = Pipeline(_source=ValueSource([]))

            # Add steps
            emit_key = None
            for step_spec in spec.steps:
                step_type = step_spec.get("type")

                if step_type == "filter":
                    predicate = step_spec.get("predicate", lambda r, ctx=None: True)
                    pipeline = pipeline.filter(predicate)

                elif step_type == "select":
                    fields = step_spec.get("fields", {})
                    pipeline = pipeline >> SelectStep(fields)

                elif step_type == "map":
                    transform = step_spec.get("transform", lambda r, ctx=None: r)
                    pipeline = pipeline.map(transform)

                elif step_type == "flat_map":
                    transform = step_spec.get("transform", lambda r, ctx=None: [r])
                    pipeline = pipeline.flat_map(transform)

                elif step_type == "order_by":
                    orders = step_spec.get("orders", [])
                    for field, desc in orders:
                        if desc:
                            pipeline = pipeline.order_by("-" + field.lstrip("-"))
                        else:
                            pipeline = pipeline.order_by(field)

                elif step_type == "limit":
                    if "param" in step_spec:
                        param = step_spec["param"]
                        count = ctx.params.get(param, 100)
                    else:
                        count = step_spec.get("count", 100)
                    pipeline = pipeline.limit(count)

                elif step_type == "offset":
                    if "param" in step_spec:
                        param = step_spec["param"]
                        count = ctx.params.get(param, 0)
                    else:
                        count = step_spec.get("count", 0)
                    pipeline = pipeline.offset(count)

                elif step_type == "group_by":
                    field = step_spec.get("field")
                    key_fn = step_spec.get("key_fn")
                    agg = step_spec.get("aggregate")
                    pipeline = pipeline.group_by(field=field, key=key_fn)

                elif step_type == "aggregate":
                    aggs = step_spec.get("aggregations", {})
                    pipeline = pipeline >> AggregateStep(aggs)

                elif step_type == "flatten":
                    pipeline = pipeline.flatten()

                elif step_type == "unique":
                    key = step_spec.get("key")
                    pipeline = pipeline.unique(key)

                elif step_type == "python":
                    code = step_spec.get("code", "result = rows")
                    pipeline = pipeline >> PythonStep(code)

                elif step_type == "render":
                    fmt = step_spec.get("format", "text")
                    if "format_param" in step_spec:
                        fmt = ctx.params.get(step_spec["format_param"], fmt)
                    renderer_name = step_spec.get("renderer")
                    renderer = get_renderer(renderer_name) if renderer_name else default_renderer
                    pipeline = pipeline.render(fmt, renderer)

                elif step_type == "emit":
                    emit_key = step_spec.get("key", "result")

                elif step_type == "when":
                    # Conditional execution - only execute inner step when condition is true
                    condition = step_spec.get("condition", lambda r, ctx=None: True)
                    inner_step = step_spec.get("inner_step")
                    if inner_step:
                        pipeline = pipeline >> WhenStep(condition, inner_step)

                elif step_type == "unless":
                    # Inverted conditional - only execute inner step when condition is false
                    condition = step_spec.get("condition", lambda r, ctx=None: False)
                    inner_step = step_spec.get("inner_step")
                    if inner_step:
                        pipeline = pipeline >> UnlessStep(condition, inner_step)

                elif step_type == "branch":
                    # Branching - execute then_step if condition, else else_step
                    condition = step_spec.get("condition", lambda r, ctx=None: True)
                    then_step = step_spec.get("then_step")
                    else_step = step_spec.get("else_step")
                    pipeline = pipeline >> BranchStep(condition, then_step, else_step)

                elif step_type == "catch":
                    # Error handling - return default if previous steps fail
                    default_fn = step_spec.get("default", lambda r, ctx=None: [])
                    pipeline = pipeline >> CatchStep(default_fn)

                elif step_type == "parallel":
                    # Execute multiple steps in parallel
                    inner_steps = step_spec.get("steps", [])
                    pipeline = pipeline >> ParallelStep(inner_steps)

                elif step_type == "join":
                    # Join with another source using PyArrow
                    left_key = step_spec.get("left_key")
                    right_source = step_spec.get("right_source")
                    right_key = step_spec.get("right_key", left_key)
                    join_type = step_spec.get("join_type", "inner")
                    pipeline = pipeline >> JoinStep(left_key, right_source, right_key, join_type)

                elif step_type == "graph_cycles":
                    from_field = step_spec.get("from_field")
                    to_field = step_spec.get("to_field")
                    pipeline = pipeline >> GraphCyclesStep(from_field, to_field)

                elif step_type == "graph_closure":
                    from_field = step_spec.get("from_field")
                    to_field = step_spec.get("to_field")
                    max_depth = step_spec.get("max_depth", 10)
                    pipeline = pipeline >> GraphClosureStep(from_field, to_field, max_depth)

                elif step_type == "graph_traverse":
                    from_field = step_spec.get("from_field")
                    to_field = step_spec.get("to_field")
                    algorithm = step_spec.get("algorithm", "bfs")
                    max_depth = step_spec.get("max_depth", 10)
                    root = step_spec.get("root")
                    pipeline = pipeline >> GraphTraverseStep(from_field, to_field, algorithm, max_depth, root)

                elif step_type == "render_mermaid":
                    pipeline = pipeline >> RenderMermaidStep.from_spec(step_spec)

                elif step_type == "pivot":
                    pipeline = pipeline >> PivotStep(
                        rows=step_spec.get("rows"),
                        cols=step_spec.get("cols"),
                        value=step_spec.get("value"),
                        aggregate=step_spec.get("aggregate", "sum"),
                    )

                elif step_type == "compute":
                    computations = step_spec.get("computations", {})
                    pipeline = pipeline >> ComputeStep(computations)

                elif step_type == "collect":
                    by_field = step_spec.get("by")
                    fields = step_spec.get("fields", {})
                    pipeline = pipeline >> CollectStep(by_field, fields)

                elif step_type == "nest":
                    pipeline = pipeline >> NestStep(
                        parent=step_spec.get("parent"),
                        child=step_spec.get("child"),
                        root=step_spec.get("root"),
                        max_depth=step_spec.get("max_depth", 10),
                        children_key=step_spec.get("children_key", "children"),
                    )

                elif step_type == "render_table":
                    pipeline = pipeline >> RenderTableStep(
                        format=step_spec.get("format", "markdown"),
                        columns=step_spec.get("columns", []),
                        title=step_spec.get("title"),
                        totals=step_spec.get("totals", False),
                        sort=step_spec.get("sort"),
                        group_by=step_spec.get("group_by"),
                        max_rows=step_spec.get("max_rows"),
                    )

                elif step_type == "render_chart":
                    pipeline = pipeline >> RenderChartStep(
                        chart_type=step_spec.get("chart_type", "bar"),
                        x=step_spec.get("x"),
                        y=step_spec.get("y"),
                        series=step_spec.get("series"),
                        title=step_spec.get("title"),
                        format=step_spec.get("format", "mermaid"),
                        colors=step_spec.get("colors"),
                        stacked=step_spec.get("stacked", False),
                        horizontal=step_spec.get("horizontal", False),
                    )

                elif step_type == "cross_join":
                    pipeline = pipeline >> CrossJoinStep(
                        unique_pairs=step_spec.get("unique_pairs", True),
                        exclude_self=step_spec.get("exclude_self", True),
                        left_prefix=step_spec.get("left_prefix", "left_"),
                        right_prefix=step_spec.get("right_prefix", "right_"),
                    )

                elif step_type == "set_similarity":
                    pipeline = pipeline >> SetSimilarityStep(
                        left_col=step_spec.get("left"),
                        right_col=step_spec.get("right"),
                        sim_type=step_spec.get("sim_type", "jaccard"),
                        output=step_spec.get("output", "similarity"),
                        intersection_output=step_spec.get("intersection_output"),
                        union_output=step_spec.get("union_output"),
                    )

                elif step_type == "string_match":
                    # Handle param reference for min_length
                    min_length = step_spec.get("min_length", 3)
                    if step_spec.get("min_length_param"):
                        min_length = params.get(step_spec["min_length_param"], min_length)
                    pipeline = pipeline >> StringMatchStep(
                        left_col=step_spec.get("left"),
                        right_col=step_spec.get("right"),
                        match_type=step_spec.get("match_type", "common_affix"),
                        min_length=min_length,
                        output=step_spec.get("output", "has_match"),
                        match_output=step_spec.get("match_output"),
                    )

                elif step_type == "rag_enrich":
                    # Handle param references
                    query_template = step_spec.get("query_template", "")
                    if step_spec.get("query_template_param"):
                        query_template = ctx.params.get(step_spec["query_template_param"], query_template)
                    top_k = step_spec.get("top_k", 1)
                    if step_spec.get("top_k_param"):
                        top_k = ctx.params.get(step_spec["top_k_param"], top_k)
                    threshold = step_spec.get("threshold")
                    if step_spec.get("threshold_param"):
                        threshold = ctx.params.get(step_spec["threshold_param"], threshold)
                    batch_size = step_spec.get("batch_size", 50)
                    if step_spec.get("batch_size_param"):
                        batch_size = ctx.params.get(step_spec["batch_size_param"], batch_size)
                    max_rows = step_spec.get("max_rows", 1000)
                    if step_spec.get("max_rows_param"):
                        max_rows = ctx.params.get(step_spec["max_rows_param"], max_rows)

                    pipeline = pipeline >> RagEnrichStep(
                        query_template=query_template,
                        top_k=top_k,
                        threshold=threshold,
                        mode=step_spec.get("mode", "best"),
                        batch_size=batch_size,
                        max_rows=max_rows,
                        entity_types=step_spec.get("entity_types"),
                    )

                else:
                    # Unknown step type - this means a step was added to the grammar
                    # but not implemented in PipelineBuilder.build()
                    raise ValueError(f"Unknown step type in pipeline: {step_type}")

            if emit_key:
                pipeline = pipeline.emit(emit_key)

            return pipeline

        return pipeline_factory


# Step classes extracted to cadsl/steps/ package — re-exported for backward compatibility
from .steps import (  # noqa: E402
    RenderTableStep, RenderChartStep,
    FlowchartConfig, SequenceConfig, ClassDiagramConfig, PieChartConfig,
    StateDiagramConfig, ERDiagramConfig, BlockBetaConfig,
    MermaidConfig, RenderMermaidStep,
    PivotStep, ComputeStep,
    JoinStep, MergeSource, CrossJoinStep,
    SetSimilarityStep, StringMatchStep,
    RagEnrichStep, CreateTaskStep,
    FetchContentStep, ViewStep, WriteFileStep, PythonStep,
)


# ============================================================
# RENDERERS
# ============================================================

def default_renderer(data: Any, format: str) -> str:
    """Default renderer - converts to string."""
    import json
    if format == "json":
        return json.dumps(data, indent=2, default=str)
    elif format == "markdown":
        if isinstance(data, list):
            return "\n".join(f"- {item}" for item in data)
        return str(data)
    return str(data)


def get_renderer(name: str) -> Callable:
    """Get a renderer by name."""
    renderers = {
        "json": lambda d, f: __import__("json").dumps(d, indent=2, default=str),
        "text": lambda d, f: str(d),
        "markdown": default_renderer,
    }
    return renderers.get(name, default_renderer)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def transform_cadsl(tree) -> List[ToolSpec]:
    """Transform a parse tree into ToolSpec objects."""
    transformer = CADSLTransformer()
    return transformer.transform(tree)


def build_pipeline(spec: ToolSpec) -> Callable:
    """Build a pipeline factory from a ToolSpec."""
    builder = PipelineBuilder()
    return builder.build(spec)
