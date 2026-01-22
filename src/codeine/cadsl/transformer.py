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
)


# ============================================================
# TOOL SPECIFICATION
# ============================================================

@dataclass
class ParamSpec:
    """Specification for a tool parameter."""
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
    """
    name: str
    tool_type: str  # "query", "detector", "diagram"
    description: str = ""
    params: List[ParamSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Pipeline components
    source_type: str = ""  # "reql", "rag_search", "rag_duplicates", "rag_clusters", "value", "merge"
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
        tool_type = self._get_tool_type(node)
        tool_name = self._get_tool_name(node)

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

    def _get_tool_type(self, node: Tree) -> str:
        """Extract tool type from tool_def node."""
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "tool_query":
                    return "query"
                elif child.data == "tool_detector":
                    return "detector"
                elif child.data == "tool_diagram":
                    return "diagram"
        return "query"

    def _get_tool_name(self, node: Tree) -> str:
        """Extract tool name from tool_def node."""
        for child in node.children:
            if isinstance(child, Token) and child.type == "NAME":
                return str(child)
        return "unnamed"

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
                    return self._unquote(str(child))
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
                        caps.append(self._unquote(str(item)))
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
                                    result["format"] = self._unquote(str(fmt))
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
        """Transform render_mermaid step."""
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
        }

        for child in node.children:
            if isinstance(child, Tree) and child.data == "mermaid_spec":
                for param in child.children:
                    if isinstance(param, Tree):
                        if param.data == "mermaid_type_spec":
                            type_node = param.children[0]
                            if isinstance(type_node, Tree):
                                result["mermaid_type"] = type_node.data.replace("mermaid_", "")
                        elif param.data == "mermaid_nodes":
                            result["nodes"] = str(param.children[0])
                        elif param.data == "mermaid_edges":
                            result["edges_from"] = str(param.children[0])
                            result["edges_to"] = str(param.children[1])
                        elif param.data == "mermaid_direction":
                            dir_node = param.children[0]
                            if isinstance(dir_node, Tree):
                                result["direction"] = dir_node.data.replace("dir_", "").upper()
                        elif param.data == "mermaid_title":
                            result["title"] = self._unquote(str(param.children[0]))
                        elif param.data == "mermaid_participants":
                            result["participants"] = str(param.children[0])
                        elif param.data == "mermaid_messages":
                            result["messages_from"] = str(param.children[0])
                            result["messages_to"] = str(param.children[1])
                            result["messages_label"] = str(param.children[2])
                        # Class diagram
                        elif param.data == "mermaid_classes":
                            result["classes"] = str(param.children[0])
                        elif param.data == "mermaid_methods":
                            result["methods"] = str(param.children[0])
                        elif param.data == "mermaid_attributes":
                            result["attributes"] = str(param.children[0])
                        elif param.data == "mermaid_inheritance":
                            result["inheritance_from"] = str(param.children[0])
                            result["inheritance_to"] = str(param.children[1])
                        elif param.data == "mermaid_composition":
                            result["composition_from"] = str(param.children[0])
                            result["composition_to"] = str(param.children[1])
                        elif param.data == "mermaid_association":
                            result["association_from"] = str(param.children[0])
                            result["association_to"] = str(param.children[1])
                        # Pie chart
                        elif param.data == "mermaid_labels":
                            result["labels"] = str(param.children[0])
                        elif param.data == "mermaid_values":
                            result["values"] = str(param.children[0])
                        # State diagram
                        elif param.data == "mermaid_states":
                            result["states"] = str(param.children[0])
                        elif param.data == "mermaid_transitions":
                            result["transitions_from"] = str(param.children[0])
                            result["transitions_to"] = str(param.children[1])
                        # ER diagram
                        elif param.data == "mermaid_entities":
                            result["entities"] = str(param.children[0])
                        elif param.data == "mermaid_relationships":
                            result["relationships"] = str(param.children[0])

        return result

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
                            result["children_key"] = self._unquote(str(param.children[0]))

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
                            result["title"] = self._unquote(str(param.children[0]))
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
                            alias = self._unquote(str(col.children[1]))
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
                            result["title"] = self._unquote(str(param.children[0]))
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
                        colors.append(self._unquote(str(item)))
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
                            result["left_prefix"] = self._unquote(str(param.children[0]))
                        elif param.data == "cj_right_prefix":
                            result["right_prefix"] = self._unquote(str(param.children[0]))

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
                            result["query_template"] = self._unquote(str(param.children[0]))
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
                                    self._unquote(str(t))
                                    for t in entity_list.children
                                    if isinstance(t, Token) and t.type == "STRING"
                                ]

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
                    return self._unquote(str(child.children[0]))
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
            return self._unquote(str(token))
        elif token.type in ("SIGNED_INT", "INT"):
            return int(str(token))
        elif token.type in ("SIGNED_FLOAT", "FLOAT"):
            return float(str(token))
        return str(token)

    def _extract_expr_as_string(self, node: Tree) -> str:
        """Extract expression as string representation."""
        if isinstance(node, Token):
            if node.type == "STRING":
                return self._unquote(str(node))
            return str(node)

        # For complex expressions, return a placeholder
        return "{expr}"

    def _unquote(self, s: str) -> str:
        """Remove quotes from a string."""
        if len(s) >= 2:
            if (s.startswith('"') and s.endswith('"')) or \
               (s.startswith("'") and s.endswith("'")):
                return s[1:-1]
        return s


# ============================================================
# PIPELINE BUILDER
# ============================================================

class PipelineBuilder:
    """
    Builds executable Pipeline objects from ToolSpec.

    This separates the AST transformation from Pipeline construction,
    allowing for different target Pipeline implementations.
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
        from codeine.dsl.core import (
            Pipeline, REQLSource, ValueSource,
            RAGSearchSource, RAGDuplicatesSource, RAGClustersSource,
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
                    pipeline = pipeline >> RenderMermaidStep(
                        mermaid_type=step_spec.get("mermaid_type", "flowchart"),
                        nodes=step_spec.get("nodes"),
                        edges_from=step_spec.get("edges_from"),
                        edges_to=step_spec.get("edges_to"),
                        direction=step_spec.get("direction", "TB"),
                        title=step_spec.get("title"),
                        participants=step_spec.get("participants"),
                        messages_from=step_spec.get("messages_from"),
                        messages_to=step_spec.get("messages_to"),
                        messages_label=step_spec.get("messages_label"),
                        # Class diagram
                        classes=step_spec.get("classes"),
                        methods=step_spec.get("methods"),
                        attributes=step_spec.get("attributes"),
                        inheritance_from=step_spec.get("inheritance_from"),
                        inheritance_to=step_spec.get("inheritance_to"),
                        composition_from=step_spec.get("composition_from"),
                        composition_to=step_spec.get("composition_to"),
                        association_from=step_spec.get("association_from"),
                        association_to=step_spec.get("association_to"),
                        # Pie chart
                        labels=step_spec.get("labels"),
                        values=step_spec.get("values"),
                        # State diagram
                        states=step_spec.get("states"),
                        transitions_from=step_spec.get("transitions_from"),
                        transitions_to=step_spec.get("transitions_to"),
                        # ER diagram
                        entities=step_spec.get("entities"),
                        relationships=step_spec.get("relationships"),
                    )

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


# ============================================================
# CONDITIONAL STEPS
# ============================================================

class WhenStep:
    """
    Conditional step - executes inner step only when condition is true.

    Syntax: when { condition } step
    """

    def __init__(self, condition, inner_step_spec):
        self.condition = condition
        self.inner_step_spec = inner_step_spec

    def execute(self, data, ctx=None):
        """Execute inner step if condition is true, otherwise pass through."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        try:
            # Evaluate condition on each row
            should_execute = True
            if callable(self.condition):
                if isinstance(data, list) and data:
                    # Use first row to check condition
                    should_execute = self.condition(data[0], ctx)
                elif isinstance(data, dict):
                    should_execute = self.condition(data, ctx)

            if should_execute and self.inner_step_spec:
                # Execute inner step
                return self._execute_inner_step(data, ctx)
            else:
                # Pass through unchanged
                return pipeline_ok(data)
        except Exception as e:
            return pipeline_err("when", f"Condition evaluation failed: {e}", e)

    def _execute_inner_step(self, data, ctx):
        """Execute the inner step spec."""
        from codeine.dsl.core import (
            pipeline_ok, pipeline_err,
            FilterStep, SelectStep, MapStep, LimitStep
        )

        spec = self.inner_step_spec
        step_type = spec.get("type")

        if step_type == "filter":
            predicate = spec.get("predicate", lambda r, c=None: True)
            step = FilterStep(predicate)
        elif step_type == "limit":
            count = spec.get("count", 100)
            step = LimitStep(count)
        elif step_type == "map":
            transform = spec.get("transform", lambda r, c=None: r)
            step = MapStep(transform)
        else:
            # Fallback - pass through
            return pipeline_ok(data)

        return step.execute(data, ctx)


class UnlessStep:
    """
    Inverted conditional step - executes inner step only when condition is false.

    Syntax: unless { condition } step
    """

    def __init__(self, condition, inner_step_spec):
        self.condition = condition
        self.inner_step_spec = inner_step_spec

    def execute(self, data, ctx=None):
        """Execute inner step if condition is false, otherwise pass through."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        try:
            # Evaluate condition on each row
            should_skip = False
            if callable(self.condition):
                if isinstance(data, list) and data:
                    should_skip = self.condition(data[0], ctx)
                elif isinstance(data, dict):
                    should_skip = self.condition(data, ctx)

            if not should_skip and self.inner_step_spec:
                # Execute inner step
                when_step = WhenStep(lambda r, c=None: True, self.inner_step_spec)
                return when_step._execute_inner_step(data, ctx)
            else:
                # Pass through unchanged
                return pipeline_ok(data)
        except Exception as e:
            return pipeline_err("unless", f"Condition evaluation failed: {e}", e)


class BranchStep:
    """
    Branching step - executes then_step if condition is true, else_step otherwise.

    Syntax: branch { condition } then step [else step]
    """

    def __init__(self, condition, then_step_spec, else_step_spec=None):
        self.condition = condition
        self.then_step_spec = then_step_spec
        self.else_step_spec = else_step_spec

    def execute(self, data, ctx=None):
        """Execute appropriate branch based on condition."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        try:
            # Evaluate condition
            should_then = True
            if callable(self.condition):
                if isinstance(data, list) and data:
                    should_then = self.condition(data[0], ctx)
                elif isinstance(data, dict):
                    should_then = self.condition(data, ctx)

            if should_then and self.then_step_spec:
                when_step = WhenStep(lambda r, c=None: True, self.then_step_spec)
                return when_step._execute_inner_step(data, ctx)
            elif not should_then and self.else_step_spec:
                when_step = WhenStep(lambda r, c=None: True, self.else_step_spec)
                return when_step._execute_inner_step(data, ctx)
            else:
                # No matching branch - pass through
                return pipeline_ok(data)
        except Exception as e:
            return pipeline_err("branch", f"Branch evaluation failed: {e}", e)


class CatchStep:
    """
    Error handling step - returns default value if previous steps failed.

    Syntax: catch { default_value }

    Note: This step wraps the pipeline execution, catching any errors
    and returning the default value instead.
    """

    def __init__(self, default_fn):
        self.default_fn = default_fn

    def execute(self, data, ctx=None):
        """Pass through data (actual error catching is done at pipeline level)."""
        from codeine.dsl.core import pipeline_ok

        # If we get here, no error occurred - just pass through
        return pipeline_ok(data)


class ParallelStep:
    """
    Execute multiple steps in parallel on the same input.

    Syntax: parallel { step1, step2, ... }

    Results from all steps are collected into a list.
    """

    def __init__(self, step_specs):
        self.step_specs = step_specs

    def execute(self, data, ctx=None):
        """Execute all steps on the same input and collect results."""
        from codeine.dsl.core import (
            pipeline_ok, pipeline_err,
            FilterStep, SelectStep, MapStep, LimitStep, AggregateStep
        )

        results = []
        errors = []

        for spec in self.step_specs:
            step_type = spec.get("type")

            try:
                if step_type == "filter":
                    predicate = spec.get("predicate", lambda r, c=None: True)
                    step = FilterStep(predicate)
                elif step_type == "select":
                    fields = spec.get("fields", {})
                    step = SelectStep(fields)
                elif step_type == "map":
                    transform = spec.get("transform", lambda r, c=None: r)
                    step = MapStep(transform)
                elif step_type == "limit":
                    count = spec.get("count", 100)
                    step = LimitStep(count)
                elif step_type == "aggregate":
                    aggs = spec.get("aggregations", {})
                    step = AggregateStep(aggs)
                else:
                    # Unknown step type, skip
                    continue

                result = step.execute(data, ctx)
                if result.is_ok():
                    results.append(result.unwrap())
                else:
                    errors.append(result)
            except Exception as e:
                errors.append(pipeline_err("parallel", f"Step failed: {e}", e))

        if errors and not results:
            return errors[0]

        return pipeline_ok(results)


class GraphCyclesStep:
    """
    Detect cycles in a directed graph.

    Syntax: graph_cycles { from: field, to: field }

    Uses DFS to detect cycles and returns a list of cycles found.
    """

    def __init__(self, from_field, to_field):
        self.from_field = from_field
        self.to_field = to_field

    def execute(self, data, ctx=None):
        """Execute cycle detection using DFS."""
        from codeine.dsl.core import pipeline_ok, pipeline_err
        from collections import defaultdict

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Build adjacency list
            graph = defaultdict(list)
            for row in data:
                from_val = row.get(self.from_field)
                to_val = row.get(self.to_field)
                if from_val and to_val:
                    graph[from_val].append(to_val)

            # DFS for cycle detection
            cycles = []
            visited = set()
            rec_stack = set()

            def dfs(node, path):
                if node in rec_stack:
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:]
                    cycles.append(tuple(cycle))
                    return
                if node in visited:
                    return

                visited.add(node)
                rec_stack.add(node)
                path.append(node)

                for neighbor in graph.get(node, []):
                    dfs(neighbor, path)

                path.pop()
                rec_stack.remove(node)

            for node in graph:
                if node not in visited:
                    dfs(node, [])

            # Convert cycles to result format
            result = [
                {"cycle": list(c), "length": len(c), "message": f"Cycle: {' -> '.join(map(str, c))} -> {c[0]}"}
                for c in cycles
            ]

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("graph_cycles", f"Cycle detection failed: {e}", e)


class GraphClosureStep:
    """
    Compute transitive closure of a directed graph.

    Syntax: graph_closure { from: field, to: field, max_depth: 10 }

    Returns all reachable nodes from each source node.
    """

    def __init__(self, from_field, to_field, max_depth=10):
        self.from_field = from_field
        self.to_field = to_field
        self.max_depth = max_depth

    def execute(self, data, ctx=None):
        """Execute transitive closure computation."""
        from codeine.dsl.core import pipeline_ok, pipeline_err
        from collections import defaultdict, deque

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Build adjacency list
            graph = defaultdict(set)
            for row in data:
                from_val = row.get(self.from_field)
                to_val = row.get(self.to_field)
                if from_val and to_val:
                    graph[from_val].add(to_val)

            # Compute closure using BFS
            result = []
            for start in graph:
                visited = set()
                queue = deque([(start, 0)])
                path = []

                while queue:
                    node, depth = queue.popleft()
                    if depth > self.max_depth:
                        continue
                    if node in visited:
                        continue

                    visited.add(node)
                    if node != start:
                        path.append(node)

                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))

                result.append({
                    "source": start,
                    "reachable": list(visited - {start}),
                    "count": len(visited) - 1,
                    "path": path[:self.max_depth],
                })

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("graph_closure", f"Transitive closure failed: {e}", e)


class GraphTraverseStep:
    """
    Traverse a directed graph using BFS or DFS.

    Syntax: graph_traverse { from: field, to: field, algorithm: bfs, max_depth: 10 }
    """

    def __init__(self, from_field, to_field, algorithm="bfs", max_depth=10, root=None):
        self.from_field = from_field
        self.to_field = to_field
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.root = root

    def execute(self, data, ctx=None):
        """Execute graph traversal and filter edges to reachable subgraph."""
        from codeine.dsl.core import pipeline_ok, pipeline_err
        from collections import defaultdict, deque
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Convert to list if Arrow table, preserve original for filtering
            original_data = data
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Build adjacency list
            graph = defaultdict(list)
            nodes = set()
            for row in data:
                from_val = row.get(self.from_field)
                to_val = row.get(self.to_field)
                if from_val and to_val:
                    graph[from_val].append(to_val)
                    nodes.add(from_val)
                    nodes.add(to_val)

            # Determine root nodes
            # BUG-002 FIX: Handle string root parameter correctly
            if self.root:
                if callable(self.root):
                    # Root is a filter function
                    roots = [n for n in nodes if self.root({"node": n}, ctx)]
                elif isinstance(self.root, str):
                    # Root is a specific node name
                    if self.root in nodes:
                        roots = [self.root]
                    elif '::' in self.root:
                        # Try partial matching for qualified names only
                        # Match by suffix (last component) or full containment
                        suffix = self.root.split('::')[-1]
                        matching = [n for n in nodes if n.endswith('::' + suffix) or n == suffix or self.root in n]
                        if matching:
                            roots = matching[:1]  # Take first match
                            logger.debug(f"Root '{self.root}' matched to '{roots[0]}'")
                        else:
                            logger.warning(f"Root node '{self.root}' not found in graph with {len(nodes)} nodes")
                            return pipeline_ok([])
                    else:
                        # Non-qualified name - require exact match only
                        logger.warning(f"Root node '{self.root}' not found in graph with {len(nodes)} nodes")
                        return pipeline_ok([])
                else:
                    roots = [self.root] if self.root in nodes else []
            else:
                # No root specified - find nodes with no incoming edges
                has_incoming = set()
                for neighbors in graph.values():
                    has_incoming.update(neighbors)
                roots = [n for n in nodes if n not in has_incoming] or list(nodes)[:1]

            if not roots:
                logger.warning("No root nodes found for graph traversal")
                return pipeline_ok([])

            # Traverse to find reachable nodes
            visited = set()

            if self.algorithm == "bfs":
                queue = deque([(r, 0) for r in roots])
                while queue:
                    node, depth = queue.popleft()
                    if depth > self.max_depth or node in visited:
                        continue
                    visited.add(node)
                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))
            else:  # DFS
                def dfs(node, depth):
                    if depth > self.max_depth or node in visited:
                        return
                    visited.add(node)
                    for neighbor in graph.get(node, []):
                        dfs(neighbor, depth + 1)

                for root in roots:
                    dfs(root, 0)

            # BUG-002 FIX: Filter original data to only edges within the visited subgraph
            # Only include edges where BOTH endpoints are visited
            # This ensures max_depth is respected (unvisited to_nodes mean depth exceeded)
            filtered = []
            for row in data:
                from_val = row.get(self.from_field)
                to_val = row.get(self.to_field)
                # Include edge only if both from and to nodes are in visited set
                if from_val in visited and to_val in visited:
                    filtered.append(row)

            return pipeline_ok(filtered)
        except Exception as e:
            return pipeline_err("graph_traverse", f"Graph traversal failed: {e}", e)


class CollectStep:
    """
    Aggregate rows by key, collecting fields into sets/lists.

    Syntax: collect { by: field, name: op(field) }

    Operations: set, list, first, last, count, sum, avg, min, max
    """

    def __init__(self, by: str, fields: dict):
        self.by = by
        self.fields = fields  # {output_name: (source_field, operation)}

    def execute(self, data, ctx=None):
        """Execute collection/aggregation."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            groups = {}
            for row in data:
                key = row.get(self.by)
                if key not in groups:
                    groups[key] = {self.by: key}
                    for name, (source, op) in self.fields.items():
                        if op in ('set', 'list'):
                            groups[key][f"_{name}_values"] = []
                        else:
                            groups[key][f"_{name}_values"] = []

                for name, (source, op) in self.fields.items():
                    value = row.get(source)
                    if value is not None:
                        groups[key][f"_{name}_values"].append(value)

            # Apply aggregation operations
            result = []
            for key, group in groups.items():
                out = {self.by: key}
                for name, (source, op) in self.fields.items():
                    values = group.get(f"_{name}_values", [])
                    if op == 'set':
                        out[name] = list(dict.fromkeys(values))  # Preserve order, remove dupes
                    elif op == 'list':
                        out[name] = values
                    elif op == 'first':
                        out[name] = values[0] if values else None
                    elif op == 'last':
                        out[name] = values[-1] if values else None
                    elif op == 'count':
                        out[name] = len(values)
                    elif op == 'sum':
                        out[name] = sum(v for v in values if isinstance(v, (int, float)))
                    elif op == 'avg':
                        nums = [v for v in values if isinstance(v, (int, float))]
                        out[name] = sum(nums) / len(nums) if nums else 0
                    elif op == 'min':
                        out[name] = min(values) if values else None
                    elif op == 'max':
                        out[name] = max(values) if values else None
                    else:
                        out[name] = values
                result.append(out)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("collect", f"Collect failed: {e}", e)


class NestStep:
    """
    Create nested/tree structure from flat data.

    Syntax: nest { parent: field, child: field, root: expr, max_depth: 10 }
    """

    def __init__(self, parent: str, child: str, root=None, max_depth=10, children_key="children"):
        self.parent = parent
        self.child = child
        self.root = root
        self.max_depth = max_depth
        self.children_key = children_key

    def execute(self, data, ctx=None):
        """Execute nesting."""
        from codeine.dsl.core import pipeline_ok, pipeline_err
        from collections import defaultdict

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Build parent->children map
            children_map = defaultdict(list)
            all_items = {}
            for row in data:
                child_id = row.get(self.child)
                parent_id = row.get(self.parent)
                if child_id:
                    all_items[child_id] = dict(row)
                    if parent_id:
                        children_map[parent_id].append(child_id)

            # Find roots
            if self.root and callable(self.root):
                roots = [cid for cid, item in all_items.items()
                        if self.root(item, ctx)]
            else:
                # Items with no parent are roots
                has_parent = set()
                for row in data:
                    parent_id = row.get(self.parent)
                    child_id = row.get(self.child)
                    if parent_id and child_id:
                        has_parent.add(child_id)
                roots = [cid for cid in all_items if cid not in has_parent]

            # Build tree recursively
            def build_tree(item_id, depth=0):
                if depth > self.max_depth or item_id not in all_items:
                    return None
                node = dict(all_items[item_id])
                child_ids = children_map.get(item_id, [])
                if child_ids:
                    node[self.children_key] = [
                        build_tree(cid, depth + 1)
                        for cid in child_ids
                        if build_tree(cid, depth + 1) is not None
                    ]
                return node

            result = [build_tree(r) for r in roots if build_tree(r) is not None]
            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("nest", f"Nest failed: {e}", e)


class RenderTableStep:
    """
    Render data as formatted table.

    Syntax: render_table { format: markdown, columns: [name, count], title: "Summary" }
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
        from codeine.dsl.core import pipeline_ok, pipeline_err

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
        from codeine.dsl.core import pipeline_ok, pipeline_err

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


class RenderMermaidStep:
    """
    Render data to a Mermaid diagram.

    Syntax: render_mermaid { type: flowchart, nodes: name, edges: from -> to }
    Supports: flowchart, sequence, class, gantt, state, er, pie
    """

    def __init__(self, mermaid_type, nodes=None, edges_from=None, edges_to=None, direction="TB",
                 title=None, participants=None, messages_from=None, messages_to=None, messages_label=None,
                 classes=None, methods=None, attributes=None, inheritance_from=None, inheritance_to=None,
                 composition_from=None, composition_to=None, association_from=None, association_to=None,
                 labels=None, values=None, states=None, transitions_from=None, transitions_to=None,
                 entities=None, relationships=None):
        self.mermaid_type = mermaid_type
        self.nodes = nodes
        self.edges_from = edges_from
        self.edges_to = edges_to
        self.direction = direction
        self.title = title
        self.participants = participants
        self.messages_from = messages_from
        self.messages_to = messages_to
        self.messages_label = messages_label
        # Class diagram
        self.classes = classes
        self.methods = methods
        self.attributes = attributes
        self.inheritance_from = inheritance_from
        self.inheritance_to = inheritance_to
        self.composition_from = composition_from
        self.composition_to = composition_to
        self.association_from = association_from
        self.association_to = association_to
        # Pie chart
        self.labels = labels
        self.values = values
        # State diagram
        self.states = states
        self.transitions_from = transitions_from
        self.transitions_to = transitions_to
        # ER diagram
        self.entities = entities
        self.relationships = relationships

    def execute(self, data, ctx=None):
        """Render to Mermaid diagram."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

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
            else:
                diagram = self._render_flowchart(data)

            return pipeline_ok({"diagram": diagram, "format": "mermaid", "type": self.mermaid_type})
        except Exception as e:
            return pipeline_err("render_mermaid", f"Mermaid rendering failed: {e}", e)

    def _render_flowchart(self, data):
        """Render a flowchart."""
        lines = [f"flowchart {self.direction}"]

        # Collect unique nodes and edges
        nodes = set()
        edges = set()
        for row in data:
            if self.edges_from and self.edges_to:
                from_val = row.get(self.edges_from)
                to_val = row.get(self.edges_to)
                if from_val and to_val:
                    nodes.add(from_val)
                    nodes.add(to_val)
                    edges.add((from_val, to_val))
            elif self.nodes:
                nodes.add(row.get(self.nodes))

        # Render edges (nodes are implicit in edges)
        for from_val, to_val in sorted(edges):
            safe_from = str(from_val).replace(" ", "_").replace("-", "_").replace(".", "_")
            safe_to = str(to_val).replace(" ", "_").replace("-", "_").replace(".", "_")
            lines.append(f'    {safe_from}["{from_val}"] --> {safe_to}["{to_val}"]')

        return "\n".join(lines)

    def _render_sequence(self, data):
        """Render a sequence diagram."""
        lines = ["sequenceDiagram"]

        # Collect participants
        participants = set()
        messages = []
        for row in data:
            if self.messages_from and self.messages_to:
                from_val = row.get(self.messages_from)
                to_val = row.get(self.messages_to)
                label = row.get(self.messages_label, "") if self.messages_label else ""
                if from_val and to_val:
                    participants.add(from_val)
                    participants.add(to_val)
                    messages.append((from_val, to_val, label))
            elif self.participants:
                participants.add(row.get(self.participants))

        # Render participants
        for p in sorted(participants):
            if p:
                safe_p = str(p).replace(" ", "_")
                lines.append(f"    participant {safe_p} as {p}")

        # Render messages
        for from_val, to_val, label in messages:
            safe_from = str(from_val).replace(" ", "_")
            safe_to = str(to_val).replace(" ", "_")
            lines.append(f"    {safe_from}->>+{safe_to}: {label}")

        return "\n".join(lines)

    def _render_class(self, data):
        """Render a class diagram with methods, attributes, and relationships."""
        lines = ["classDiagram"]

        # Aggregate class data
        class_info = {}
        inheritance_rels = set()  # parent <|-- child
        composition_rels = set()  # owner *-- owned (strong ownership)
        association_rels = set()  # from --> to (uses/references)

        for row in data:
            class_name = row.get(self.classes) if self.classes else row.get("class_name") or row.get("name")
            if not class_name:
                continue

            if class_name not in class_info:
                class_info[class_name] = {"methods": set(), "attributes": set()}

            # Collect methods
            if self.methods:
                method = row.get(self.methods)
                if method:
                    class_info[class_name]["methods"].add(method)

            # Collect attributes
            if self.attributes:
                attr = row.get(self.attributes)
                if attr:
                    class_info[class_name]["attributes"].add(attr)

            # Collect inheritance
            if self.inheritance_from and self.inheritance_to:
                parent = row.get(self.inheritance_from)
                child = row.get(self.inheritance_to)
                if parent and child:
                    inheritance_rels.add((parent, child))
            elif row.get("parent_name") or row.get("inherits_from"):
                parent = row.get("parent_name") or row.get("inherits_from")
                if parent:
                    inheritance_rels.add((parent, class_name))

            # Collect composition (owner *-- owned)
            if self.composition_from and self.composition_to:
                owner = row.get(self.composition_from)
                owned = row.get(self.composition_to)
                if owner and owned:
                    composition_rels.add((owner, owned))
            elif row.get("composition_from") and row.get("composition_to"):
                owner = row.get("composition_from")
                owned = row.get("composition_to")
                if owner and owned:
                    composition_rels.add((owner, owned))

            # Collect associations (from --> to)
            if self.association_from and self.association_to:
                from_cls = row.get(self.association_from)
                to_cls = row.get(self.association_to)
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
        lines = ["pie showData"]
        if self.title:
            lines[0] = f'pie showData title {self.title}'

        for row in data:
            label = row.get(self.labels) if self.labels else row.get("label") or row.get("name")
            value = row.get(self.values) if self.values else row.get("value") or row.get("count")
            if label and value:
                lines.append(f'    "{label}" : {value}')

        return "\n".join(lines)

    def _render_state(self, data):
        """Render a state diagram."""
        lines = ["stateDiagram-v2"]

        transitions = set()
        states = set()

        for row in data:
            if self.states:
                state = row.get(self.states)
                if state:
                    states.add(state)

            if self.transitions_from and self.transitions_to:
                from_state = row.get(self.transitions_from)
                to_state = row.get(self.transitions_to)
                if from_state and to_state:
                    transitions.add((from_state, to_state))
                    states.add(from_state)
                    states.add(to_state)

        for from_s, to_s in sorted(transitions):
            lines.append(f"    {from_s} --> {to_s}")

        return "\n".join(lines)

    def _render_er(self, data):
        """Render an ER diagram."""
        lines = ["erDiagram"]

        for row in data:
            entity = row.get(self.entities) if self.entities else row.get("entity")
            if entity:
                lines.append(f"    {entity}")

        return "\n".join(lines)


class PivotStep:
    """
    Create a pivot table from data.

    Syntax: pivot { rows: field, cols: field, value: field, aggregate: sum }
    """

    def __init__(self, rows, cols, value, aggregate="sum"):
        self.rows = rows
        self.cols = cols
        self.value = value
        self.aggregate = aggregate

    def execute(self, data, ctx=None):
        """Execute pivot table creation."""
        from codeine.dsl.core import pipeline_ok, pipeline_err
        from collections import defaultdict

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            # Collect values by (row, col)
            pivot_data = defaultdict(list)
            all_cols = set()
            all_rows = set()

            for row in data:
                row_key = row.get(self.rows)
                col_key = row.get(self.cols)
                val = row.get(self.value, 0)

                if row_key is not None and col_key is not None:
                    pivot_data[(row_key, col_key)].append(val if val is not None else 0)
                    all_rows.add(row_key)
                    all_cols.add(col_key)

            # Aggregate
            def aggregate_values(values):
                if not values:
                    return 0
                if self.aggregate == "sum":
                    return sum(values)
                elif self.aggregate == "avg":
                    return sum(values) / len(values)
                elif self.aggregate == "count":
                    return len(values)
                elif self.aggregate == "min":
                    return min(values)
                elif self.aggregate == "max":
                    return max(values)
                elif self.aggregate == "first":
                    return values[0]
                elif self.aggregate == "last":
                    return values[-1]
                return sum(values)

            # Build result table
            result = []
            sorted_cols = sorted(all_cols, key=str)
            for row_key in sorted(all_rows, key=str):
                row_result = {self.rows: row_key}
                for col_key in sorted_cols:
                    values = pivot_data.get((row_key, col_key), [])
                    row_result[str(col_key)] = aggregate_values(values)
                result.append(row_result)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("pivot", f"Pivot failed: {e}", e)


class ComputeStep:
    """
    Compute new fields using expressions.

    Syntax: compute { ratio: a / b, pct: ratio * 100 }
    """

    def __init__(self, computations):
        self.computations = computations

    def execute(self, data, ctx=None):
        """Execute field computation."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            result = []
            for row in data:
                new_row = dict(row)
                # Compute each field in order (so later fields can reference earlier ones)
                for name, expr in self.computations.items():
                    try:
                        new_row[name] = expr(new_row, ctx)
                    except Exception:
                        new_row[name] = None
                result.append(new_row)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("compute", f"Compute failed: {e}", e)


class JoinStep:
    """
    Join step - joins pipeline data with another source using PyArrow.

    Syntax: join { left: key, right: source, right_key: key, type: inner }

    Supports all PyArrow join types: inner, left, right, outer, semi, anti.
    """

    def __init__(self, left_key, right_source_spec, right_key, join_type="inner"):
        self.left_key = left_key
        self.right_source_spec = right_source_spec
        self.right_key = right_key
        self.join_type = join_type

    def execute(self, data, ctx=None):
        """Execute join using PyArrow."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        try:
            import pyarrow as pa

            # Convert left data to Arrow table
            if isinstance(data, pa.Table):
                left_table = data
            elif isinstance(data, list):
                if not data:
                    return pipeline_ok([])
                # Convert list of dicts to Arrow table
                left_table = pa.Table.from_pylist(data)
            else:
                return pipeline_err("join", "Left data must be a list or Arrow table")

            # Execute right source
            right_data = self._execute_right_source(ctx)
            if isinstance(right_data, list):
                if not right_data:
                    # Right side empty - return empty for inner, left data for left/outer
                    if self.join_type in ("inner", "semi"):
                        return pipeline_ok([])
                    elif self.join_type in ("left", "outer"):
                        return pipeline_ok(data if isinstance(data, list) else data.to_pylist())
                    elif self.join_type == "anti":
                        return pipeline_ok(data if isinstance(data, list) else data.to_pylist())
                    return pipeline_ok([])
                right_table = pa.Table.from_pylist(right_data)
            elif isinstance(right_data, pa.Table):
                right_table = right_data
            else:
                return pipeline_err("join", "Right source must return a list or Arrow table")

            # Resolve column names (handle ? prefix)
            left_col = self._resolve_column(left_table, self.left_key)
            right_col = self._resolve_column(right_table, self.right_key)

            if left_col is None:
                return pipeline_err("join", f"Left key column not found: {self.left_key}")
            if right_col is None:
                return pipeline_err("join", f"Right key column not found: {self.right_key}")

            # Perform join
            joined = left_table.join(
                right_table,
                keys=left_col,
                right_keys=right_col,
                join_type=self.join_type
            )

            # PyArrow join drops the right key column (since it equals left key).
            # If the right key is different from left key name, add it back.
            if right_col != left_col and right_col not in joined.column_names:
                # Add right key column as copy of left key column
                left_col_data = joined.column(left_col)
                joined = joined.append_column(right_col, left_col_data)

            # Convert back to list of dicts
            return pipeline_ok(joined.to_pylist())
        except Exception as e:
            return pipeline_err("join", f"Join failed: {e}", e)

    def _resolve_column(self, table, col_name):
        """Resolve column name, handling ? prefix."""
        if col_name in table.column_names:
            return col_name
        if f"?{col_name}" in table.column_names:
            return f"?{col_name}"
        return None

    def _execute_right_source(self, ctx):
        """Execute the right source and return data."""
        from codeine.dsl.core import (
            REQLSource, ValueSource,
            RAGSearchSource, RAGDuplicatesSource, RAGClustersSource
        )

        spec = self.right_source_spec
        if spec is None:
            return []

        source_type = spec.get("type")

        # Helper to resolve parameter placeholders like "{similarity}" to actual values
        def resolve_param(value, default=None):
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                param_name = value[1:-1]
                if ctx and hasattr(ctx, 'params') and param_name in ctx.params:
                    return ctx.params[param_name]
                return default
            return value if value is not None else default

        if source_type == "reql":
            source = REQLSource(spec.get("content", ""))
        elif source_type == "rag_search":
            params = spec.get("params", {})
            source = RAGSearchSource(
                query=resolve_param(params.get("query"), ""),
                top_k=resolve_param(params.get("top_k"), 10),
                entity_types=resolve_param(params.get("entity_types")),
            )
        elif source_type == "rag_duplicates":
            params = spec.get("params", {})
            source = RAGDuplicatesSource(
                similarity=resolve_param(params.get("similarity"), 0.85),
                limit=resolve_param(params.get("limit"), 50),
                exclude_same_file=resolve_param(params.get("exclude_same_file"), True),
                exclude_same_class=resolve_param(params.get("exclude_same_class"), True),
                entity_types=resolve_param(params.get("entity_types")),
            )
        elif source_type == "rag_clusters":
            params = spec.get("params", {})
            source = RAGClustersSource(
                n_clusters=resolve_param(params.get("n_clusters"), 50),
                min_size=resolve_param(params.get("min_size"), 2),
                exclude_same_file=resolve_param(params.get("exclude_same_file"), True),
                exclude_same_class=resolve_param(params.get("exclude_same_class"), True),
                entity_types=resolve_param(params.get("entity_types")),
            )
        elif source_type == "value":
            source = ValueSource(spec.get("content", []))
        else:
            return []

        result = source.execute(ctx)
        if result.is_ok():
            return result.unwrap()
        return []


class MergeSource:
    """
    Merge multiple sources into one.

    Syntax: merge { source1, source2, ... }
    Or with per-source steps: merge { source1 | step1 | step2, source2 | step3 }

    Executes all sources (with their steps) and concatenates their results.
    """

    def __init__(self, source_specs):
        self.source_specs = source_specs

    def execute(self, ctx=None):
        """Execute all sources and merge results."""
        from codeine.dsl.core import (
            pipeline_ok, pipeline_err,
            REQLSource, ValueSource,
            RAGSearchSource, RAGDuplicatesSource, RAGClustersSource,
            Pipeline, SelectStep, MapStep, FilterStep
        )

        merged = []
        errors = []

        for spec in self.source_specs:
            source_type = spec.get("type")

            try:
                if source_type == "reql":
                    source = REQLSource(spec.get("content", ""))
                elif source_type == "rag_search":
                    params = spec.get("params", {})
                    source = RAGSearchSource(
                        query=params.get("query", ""),
                        top_k=params.get("top_k", 10),
                        entity_types=params.get("entity_types"),
                    )
                elif source_type == "rag_duplicates":
                    params = spec.get("params", {})
                    source = RAGDuplicatesSource(
                        similarity=params.get("similarity", 0.85),
                        limit=params.get("limit", 50),
                        exclude_same_file=params.get("exclude_same_file", True),
                        exclude_same_class=params.get("exclude_same_class", True),
                        entity_types=params.get("entity_types"),
                    )
                elif source_type == "rag_clusters":
                    params = spec.get("params", {})
                    source = RAGClustersSource(
                        n_clusters=params.get("n_clusters", 50),
                        min_size=params.get("min_size", 2),
                        exclude_same_file=params.get("exclude_same_file", True),
                        exclude_same_class=params.get("exclude_same_class", True),
                        entity_types=params.get("entity_types"),
                    )
                elif source_type == "value":
                    source = ValueSource(spec.get("content", []))
                else:
                    continue

                # Execute source
                result = source.execute(ctx)
                if result.is_err():
                    errors.append(result)
                    continue

                data = result.unwrap()

                # Apply per-source steps if any
                steps = spec.get("steps", [])
                if steps:
                    data = self._apply_steps(data, steps, ctx)

                # Merge into results
                if isinstance(data, list):
                    merged.extend(data)
                elif hasattr(data, 'to_pylist'):  # PyArrow table
                    merged.extend(data.to_pylist())
                else:
                    merged.append(data)

            except Exception as e:
                errors.append(pipeline_err("merge", f"Source failed: {e}", e))

        if errors and not merged:
            return errors[0]

        return pipeline_ok(merged)

    def _apply_steps(self, data, steps, ctx):
        """Apply pipeline steps to data."""
        from codeine.dsl.core import SelectStep, MapStep, FilterStep, pipeline_ok

        # Convert Arrow table to list if needed
        if hasattr(data, 'to_pylist'):
            data = data.to_pylist()

        for step_spec in steps:
            step_type = step_spec.get("type")

            if step_type == "select":
                fields = step_spec.get("fields", {})
                step = SelectStep(fields)
                result = step.execute(data, ctx)
                if result.is_ok():
                    data = result.unwrap()

            elif step_type == "map":
                transform = step_spec.get("transform", lambda r, ctx=None: r)
                step = MapStep(transform)
                result = step.execute(data, ctx)
                if result.is_ok():
                    data = result.unwrap()

            elif step_type == "filter":
                predicate = step_spec.get("predicate", lambda r, ctx=None: True)
                step = FilterStep(predicate)
                result = step.execute(data, ctx)
                if result.is_ok():
                    data = result.unwrap()

        # Convert back to list if it's Arrow
        if hasattr(data, 'to_pylist'):
            data = data.to_pylist()

        return data


# ============================================================
# N² COMPARISON STEPS
# ============================================================

class CrossJoinStep:
    """
    Cross join (Cartesian product) step for N² pairwise comparison.

    Syntax: cross_join { unique_pairs: true, exclude_self: true, left_prefix: "left_", right_prefix: "right_" }

    Creates all pairs from input rows. With unique_pairs=true, generates (n*(n-1))/2 pairs.
    Uses PyArrow for efficient vectorized operations.
    """

    def __init__(self, unique_pairs=True, exclude_self=True, left_prefix="left_", right_prefix="right_"):
        self.unique_pairs = unique_pairs
        self.exclude_self = exclude_self
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix

    def execute(self, data, ctx=None):
        """Execute cross join using PyArrow."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        try:
            import numpy as np
            import pyarrow as pa
            import pyarrow.compute as pc

            # Convert to Arrow table
            if isinstance(data, pa.Table):
                table = data
            elif isinstance(data, list):
                if not data:
                    return pipeline_ok([])
                table = pa.Table.from_pylist(data)
            else:
                return pipeline_err("cross_join", "Input must be a list or Arrow table")

            n = table.num_rows
            if n < 2:
                return pipeline_ok([])

            # Create index arrays
            idx = np.arange(n, dtype=np.int64)

            if self.unique_pairs:
                # Generate (i, j) pairs where i < j
                i_grid, j_grid = np.meshgrid(idx, idx, indexing='ij')
                mask = i_grid < j_grid
                left_idx = pa.array(i_grid[mask])
                right_idx = pa.array(j_grid[mask])
            else:
                # Full Cartesian product
                i_grid, j_grid = np.meshgrid(idx, idx, indexing='ij')
                if self.exclude_self:
                    mask = i_grid != j_grid
                    left_idx = pa.array(i_grid[mask])
                    right_idx = pa.array(j_grid[mask])
                else:
                    left_idx = pa.array(i_grid.ravel())
                    right_idx = pa.array(j_grid.ravel())

            # Build result table with prefixed columns
            result = {}
            for col in table.column_names:
                result[f'{self.left_prefix}{col}'] = pc.take(table.column(col), left_idx)
                result[f'{self.right_prefix}{col}'] = pc.take(table.column(col), right_idx)

            result_table = pa.table(result)
            return pipeline_ok(result_table.to_pylist())

        except ImportError:
            return pipeline_err("cross_join", "PyArrow and NumPy are required for cross_join")
        except Exception as e:
            return pipeline_err("cross_join", f"Cross join failed: {e}", e)


class SetSimilarityStep:
    """
    Compute set similarity between two columns.

    Syntax: set_similarity { left: col1, right: col2, type: jaccard, output: similarity }

    Types:
    - jaccard: |intersection| / |union|
    - dice: 2 * |intersection| / (|A| + |B|)
    - overlap: |intersection| / min(|A|, |B|)
    - cosine: |intersection| / sqrt(|A| * |B|)
    """

    def __init__(self, left_col, right_col, sim_type="jaccard", output="similarity",
                 intersection_output=None, union_output=None):
        self.left_col = left_col
        self.right_col = right_col
        self.sim_type = sim_type
        self.output = output
        self.intersection_output = intersection_output
        self.union_output = union_output

    def execute(self, data, ctx=None):
        """Calculate set similarity."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok([])

            result = []
            for row in data:
                new_row = dict(row)
                left = row.get(self.left_col) or []
                right = row.get(self.right_col) or []

                # Convert to sets
                left_set = set(left) if isinstance(left, (list, tuple, set)) else {left}
                right_set = set(right) if isinstance(right, (list, tuple, set)) else {right}

                intersection = left_set & right_set
                union = left_set | right_set

                # Calculate similarity
                if self.sim_type == "jaccard":
                    similarity = len(intersection) / len(union) if union else 0
                elif self.sim_type == "dice":
                    total = len(left_set) + len(right_set)
                    similarity = 2 * len(intersection) / total if total else 0
                elif self.sim_type == "overlap":
                    min_size = min(len(left_set), len(right_set))
                    similarity = len(intersection) / min_size if min_size else 0
                elif self.sim_type == "cosine":
                    denom = (len(left_set) * len(right_set)) ** 0.5
                    similarity = len(intersection) / denom if denom else 0
                else:
                    similarity = len(intersection) / len(union) if union else 0

                new_row[self.output] = round(similarity, 4)

                if self.intersection_output:
                    new_row[self.intersection_output] = list(intersection)
                if self.union_output:
                    new_row[self.union_output] = list(union)

                result.append(new_row)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("set_similarity", f"Set similarity failed: {e}", e)


class StringMatchStep:
    """
    Detect string pattern matches between two columns.

    Syntax: string_match { left: col1, right: col2, type: common_affix, min_length: 3, output: has_match }

    Types:
    - common_affix: Check for common prefix OR suffix
    - common_prefix: Check for common prefix only
    - common_suffix: Check for common suffix only
    - levenshtein: Calculate edit distance (requires output_distance)
    - contains: Check if one contains the other
    """

    def __init__(self, left_col, right_col, match_type="common_affix", min_length=3,
                 output="has_match", match_output=None):
        self.left_col = left_col
        self.right_col = right_col
        self.match_type = match_type
        self.min_length = min_length
        self.output = output
        self.match_output = match_output

    def execute(self, data, ctx=None):
        """Execute string matching."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok([])

            result = []
            for row in data:
                new_row = dict(row)
                left = str(row.get(self.left_col, ""))
                right = str(row.get(self.right_col, ""))

                has_match = False
                match_value = None

                if self.match_type in ("common_affix", "common_prefix", "common_suffix"):
                    # Check prefix
                    if self.match_type in ("common_affix", "common_prefix"):
                        min_len = min(len(left), len(right))
                        for plen in range(self.min_length, min_len + 1):
                            if left[:plen] == right[:plen]:
                                # Found common prefix, check if different suffixes
                                s1, s2 = left[plen:], right[plen:]
                                if s1 and s2 and s1 != s2:
                                    has_match = True
                                    match_value = f"prefix:{left[:plen]}"
                                    break

                    # Check suffix
                    if not has_match and self.match_type in ("common_affix", "common_suffix"):
                        min_len = min(len(left), len(right))
                        for slen in range(self.min_length, min_len + 1):
                            if left[-slen:] == right[-slen:]:
                                # Found common suffix, check if different prefixes
                                p1, p2 = left[:-slen], right[:-slen]
                                if p1 and p2 and p1 != p2:
                                    has_match = True
                                    match_value = f"suffix:{left[-slen:]}"
                                    break

                elif self.match_type == "contains":
                    if left in right:
                        has_match = True
                        match_value = f"left_in_right:{left}"
                    elif right in left:
                        has_match = True
                        match_value = f"right_in_left:{right}"

                elif self.match_type == "levenshtein":
                    # Simple Levenshtein distance
                    distance = self._levenshtein(left, right)
                    has_match = distance <= self.min_length
                    match_value = distance

                new_row[self.output] = has_match
                if self.match_output and match_value is not None:
                    new_row[self.match_output] = match_value

                result.append(new_row)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("string_match", f"String match failed: {e}", e)

    def _levenshtein(self, s1, s2):
        """Calculate Levenshtein distance."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]


# ============================================================
# RAG ENRICH STEP
# ============================================================

class RagEnrichStep:
    """
    Per-row RAG enrichment step - enriches each row with semantic search results.

    Syntax: rag_enrich { query: "template {field}", top_k: 3, threshold: 0.5, mode: "best" }

    Template placeholders like {field} are replaced with row values before search.

    Modes:
    - "best": Adds best match fields directly to row (similarity, similar_entity, similar_file)
    - "all": Adds array of all matches as rag_matches field

    Uses batching for performance optimization.
    """

    def __init__(self, query_template, top_k=1, threshold=None, mode="best",
                 batch_size=50, max_rows=1000, entity_types=None):
        self.query_template = query_template
        self.top_k = top_k
        self.threshold = threshold
        self.mode = mode
        self.batch_size = batch_size
        self.max_rows = max_rows
        self.entity_types = entity_types

    def execute(self, data, ctx=None):
        """Execute RAG enrichment with batching."""
        from codeine.dsl.core import pipeline_ok, pipeline_err
        import logging
        import re

        logger = logging.getLogger(__name__)

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok([])

            # Check row count limit
            if len(data) > self.max_rows:
                logger.warning(
                    f"RAG enrichment: {len(data)} rows exceeds max_rows ({self.max_rows}). "
                    f"Processing first {self.max_rows} rows only."
                )
                data = data[:self.max_rows]

            # Validate template fields against first row
            template_fields = re.findall(r'\{(\w+)\}', self.query_template)
            if data and template_fields:
                missing = [f for f in template_fields if f not in data[0]]
                if missing:
                    # Check for ?-prefixed versions (REQL output)
                    still_missing = []
                    for f in missing:
                        if f"?{f}" not in data[0]:
                            still_missing.append(f)
                    if still_missing:
                        return pipeline_err(
                            "rag_enrich",
                            f"Template field(s) not found in row: {still_missing}. "
                            f"Available fields: {list(data[0].keys())}"
                        )

            # Get RAG manager from context or default instance (same pattern as RAGSearchSource)
            rag_manager = None
            if ctx and hasattr(ctx, 'get'):
                rag_manager = ctx.get("rag_manager")
            if rag_manager is None:
                try:
                    from codeine.services.default_instance_manager import DefaultInstanceManager
                    default_mgr = DefaultInstanceManager.get_instance()
                    if default_mgr:
                        rag_manager = default_mgr.get_rag_manager()
                except Exception as e:
                    logger.debug(f"Could not get RAG manager: {e}")

            if rag_manager is None:
                return pipeline_err(
                    "rag_enrich",
                    "RAG manager not available. Ensure project is initialized."
                )

            # Process in batches for performance
            result = []
            for batch_start in range(0, len(data), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(data))
                batch = data[batch_start:batch_end]

                # Prepare queries for batch
                queries = []
                for row in batch:
                    query = self._expand_template(row)
                    queries.append(query)

                # Execute batch search
                batch_results = self._batch_search(rag_manager, queries, ctx)

                # Enrich rows with results
                for i, row in enumerate(batch):
                    new_row = dict(row)
                    matches = batch_results[i] if i < len(batch_results) else []

                    # Ensure matches is always a list
                    if matches is None:
                        matches = []

                    # Filter by threshold (RAG results use 'score' field)
                    if self.threshold is not None and matches:
                        matches = [m for m in matches if m.get('score', m.get('similarity', 0)) >= self.threshold]

                    if self.mode == "best":
                        # Add best match fields directly
                        if matches:
                            best = matches[0]
                            # RAG results use 'score', fallback to 'similarity' for compatibility
                            new_row['similarity'] = best.get('score', best.get('similarity', 0))
                            new_row['similar_entity'] = best.get('name', best.get('entity', ''))
                            new_row['similar_file'] = best.get('file', '')
                            new_row['similar_line'] = best.get('line', 0)
                            new_row['similar_type'] = best.get('entity_type', '')
                        else:
                            new_row['similarity'] = 0
                            new_row['similar_entity'] = None
                            new_row['similar_file'] = None
                            new_row['similar_line'] = None
                            new_row['similar_type'] = None
                    else:  # mode == "all"
                        new_row['rag_matches'] = matches

                    result.append(new_row)

            return pipeline_ok(result)

        except Exception as e:
            import traceback
            logger.error(f"RAG enrichment failed: {e}\n{traceback.format_exc()}")
            return pipeline_err("rag_enrich", f"RAG enrichment failed: {e}", e)

    def _expand_template(self, row):
        """Expand template placeholders with row values."""
        import re

        def replacer(match):
            field = match.group(1)
            # Try exact field name first
            if field in row:
                return str(row[field])
            # Try ?-prefixed version (REQL output)
            if f"?{field}" in row:
                return str(row[f"?{field}"])
            return match.group(0)  # Keep original if not found

        return re.sub(r'\{(\w+)\}', replacer, self.query_template)

    def _batch_search(self, rag_manager, queries, ctx):
        """Execute batch RAG search. Returns list of result lists."""
        results = []

        for query in queries:
            try:
                # Use RAG manager's search method (returns (results, stats))
                if hasattr(rag_manager, 'search'):
                    search_results, stats = rag_manager.search(
                        query=query,
                        top_k=self.top_k,
                        entity_types=self.entity_types
                    )

                    # Check for errors
                    if stats.get("error"):
                        results.append([])
                        continue

                    # Convert RAGSearchResult objects to dicts
                    matches = []
                    for r in search_results:
                        if hasattr(r, 'to_dict'):
                            matches.append(r.to_dict())
                        elif isinstance(r, dict):
                            matches.append(r)
                        elif hasattr(r, '__dict__'):
                            matches.append(vars(r))
                        else:
                            matches.append({'entity': str(r), 'similarity': 0})
                    results.append(matches)
                else:
                    results.append([])
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"RAG search failed for query '{query[:50]}...': {e}")
                results.append([])

        return results


# ============================================================
# PYTHON STEP
# ============================================================

class PythonStep:
    """
    Executes inline Python code as a pipeline step.

    Available in the Python block:
    - rows: Input data from previous step
    - ctx: Execution context with params
    - result: Must be set to the output value
    """

    def __init__(self, code: str):
        self.code = code
        try:
            self.compiled = compile(code, "<cadsl_python>", "exec")
        except SyntaxError as e:
            self.compiled = None
            self.error = str(e)

    def execute(self, data, ctx=None):
        """Execute the Python code."""
        from codeine.dsl.core import pipeline_ok, pipeline_err

        if self.compiled is None:
            return pipeline_err("python", f"Python syntax error: {self.error}")

        # Create execution namespace
        namespace = {
            "rows": data,
            "ctx": ctx,
            "result": None,
            # Common imports
            "defaultdict": __import__("collections").defaultdict,
            "Counter": __import__("collections").Counter,
            "re": __import__("re"),
            "json": __import__("json"),
            "math": __import__("math"),
        }

        try:
            exec(self.compiled, namespace)

            if namespace.get("result") is None:
                # If no result set, use rows
                return pipeline_ok(data)

            return pipeline_ok(namespace["result"])
        except Exception as e:
            return pipeline_err("python", f"Python execution error: {e}", e)


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
