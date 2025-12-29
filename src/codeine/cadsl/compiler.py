"""
CADSL Compiler - Expression and Condition Compiler.

This module compiles CADSL expressions, conditions, and object expressions
into Python callables that can be used in Pipeline steps.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from lark import Tree, Token


# ============================================================
# BUILT-IN FUNCTIONS
# ============================================================

BUILTINS: Dict[str, Callable] = {
    # String functions
    "len": len,
    "str": str,
    "lower": lambda s: s.lower() if isinstance(s, str) else str(s).lower(),
    "upper": lambda s: s.upper() if isinstance(s, str) else str(s).upper(),
    "strip": lambda s: s.strip() if isinstance(s, str) else str(s).strip(),
    "split": lambda s, sep=None: s.split(sep) if isinstance(s, str) else [],
    "join": lambda sep, lst: sep.join(str(x) for x in lst),
    "replace": lambda s, old, new: s.replace(old, new) if isinstance(s, str) else s,

    # Math functions
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,

    # Type checks
    "is_null": lambda x: x is None,
    "is_empty": lambda x: not x,
    "is_list": lambda x: isinstance(x, list),
    "is_str": lambda x: isinstance(x, str),
    "is_int": lambda x: isinstance(x, int) and not isinstance(x, bool),
    "is_bool": lambda x: isinstance(x, bool),

    # List functions
    "first": lambda lst: lst[0] if lst else None,
    "last": lambda lst: lst[-1] if lst else None,
    "count": lambda lst: len(lst) if isinstance(lst, (list, tuple)) else 0,
    "sorted": sorted,
    "reversed": lambda lst: list(reversed(lst)),
    "unique": lambda lst: list(dict.fromkeys(lst)),

    # Utility
    "default": lambda val, default: val if val is not None else default,
    "coalesce": lambda *args: next((a for a in args if a is not None), None),
}


# ============================================================
# EXPRESSION COMPILER
# ============================================================

class ExpressionCompiler:
    """
    Compiles CADSL expressions to Python callables.

    Expressions in CADSL can reference:
    - Field names (from row data)
    - Parameters (from context)
    - Literals (strings, numbers, booleans, null)
    - Function calls
    - Arithmetic operations
    """

    def __init__(self):
        self.builtins = BUILTINS.copy()

    def compile(self, node: Union[Tree, Token]) -> Callable[[Dict, Optional[Any]], Any]:
        """
        Compile an expression node to a callable.

        Args:
            node: Lark tree node representing an expression

        Returns:
            Callable that takes (row, ctx) and returns a value
        """
        if isinstance(node, Token):
            return self._compile_token(node)

        # Handle tree nodes by data type
        method_name = f"_compile_{node.data}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)

        # Default: try to compile children
        if node.children:
            return self.compile(node.children[0])

        # Empty node
        return lambda r, ctx=None: None

    def _compile_token(self, token: Token) -> Callable[[Dict, Optional[Any]], Any]:
        """Compile a token to a value extractor."""
        if token.type == "NAME":
            field = str(token)
            # Field reference
            return lambda r, ctx=None, f=field: r.get(f) if isinstance(r, dict) else None

        elif token.type == "STRING":
            value = self._unquote(str(token))
            return lambda r, ctx=None, v=value: v

        elif token.type in ("SIGNED_INT", "INT"):
            value = int(str(token))
            return lambda r, ctx=None, v=value: v

        elif token.type in ("SIGNED_FLOAT", "FLOAT"):
            value = float(str(token))
            return lambda r, ctx=None, v=value: v

        else:
            value = str(token)
            return lambda r, ctx=None, v=value: v

    # --------------------------------------------------------
    # Literal Values
    # --------------------------------------------------------

    def _compile_val_string(self, node: Tree) -> Callable:
        value = self._unquote(str(node.children[0]))
        # Check if string contains parameter placeholders like {param}
        if '{' in value and '}' in value:
            import re
            placeholders = re.findall(r'\{(\w+)\}', value)
            if placeholders:
                def substitute(r, ctx=None, v=value, phs=placeholders):
                    result = v
                    if ctx and hasattr(ctx, 'params'):
                        for ph in phs:
                            if ph in ctx.params:
                                result = result.replace('{' + ph + '}', str(ctx.params[ph]))
                    return result
                return substitute
        return lambda r, ctx=None, v=value: v

    def _compile_val_int(self, node: Tree) -> Callable:
        value = int(str(node.children[0]))
        return lambda r, ctx=None, v=value: v

    def _compile_val_float(self, node: Tree) -> Callable:
        value = float(str(node.children[0]))
        return lambda r, ctx=None, v=value: v

    def _compile_val_true(self, node: Tree) -> Callable:
        return lambda r, ctx=None: True

    def _compile_val_false(self, node: Tree) -> Callable:
        return lambda r, ctx=None: False

    def _compile_val_null(self, node: Tree) -> Callable:
        return lambda r, ctx=None: None

    def _compile_val_list(self, node: Tree) -> Callable:
        items = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "value_list":
                for item in child.children:
                    items.append(self.compile(item))
        return lambda r, ctx=None, i=items: [fn(r, ctx) for fn in i]

    def _compile_literal(self, node: Tree) -> Callable:
        """Compile a literal value."""
        return self.compile(node.children[0])

    # --------------------------------------------------------
    # References
    # --------------------------------------------------------

    def _compile_field_ref(self, node: Tree) -> Callable:
        """Compile a field reference."""
        field = str(node.children[0])
        return lambda r, ctx=None, f=field: r.get(f) if isinstance(r, dict) else None

    def _compile_param_ref(self, node: Tree) -> Callable:
        """Compile a parameter reference {param_name}."""
        param = str(node.children[0])
        return lambda r, ctx=None, p=param: (
            ctx.params.get(p) if ctx and hasattr(ctx, 'params') else None
        )

    def _compile_param_ref_expr(self, node: Tree) -> Callable:
        """Compile a parameter reference expression."""
        return self._compile_param_ref(node.children[0])

    # --------------------------------------------------------
    # Arithmetic Operations
    # --------------------------------------------------------

    def _compile_add_expr(self, node: Tree) -> Callable:
        """Compile addition/subtraction."""
        if len(node.children) == 1:
            return self.compile(node.children[0])

        result = self.compile(node.children[0])
        i = 1
        while i < len(node.children):
            op_node = node.children[i]
            right = self.compile(node.children[i + 1])
            if isinstance(op_node, Tree) and op_node.data == "op_add":
                prev = result
                result = lambda r, ctx=None, l=prev, ri=right: (l(r, ctx) or 0) + (ri(r, ctx) or 0)
            elif isinstance(op_node, Tree) and op_node.data == "op_sub":
                prev = result
                result = lambda r, ctx=None, l=prev, ri=right: (l(r, ctx) or 0) - (ri(r, ctx) or 0)
            i += 2
        return result

    def _compile_mul_expr(self, node: Tree) -> Callable:
        """Compile multiplication/division."""
        if len(node.children) == 1:
            return self.compile(node.children[0])

        result = self.compile(node.children[0])
        i = 1
        while i < len(node.children):
            op_node = node.children[i]
            right = self.compile(node.children[i + 1])
            if isinstance(op_node, Tree) and op_node.data == "op_mul":
                prev = result
                result = lambda r, ctx=None, l=prev, ri=right: (l(r, ctx) or 0) * (ri(r, ctx) or 0)
            elif isinstance(op_node, Tree) and op_node.data == "op_div":
                prev = result
                result = lambda r, ctx=None, l=prev, ri=right: (l(r, ctx) or 0) / (ri(r, ctx) or 1)
            elif isinstance(op_node, Tree) and op_node.data == "op_mod":
                prev = result
                result = lambda r, ctx=None, l=prev, ri=right: (l(r, ctx) or 0) % (ri(r, ctx) or 1)
            i += 2
        return result

    def _compile_neg_expr(self, node: Tree) -> Callable:
        """Compile negation."""
        inner = self.compile(node.children[0])
        return lambda r, ctx=None, i=inner: -(i(r, ctx) or 0)

    def _compile_pos_expr(self, node: Tree) -> Callable:
        """Compile positive (no-op)."""
        return self.compile(node.children[0])

    def _compile_paren_expr(self, node: Tree) -> Callable:
        """Compile parenthesized expression."""
        return self.compile(node.children[0])

    def _compile_ternary(self, node: Tree) -> Callable:
        """Compile ternary expression: condition ? then_value : else_value"""
        cond = self.compile(node.children[0])
        then_val = self.compile(node.children[1])
        else_val = self.compile(node.children[2])
        return lambda r, ctx=None, c=cond, t=then_val, e=else_val: (
            t(r, ctx) if c(r, ctx) else e(r, ctx)
        )

    def _compile_coalesce(self, node: Tree) -> Callable:
        """Compile coalesce expression: value ?? default"""
        left = self.compile(node.children[0])
        right = self.compile(node.children[1])
        return lambda r, ctx=None, l=left, ri=right: (
            l(r, ctx) if l(r, ctx) is not None else ri(r, ctx)
        )

    # --------------------------------------------------------
    # Property Access and Function Calls
    # --------------------------------------------------------

    def _compile_prop_access(self, node: Tree) -> Callable:
        """Compile property access: obj.prop"""
        obj = self.compile(node.children[0])
        prop = str(node.children[1])

        def access(r, ctx=None, o=obj, p=prop):
            value = o(r, ctx)
            if isinstance(value, dict):
                return value.get(p)
            elif hasattr(value, p):
                return getattr(value, p)
            return None

        return access

    def _compile_func_call(self, node: Tree) -> Callable:
        """Compile function call: func(args...)"""
        func_name = str(node.children[0])

        # Collect arguments
        args = []
        for child in node.children[1:]:
            if isinstance(child, Tree) and child.data == "arg_list":
                for arg in child.children:
                    args.append(self.compile(arg))
            elif isinstance(child, Tree):
                args.append(self.compile(child))

        # Check builtin functions
        if func_name in self.builtins:
            fn = self.builtins[func_name]
            return lambda r, ctx=None, f=fn, a=args: f(*[arg(r, ctx) for arg in a])

        # Unknown function - return None
        return lambda r, ctx=None: None

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _unquote(self, s: str) -> str:
        """Remove quotes from a string."""
        if len(s) >= 2:
            if (s.startswith('"') and s.endswith('"')) or \
               (s.startswith("'") and s.endswith("'")):
                return s[1:-1]
        return s


# ============================================================
# CONDITION COMPILER
# ============================================================

class ConditionCompiler:
    """
    Compiles CADSL filter conditions to predicate functions.

    Conditions support:
    - Comparison operators: >, <, >=, <=, ==, !=
    - Logical operators: and, or, not
    - String operators: matches, starts_with, ends_with, contains
    - Membership: in [list], in {param}
    - Null checks: is null, is not null
    """

    def __init__(self):
        self.expr_compiler = ExpressionCompiler()

    def compile(self, node: Tree) -> Callable[[Dict, Optional[Any]], bool]:
        """
        Compile a condition node to a predicate function.

        Args:
            node: Lark tree node representing a condition

        Returns:
            Callable that takes (row, ctx) and returns bool
        """
        method_name = f"_compile_{node.data}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)

        # Fallback: try to compile as expression and convert to bool
        expr = self.expr_compiler.compile(node)
        return lambda r, ctx=None, e=expr: bool(e(r, ctx))

    # --------------------------------------------------------
    # Logical Operators
    # --------------------------------------------------------

    def _compile_or_expr(self, node: Tree) -> Callable:
        """Compile OR expression."""
        conditions = [self.compile(child) for child in node.children
                      if isinstance(child, Tree)]
        if len(conditions) == 1:
            return conditions[0]
        return lambda r, ctx=None, c=conditions: any(cond(r, ctx) for cond in c)

    def _compile_and_expr(self, node: Tree) -> Callable:
        """Compile AND expression."""
        conditions = [self.compile(child) for child in node.children
                      if isinstance(child, Tree)]
        if len(conditions) == 1:
            return conditions[0]
        return lambda r, ctx=None, c=conditions: all(cond(r, ctx) for cond in c)

    def _compile_not_expr(self, node: Tree) -> Callable:
        """Compile NOT expression."""
        inner = self.compile(node.children[0])
        return lambda r, ctx=None, i=inner: not i(r, ctx)

    def _compile_not_cond(self, node: Tree) -> Callable:
        """Compile not_cond node (may be passthrough)."""
        return self.compile(node.children[0])

    def _compile_paren_cond(self, node: Tree) -> Callable:
        """Compile parenthesized condition."""
        return self.compile(node.children[0])

    # --------------------------------------------------------
    # Comparison Operators
    # --------------------------------------------------------

    def _compile_binary_comp(self, node: Tree) -> Callable:
        """Compile binary comparison: left op right"""
        left = self.expr_compiler.compile(node.children[0])
        op_node = node.children[1]
        right = self.expr_compiler.compile(node.children[2])

        op_name = op_node.data if isinstance(op_node, Tree) else str(op_node)

        comparators = {
            "op_gt": lambda a, b: a > b if a is not None and b is not None else False,
            "op_lt": lambda a, b: a < b if a is not None and b is not None else False,
            "op_gte": lambda a, b: a >= b if a is not None and b is not None else False,
            "op_lte": lambda a, b: a <= b if a is not None and b is not None else False,
            "op_eq": lambda a, b: a == b,
            "op_ne": lambda a, b: a != b,
        }

        cmp_fn = comparators.get(op_name, lambda a, b: False)
        return lambda r, ctx=None, l=left, ri=right, c=cmp_fn: c(l(r, ctx), ri(r, ctx))

    # --------------------------------------------------------
    # String Operators
    # --------------------------------------------------------

    def _compile_regex_match(self, node: Tree) -> Callable:
        """Compile 'matches' regex pattern."""
        left = self.expr_compiler.compile(node.children[0])
        pattern = self._unquote(str(node.children[1]))
        compiled = re.compile(pattern)

        def match(r, ctx=None, l=left, p=compiled):
            value = l(r, ctx)
            if value is None:
                return False
            return bool(p.search(str(value)))

        return match

    def _compile_starts_with(self, node: Tree) -> Callable:
        """Compile 'starts_with' check."""
        left = self.expr_compiler.compile(node.children[0])
        prefix = self._unquote(str(node.children[1]))

        def check(r, ctx=None, l=left, p=prefix):
            value = l(r, ctx)
            if value is None:
                return False
            return str(value).startswith(p)

        return check

    def _compile_ends_with(self, node: Tree) -> Callable:
        """Compile 'ends_with' check."""
        left = self.expr_compiler.compile(node.children[0])
        suffix = self._unquote(str(node.children[1]))

        def check(r, ctx=None, l=left, s=suffix):
            value = l(r, ctx)
            if value is None:
                return False
            return str(value).endswith(s)

        return check

    def _compile_contains_str(self, node: Tree) -> Callable:
        """Compile 'contains' check."""
        left = self.expr_compiler.compile(node.children[0])
        substring = self._unquote(str(node.children[1]))

        def check(r, ctx=None, l=left, s=substring):
            value = l(r, ctx)
            if value is None:
                return False
            return s in str(value)

        return check

    # --------------------------------------------------------
    # Membership Operators
    # --------------------------------------------------------

    def _compile_in_list(self, node: Tree) -> Callable:
        """Compile 'in [list]' check."""
        left = self.expr_compiler.compile(node.children[0])

        # Extract list values
        values = []
        for child in node.children[1:]:
            if isinstance(child, Tree) and child.data == "value_list":
                for item in child.children:
                    val = self._extract_value(item)
                    values.append(val)

        return lambda r, ctx=None, l=left, v=values: l(r, ctx) in v

    def _compile_in_param(self, node: Tree) -> Callable:
        """Compile 'in {param}' check."""
        left = self.expr_compiler.compile(node.children[0])
        param_ref = node.children[1]
        param_name = str(param_ref.children[0])

        def check(r, ctx=None, l=left, p=param_name):
            value = l(r, ctx)
            if ctx and hasattr(ctx, 'params'):
                param_val = ctx.params.get(p, [])
                if isinstance(param_val, (list, tuple, set)):
                    return value in param_val
            return False

        return check

    # --------------------------------------------------------
    # Null Checks
    # --------------------------------------------------------

    def _compile_is_null(self, node: Tree) -> Callable:
        """Compile 'is null' check."""
        left = self.expr_compiler.compile(node.children[0])
        return lambda r, ctx=None, l=left: l(r, ctx) is None

    def _compile_is_not_null(self, node: Tree) -> Callable:
        """Compile 'is not null' check."""
        left = self.expr_compiler.compile(node.children[0])
        return lambda r, ctx=None, l=left: l(r, ctx) is not None

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _extract_value(self, node: Tree) -> Any:
        """Extract a literal value from a node."""
        if isinstance(node, Token):
            if node.type == "STRING":
                return self._unquote(str(node))
            elif node.type in ("SIGNED_INT", "INT"):
                return int(str(node))
            elif node.type in ("SIGNED_FLOAT", "FLOAT"):
                return float(str(node))
            return str(node)

        if isinstance(node, Tree):
            if node.data == "val_string":
                return self._unquote(str(node.children[0]))
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
            elif node.children:
                return self._extract_value(node.children[0])

        return None

    def _unquote(self, s: str) -> str:
        """Remove quotes from a string."""
        if len(s) >= 2:
            if (s.startswith('"') and s.endswith('"')) or \
               (s.startswith("'") and s.endswith("'")):
                return s[1:-1]
        return s


# ============================================================
# OBJECT EXPRESSION COMPILER
# ============================================================

class ObjectExprCompiler:
    """
    Compiles CADSL object expressions (for map step) to transform functions.

    Object expressions support:
    - Named fields: {name: expr, ...}
    - Spread operator: {...row, extra: value}
    - String interpolation: "text {field}"
    """

    def __init__(self):
        self.expr_compiler = ExpressionCompiler()

    def compile(self, node: Tree) -> Callable[[Dict, Optional[Any]], Dict]:
        """
        Compile an object expression to a transform function.

        Args:
            node: Lark tree node for object_expr

        Returns:
            Callable that takes (row, ctx) and returns a dict
        """
        fields = []

        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "obj_field":
                    # name: expr
                    name = str(child.children[0])
                    expr = self.expr_compiler.compile(child.children[1])
                    fields.append(("field", name, expr))

                elif child.data == "spread_row":
                    # ...row
                    fields.append(("spread_row", None, None))

                elif child.data == "spread_var":
                    # ...varname
                    var = str(child.children[0])
                    fields.append(("spread_var", var, None))

        def transform(r, ctx=None, f=fields):
            result = {}
            for kind, name, expr in f:
                if kind == "spread_row":
                    if isinstance(r, dict):
                        result.update(r)
                elif kind == "spread_var":
                    if isinstance(r, dict) and name in r:
                        val = r[name]
                        if isinstance(val, dict):
                            result.update(val)
                elif kind == "field":
                    value = expr(r, ctx)
                    # Handle string interpolation in values
                    if isinstance(value, str) and '{' in value:
                        value = self._interpolate(value, r, ctx)
                    result[name] = value
            return result

        return transform

    def _interpolate(self, template: str, row: Dict, ctx: Optional[Any]) -> str:
        """Interpolate {field} placeholders in a string."""
        pattern = re.compile(r'\{(\w+)\}')

        def replacer(m):
            field = m.group(1)
            if isinstance(row, dict) and field in row:
                return str(row[field])
            if ctx and hasattr(ctx, 'params') and field in ctx.params:
                return str(ctx.params[field])
            return m.group(0)

        return pattern.sub(replacer, template)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def compile_expression(node: Union[Tree, Token]) -> Callable[[Dict, Optional[Any]], Any]:
    """Compile an expression node to a callable."""
    compiler = ExpressionCompiler()
    return compiler.compile(node)


def compile_condition(node: Tree) -> Callable[[Dict, Optional[Any]], bool]:
    """Compile a condition node to a predicate function."""
    compiler = ConditionCompiler()
    return compiler.compile(node)


def compile_object_expr(node: Tree) -> Callable[[Dict, Optional[Any]], Dict]:
    """Compile an object expression to a transform function."""
    compiler = ObjectExprCompiler()
    return compiler.compile(node)
