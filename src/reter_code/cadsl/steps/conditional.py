"""
CADSL Conditional Steps.

Contains step classes for conditional execution in CADSL pipelines:
- WhenStep: Execute inner step only when condition is true
- UnlessStep: Execute inner step only when condition is false
- BranchStep: Execute then/else branches based on condition
- CatchStep: Error handling with default value
"""

from typing import Any, Callable, Dict, Optional


class WhenStep:
    """
    Conditional step - executes inner step only when condition is true.

    Syntax: when { condition } step

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, condition, inner_step_spec):
        self.condition = condition
        self.inner_step_spec = inner_step_spec

    def execute(self, data, ctx=None):
        """Execute inner step if condition is true, otherwise pass through."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

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
        from reter_code.dsl.core import (
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

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, condition, inner_step_spec):
        self.condition = condition
        self.inner_step_spec = inner_step_spec

    def execute(self, data, ctx=None):
        """Execute inner step if condition is false, otherwise pass through."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

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

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, condition, then_step_spec, else_step_spec=None):
        self.condition = condition
        self.then_step_spec = then_step_spec
        self.else_step_spec = else_step_spec

    def execute(self, data, ctx=None):
        """Execute appropriate branch based on condition."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

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

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is a pipeline-step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, default_fn):
        self.default_fn = default_fn

    def execute(self, data, ctx=None):
        """Pass through data (actual error catching is done at pipeline level)."""
        from reter_code.dsl.core import pipeline_ok

        # If we get here, no error occurred - just pass through
        return pipeline_ok(data)
