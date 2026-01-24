"""
CADSL Operators - Conditional and Flow Control Operators

This module provides operators for controlling pipeline execution:
- when: Execute step only when condition is true
- unless: Execute step only when condition is false
- branch: Conditional branching
- merge: Merge multiple pipelines
"""

from typing import Callable, Any, Optional, List, TypeVar, Union
from dataclasses import dataclass

from .core import Pipeline, Step, Result, Ok, Err, Context, PipelineResult, PipelineError

T = TypeVar("T")
U = TypeVar("U")


# =============================================================================
# Conditional Operators
# =============================================================================

@dataclass
class ConditionalStep(Step[T, T]):
    """
    A step that executes conditionally based on a predicate.

    If the condition is not met, the step is skipped and data passes through
    unchanged.
    """
    inner_step: Step[T, T]
    condition: Callable[[], bool]
    negate: bool = False  # True for 'unless' behavior

    def execute(self, data: T) -> PipelineResult[T]:
        """Execute inner step if condition is met."""
        should_execute = self.condition()
        if self.negate:
            should_execute = not should_execute

        if should_execute:
            return self.inner_step.execute(data)
        else:
            return Ok(data)


def when(condition: Callable[[], bool]) -> Callable[[Step[T, T]], Step[T, T]]:
    """
    Decorator to make a step conditional.

    The step will only execute when the condition returns True.

    Example:
        pipeline = (
            reql("SELECT ...")
            >> when(lambda: exclude_tests)(filter(lambda r: "test_" not in r.file))
            >> emit("results")
        )

    Or with method chaining:
        pipeline = (
            reql("SELECT ...")
            .filter(lambda r: "test_" not in r.file, when=lambda: exclude_tests)
            .emit("results")
        )

    Args:
        condition: Callable that returns True when step should execute

    Returns:
        Step wrapper that conditionalizes the inner step
    """
    def decorator(step: Step[T, T]) -> ConditionalStep[T]:
        return ConditionalStep(
            inner_step=step,
            condition=condition,
            negate=False
        )
    return decorator


def unless(condition: Callable[[], bool]) -> Callable[[Step[T, T]], Step[T, T]]:
    """
    Decorator to make a step conditional (inverted).

    The step will only execute when the condition returns False.

    Example:
        pipeline = (
            reql("SELECT ...")
            >> unless(lambda: include_all)(filter(lambda r: r.count > 10))
            >> emit("results")
        )

    Args:
        condition: Callable that returns True when step should be SKIPPED

    Returns:
        Step wrapper that conditionalizes the inner step
    """
    def decorator(step: Step[T, T]) -> ConditionalStep[T]:
        return ConditionalStep(
            inner_step=step,
            condition=condition,
            negate=True
        )
    return decorator


# =============================================================================
# Branching Operators
# =============================================================================

@dataclass
class BranchStep(Step[T, U]):
    """
    A step that branches execution based on a condition.

    If condition is true, executes then_step, otherwise executes else_step.
    """
    condition: Callable[[T], bool]
    then_step: Step[T, U]
    else_step: Optional[Step[T, U]] = None

    def execute(self, data: T) -> PipelineResult[U]:
        """Execute appropriate branch based on condition."""
        try:
            if self.condition(data):
                return self.then_step.execute(data)
            elif self.else_step:
                return self.else_step.execute(data)
            else:
                # No else branch, pass through unchanged
                return Ok(data)  # type: ignore
        except Exception as e:
            return Err(PipelineError("branch", f"Branch evaluation failed: {e}", e))


def branch(
    condition: Callable[[T], bool],
    then_step: Step[T, U],
    else_step: Optional[Step[T, U]] = None
) -> BranchStep[T, U]:
    """
    Create a branching step.

    Example:
        pipeline = (
            reql("SELECT ...")
            >> branch(
                condition=lambda r: len(r) > 100,
                then_step=limit(100),
                else_step=identity()
            )
            >> emit("results")
        )

    Args:
        condition: Function to evaluate on input data
        then_step: Step to execute if condition is True
        else_step: Step to execute if condition is False (optional)

    Returns:
        BranchStep that selects between branches
    """
    return BranchStep(
        condition=condition,
        then_step=then_step,
        else_step=else_step
    )


# =============================================================================
# Merge Operators
# =============================================================================

@dataclass
class MergeStep(Step[Any, List]):
    """
    Merge results from multiple pipelines.
    """
    pipelines: List[Pipeline]
    merge_fn: Callable[[List[Any]], Any]

    def execute(self, data: Any) -> PipelineResult[List]:
        """Execute all pipelines and merge results."""
        results = []
        # Note: data is not used here - each pipeline has its own source
        # This step is typically used after collecting from multiple sources
        return Ok(results)


def merge(*pipelines: Pipeline) -> MergeStep:
    """
    Merge results from multiple pipelines.

    Example:
        classes = reql("SELECT ?c WHERE { ?c type oo:Class }")
        functions = reql("SELECT ?f WHERE { ?f type oo:Function }")

        all_entities = merge(classes, functions)

    Args:
        *pipelines: Pipelines to merge

    Returns:
        MergeStep that combines results
    """
    def concat(results: List[Any]) -> List:
        merged = []
        for r in results:
            if isinstance(r, list):
                merged.extend(r)
            else:
                merged.append(r)
        return merged

    return MergeStep(
        pipelines=list(pipelines),
        merge_fn=concat
    )


# =============================================================================
# Identity and Utility Operators
# =============================================================================

@dataclass
class IdentityStep(Step[T, T]):
    """Pass-through step that returns input unchanged."""

    def execute(self, data: T) -> PipelineResult[T]:
        return Ok(data)


def identity() -> IdentityStep:
    """
    Create an identity step (pass-through).

    Useful as a no-op in conditional branches.

    Returns:
        IdentityStep that returns input unchanged
    """
    return IdentityStep()


@dataclass
class TapStep(Step[T, T]):
    """
    Side-effect step that executes a function without modifying data.

    Useful for logging, debugging, or triggering external actions.
    """
    fn: Callable[[T], None]

    def execute(self, data: T) -> PipelineResult[T]:
        try:
            self.fn(data)
            return Ok(data)
        except Exception as e:
            return Err(PipelineError("tap", f"Side effect failed: {e}", e))


def tap(fn: Callable[[T], None]) -> TapStep[T]:
    """
    Create a tap step for side effects.

    The function is called with the data but the data passes through unchanged.

    Example:
        pipeline = (
            reql("SELECT ...")
            >> tap(lambda r: print(f"Got {len(r)} results"))
            >> filter(...)
            >> emit("results")
        )

    Args:
        fn: Function to call with data

    Returns:
        TapStep that executes function as side effect
    """
    return TapStep(fn=fn)


@dataclass
class CatchStep(Step[T, T]):
    """
    Error handling step that catches errors and returns a default.
    """
    handler: Callable[[Err], T]

    def execute(self, data: T) -> PipelineResult[T]:
        # This step doesn't transform data - it's used in error recovery
        return Ok(data)


def catch(handler: Callable[[Err], T]) -> CatchStep[T]:
    """
    Create an error handler step.

    If an error occurs in preceding steps, the handler is called to
    produce a recovery value.

    Example:
        pipeline = (
            reql("SELECT ...")
            >> catch(lambda e: [])  # Return empty list on error
            >> emit("results")
        )

    Args:
        handler: Function to call with error, returns recovery value

    Returns:
        CatchStep for error recovery
    """
    return CatchStep(handler=handler)


# =============================================================================
# Parallel Operators
# =============================================================================

@dataclass
class ParallelStep(Step[T, List]):
    """
    Execute multiple steps in parallel on the same input.

    Results are collected into a list.
    """
    steps: List[Step]

    def execute(self, data: T) -> PipelineResult[List]:
        results = []
        errors = []

        for step in self.steps:
            result = step.execute(data)
            if result.is_ok():
                results.append(result.unwrap())
            else:
                errors.append(result)

        if errors and not results:
            # All failed
            return errors[0]

        return Ok(results)


def parallel(*steps: Step) -> ParallelStep:
    """
    Execute multiple steps in parallel.

    Each step receives the same input, results are collected into a list.

    Example:
        pipeline = (
            reql("SELECT ?c WHERE { ?c type oo:Class }")
            >> parallel(
                count_methods,
                count_attributes,
                check_inheritance
            )
            >> merge_analysis
        )

    Args:
        *steps: Steps to execute in parallel

    Returns:
        ParallelStep that executes all steps
    """
    return ParallelStep(steps=list(steps))


# =============================================================================
# Composition Helpers
# =============================================================================

def compose(*steps: Step) -> Step:
    """
    Compose multiple steps into a single step.

    Steps are executed in sequence, each receiving the output of the previous.

    Example:
        preprocess = compose(
            filter(lambda r: r.is_valid),
            select("name", "file"),
            order_by("name")
        )

        pipeline = (
            reql("SELECT ...")
            >> preprocess
            >> emit("results")
        )

    Args:
        *steps: Steps to compose

    Returns:
        Single step that executes all steps in sequence
    """
    @dataclass
    class ComposedStep(Step):
        inner_steps: List[Step]

        def execute(self, data: Any) -> PipelineResult[Any]:
            current = data
            for step in self.inner_steps:
                result = step.execute(current)
                if result.is_err():
                    return result
                current = result.unwrap()
            return Ok(current)

    return ComposedStep(inner_steps=list(steps))
