"""
catpy.py — Category-theory-inspired programming foundations for CADSL.

This module provides the core typeclasses and types used throughout the DSL:
- Core typeclasses: Functor, Applicative, Monad
- Concrete instances: Maybe (Just/Nothing), Result (Ok/Err), ListF (list wrapper)
- Helpers: compose, identity, liftA2 for Applicatives

Based on catpy_full.py, adapted for CADSL pipeline operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Tuple,
    TypeVar,
    Union,
)
from abc import ABC, abstractmethod

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
E = TypeVar("E")

# ---------------------------------------------------------------------------
# Core typeclasses
# ---------------------------------------------------------------------------

class Functor(ABC, Generic[T]):
    """
    A structure that supports mapping a function over the values it contains.

    Laws (for all f: a->b, g: b->c):
      1) Identity:     fmap(id)      == id
      2) Composition:  fmap(g)∘fmap(f) == fmap(g∘f)

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a type-class.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    @abstractmethod
    def fmap(self, f: Callable[[T], U]) -> "Functor[U]":
        """Map a pure function over the structure."""
        raise NotImplementedError

    # Convenience alias
    def map(self, f: Callable[[T], U]) -> "Functor[U]":
        return self.fmap(f)


class Applicative(Functor[T], ABC):
    """
    A Functor that can lift pure values and apply wrapped functions.

    Additional operations:
      - pure: wrap a value in the context
      - ap: apply a wrapped function to a wrapped value

    Laws (for all x, y, z and functions u, v):
      1) Identity:  pure(id).ap(v) == v
      2) Homomorphism: pure(f).ap(pure(x)) == pure(f(x))
      3) Interchange:  u.ap(pure(y)) == pure(lambda f: f(y)).ap(u)
      4) Composition:  pure(compose).ap(u).ap(v).ap(w) == u.ap(v.ap(w))

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a type-class.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    @classmethod
    @abstractmethod
    def pure(cls, x: U) -> "Applicative[U]":
        """Lift a value into the applicative context."""
        raise NotImplementedError

    @abstractmethod
    def ap(self: "Applicative[Callable[[T], U]]", x: "Applicative[T]") -> "Applicative[U]":
        """Apply a wrapped function to a wrapped value."""
        raise NotImplementedError

    # Handy binary lifter: liftA2
    @classmethod
    def liftA2(cls, f: Callable[[T, U], V], a: "Applicative[T]", b: "Applicative[U]") -> "Applicative[V]":
        """
        Lift a binary function into the applicative context.
        Equivalent to: pure(f).ap(a).ap(b)
        """
        return cls.pure(lambda x: lambda y: f(x, y)).ap(a).ap(b)  # type: ignore[misc]


class Monad(Applicative[T], ABC):
    """
    A structure that supports flattening/sequencing (bind).
    Provides a way to chain computations that themselves produce wrapped values.

    Laws (for all x and functions f: a -> m b, g: b -> m c):
      1) Left identity:  pure(x).bind(f) == f(x)
      2) Right identity: m.bind(pure)    == m
      3) Associativity:  m.bind(f).bind(g) == m.bind(lambda x: f(x).bind(g))

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a type-class.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    @abstractmethod
    def bind(self, f: Callable[[T], "Monad[U]"]) -> "Monad[U]":
        """Chain a function that returns a wrapped value (aka flatMap)."""
        raise NotImplementedError

    # Default implementations in terms of bind/pure
    def fmap(self, f: Callable[[T], U]) -> "Monad[U]":  # type: ignore[override]
        return self.bind(lambda a: self.__class__.pure(f(a)))  # type: ignore[misc]

    def ap(self: "Monad[Callable[[T], U]]", x: "Monad[T]") -> "Monad[U]":  # type: ignore[override]
        return self.bind(lambda f: x.bind(lambda a: self.__class__.pure(f(a))))  # type: ignore[misc]

    # Aliases / ergonomics
    def flat_map(self, f: Callable[[T], "Monad[U]"]) -> "Monad[U]":
        return self.bind(f)

    def __rshift__(self, f: Callable[[T], "Monad[U]"]) -> "Monad[U]":
        """Syntactic sugar: m >> f == m.bind(f)"""
        return self.bind(f)


# ---------------------------------------------------------------------------
# Maybe
# ---------------------------------------------------------------------------

class Maybe(Monad[T], ABC):
    """
    Optional value: either Just(value) or Nothing().

    Common patterns:
      - fmap applies a function only when a value exists.
      - bind sequences computations that may fail/return Nothing.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    @classmethod
    def pure(cls, x: U) -> "Maybe[U]":  # type: ignore[override]
        return Just(x)

    def is_just(self) -> bool:
        return isinstance(self, Just)

    def is_nothing(self) -> bool:
        return isinstance(self, Nothing)

    def get_or_else(self, default: T) -> T:
        """Get the value or return default if Nothing."""
        if isinstance(self, Just):
            return self.value
        return default

    def or_else(self, alternative: "Maybe[T]") -> "Maybe[T]":
        """Return self if Just, otherwise return alternative."""
        if isinstance(self, Just):
            return self
        return alternative


@dataclass(frozen=True)
class Just(Maybe[T]):
    """Represents a present value in a Maybe context.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    value: T

    def bind(self, f: Callable[[T], Maybe[U]]) -> Maybe[U]:
        return f(self.value)

    def fmap(self, f: Callable[[T], U]) -> Maybe[U]:  # type: ignore[override]
        return Just(f(self.value))

    def ap(self, x: Maybe[T]) -> Maybe[U]:  # type: ignore[override]
        # self is expected to hold a function
        if callable(self.value):
            if isinstance(x, Just):
                return Just(self.value(x.value))  # type: ignore[misc]
            return Nothing()
        raise TypeError("Just.ap expects a Just(function).")

    def __repr__(self) -> str:
        return f"Just({self.value!r})"


@dataclass(frozen=True)
class Nothing(Maybe[Any]):
    """Represents an absent value in a Maybe context.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    def bind(self, f: Callable[[Any], Maybe[U]]) -> Maybe[U]:
        return self  # type: ignore[return-value]

    def fmap(self, f: Callable[[Any], U]) -> Maybe[U]:  # type: ignore[override]
        return self  # type: ignore[return-value]

    def ap(self, x: Maybe[Any]) -> Maybe[Any]:  # type: ignore[override]
        return self

    def __repr__(self) -> str:
        return "Nothing()"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class Result(Monad[T], ABC, Generic[T, E]):
    """
    Tagged union for success or failure with an error value.
    - Ok(value)
    - Err(error)

    Prefer Result over Maybe when you want to keep *why* it failed.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    @classmethod
    def pure(cls, x: U) -> "Result[U, E]":  # type: ignore[override]
        return Ok(x)

    def is_ok(self) -> bool:
        return isinstance(self, Ok)

    def is_err(self) -> bool:
        return isinstance(self, Err)

    def unwrap(self) -> T:
        """Get the value or raise if Err."""
        if isinstance(self, Ok):
            return self.value
        raise ValueError(f"Cannot unwrap Err: {self}")

    def unwrap_or(self, default: T) -> T:
        """Get the value or return default if Err."""
        if isinstance(self, Ok):
            return self.value
        return default

    def map_err(self, f: Callable[[E], E]) -> "Result[T, E]":
        """Map a function over the error value."""
        if isinstance(self, Err):
            return Err(f(self.error))
        return self


@dataclass(frozen=True)
class Ok(Result[T, E]):
    """Represents a successful result.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    value: T

    def bind(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return f(self.value)

    def fmap(self, f: Callable[[T], U]) -> Result[U, E]:  # type: ignore[override]
        return Ok(f(self.value))

    def ap(self, x: Result[T, E]) -> Result[U, E]:  # type: ignore[override]
        if callable(self.value):
            if isinstance(x, Ok):
                return Ok(self.value(x.value))  # type: ignore[misc]
            return x  # Err propagates
        raise TypeError("Ok.ap expects an Ok(function).")

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True)
class Err(Result[Any, E]):
    """Represents a failed result with error information.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    error: E

    def bind(self, f: Callable[[Any], Result[U, E]]) -> Result[U, E]:
        return self  # type: ignore[return-value]

    def fmap(self, f: Callable[[Any], U]) -> Result[U, E]:  # type: ignore[override]
        return self  # type: ignore[return-value]

    def ap(self, x: Result[Any, E]) -> Result[Any, E]:  # type: ignore[override]
        return self

    def __repr__(self) -> str:
        return f"Err({self.error!r})"


# ---------------------------------------------------------------------------
# ListF (list wrapper as Functor/Applicative/Monad)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ListF(Monad[T]):
    """
    A thin wrapper over an immutable tuple that behaves like a list
    but participates in the typeclasses in a principled way.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a monad.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    items: Tuple[T, ...]

    @classmethod
    def from_iter(cls, it: Iterable[T]) -> "ListF[T]":
        return cls(tuple(it))

    @classmethod
    def empty(cls) -> "ListF[Any]":
        return cls(())

    @classmethod
    def pure(cls, x: U) -> "ListF[U]":  # type: ignore[override]
        return cls((x,))

    def fmap(self, f: Callable[[T], U]) -> "ListF[U]":  # type: ignore[override]
        return ListF(tuple(f(a) for a in self.items))

    def ap(self: "ListF[Callable[[T], U]]", xs: "ListF[T]") -> "ListF[U]":  # type: ignore[override]
        # All combinations of functions and arguments (Cartesian product)
        return ListF(tuple(f(a) for f in self.items for a in xs.items))

    def bind(self, f: Callable[[T], "ListF[U]"]) -> "ListF[U]":
        out: list[U] = []
        for a in self.items:
            out.extend(f(a).items)
        return ListF(tuple(out))

    def filter(self, pred: Callable[[T], bool]) -> "ListF[T]":
        """Filter items by predicate."""
        return ListF(tuple(x for x in self.items if pred(x)))

    def head(self) -> Maybe[T]:
        """Get first element or Nothing."""
        if self.items:
            return Just(self.items[0])
        return Nothing()

    def tail(self) -> "ListF[T]":
        """Get all but first element."""
        return ListF(self.items[1:]) if self.items else ListF(())

    def take(self, n: int) -> "ListF[T]":
        """Take first n elements."""
        return ListF(self.items[:n])

    def drop(self, n: int) -> "ListF[T]":
        """Drop first n elements."""
        return ListF(self.items[n:])

    def concat(self, other: "ListF[T]") -> "ListF[T]":
        """Concatenate two lists."""
        return ListF(self.items + other.items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> T:
        return self.items[idx]

    def __repr__(self) -> str:
        return f"ListF({list(self.items)!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compose(f: Callable[[U], V], g: Callable[[T], U]) -> Callable[[T], V]:
    """Function composition: compose(f, g)(x) == f(g(x))"""
    return lambda x: f(g(x))


def identity(x: T) -> T:
    """Identity function."""
    return x


def const(x: T) -> Callable[[Any], T]:
    """Constant function: const(x)(y) == x for all y."""
    return lambda _: x


def flip(f: Callable[[T, U], V]) -> Callable[[U, T], V]:
    """Flip argument order: flip(f)(x, y) == f(y, x)."""
    return lambda x, y: f(y, x)


# ---------------------------------------------------------------------------
# Pipeline Error Type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineError:
    """Error that occurred during pipeline execution.

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a value-object.
    """
    step: str
    message: str
    cause: Exception = None

    def __str__(self) -> str:
        if self.cause:
            return f"[{self.step}] {self.message}: {self.cause}"
        return f"[{self.step}] {self.message}"


# Type alias for pipeline results
PipelineResult = Result[T, PipelineError]


def pipeline_ok(value: T) -> PipelineResult[T]:
    """Create a successful pipeline result."""
    return Ok(value)


def pipeline_err(step: str, message: str, cause: Exception = None) -> PipelineResult[Any]:
    """Create a failed pipeline result."""
    return Err(PipelineError(step, message, cause))
