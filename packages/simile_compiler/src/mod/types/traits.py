from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable, ClassVar, Generic, TypeVar, Any, Iterable


@dataclass(frozen=True, kw_only=True)
class Trait:
    """A trait that modifies the behavior of a type (usually based on the element type or expected element values).

    Traits can be used to indicate special properties of the set, such as ordering,
    uniqueness, or other characteristics that affect how the set operates.
    """

    trait_name: ClassVar[str]


# Possible traits:
# - domain of values
# - one literal value (can we just use domain for this?)
# - orderable
# - hashable
# - comparable
# - iterable
# - range
# - max/min bounds
# - size (fixed/variable)
# - mutable/immutable
# - relation subtypes (e.g., function, injection, surjection, bijection)
