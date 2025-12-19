from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable, Generic, TypeVar, Any, Iterable


@dataclass
class Trait:
    """A trait that modifies the behavior of a type (usually based on the element type or expected element values).

    Traits can be used to indicate special properties of the set, such as ordering,
    uniqueness, or other characteristics that affect how the set operates.
    """

    _name: str = field(init=False)
    """Name of the trait"""
