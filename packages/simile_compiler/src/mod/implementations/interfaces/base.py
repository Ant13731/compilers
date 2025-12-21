from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable, ClassVar, Type, TypeVar


@dataclass
class BaseImplementation:
    """Base class for all Simile implementation libraries."""

    target: ClassVar[str] = "llvm"
