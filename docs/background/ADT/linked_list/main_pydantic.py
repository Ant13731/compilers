import pydantic

from __future__ import annotations
from typing import Generic, TypeVar

T = TypeVar("T")


class LinkedList(pydantic.BaseModel, Generic[T]):
    value: T
    next: LinkedList[T] | None = None

    def append(self, new_value: T) -> None:
        l = self
        while l.next is not None:
            l = l.next
        l.next = LinkedList(value=new_value)

    def concat(self, other: LinkedList[T]) -> None:
        l = self
        while l.next is not None:
            l = l.next
        l.next = other
