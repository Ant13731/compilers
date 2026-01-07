from __future__ import annotations
from typing import Generic, TypeVar

T = TypeVar("T")


class LinkedList(Generic[T]):
    def __init__(self, value: T, next: LinkedList[T] | None = None) -> None:
        self.value = value
        self.next = next

    def append(self, new_value: T) -> None:
        l = self
        while l.next is not None:
            l = l.next
        l.next = LinkedList(new_value)

    def concat(self, other: LinkedList[T]) -> None:
        l = self
        while l.next is not None:
            l = l.next
        l.next = other
