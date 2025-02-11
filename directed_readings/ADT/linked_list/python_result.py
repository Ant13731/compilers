from typing import TypeAlias, TypeVar, Generic

T = TypeVar("T", covariant=True)  # Success type
E = TypeVar("E", covariant=True)  # Error type

class Ok(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

class Err(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

Result: TypeAlias = Ok[T] | Err[E]
