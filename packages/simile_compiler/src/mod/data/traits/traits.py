from __future__ import annotations
from dataclasses import dataclass, field, fields
from copy import deepcopy
from typing import ClassVar, Any, TYPE_CHECKING

from src.mod.data.traits.error import SimileTraitError
from src.mod.data.traits.trait import Trait

if TYPE_CHECKING:
    from src.mod.data.ast_.base import ASTNode
    from src.mod.data.types.base import BaseType


@dataclass
class OrderableTrait(Trait):
    name: ClassVar[str] = "orderable"

    required_methods: ClassVar[set[str]] = {
        "greater_than",
        "less_than",
        "greater_than_equals",
        "less_than_equals",
    }

    @classmethod
    def is_orderable(cls, values: list[ASTNode]) -> bool:
        for value in values:
            if value.get_type.trait_collection.orderable_trait is None:  # type: ignore # TODO implement
                return False
        return True


@dataclass
class UniqueElementsTrait(Trait):
    name: ClassVar[str] = "unique_elements"


@dataclass
class IterableTrait(Trait):
    name: ClassVar[str] = "iterable"


@dataclass
class LiteralTrait(Trait):
    name: ClassVar[str] = "literal"
    value: ASTNode


@dataclass
class DomainTrait(Trait):
    name: ClassVar[str] = "domain"
    values: list[ASTNode]

    def __post_init__(self):
        # Ensure all values are unique
        unique_values = []
        for literal in self.values:
            if literal not in unique_values:
                unique_values.append(literal)
        self.values = unique_values

    def merge(self, other: DomainTrait) -> DomainTrait:
        combined_values = self.values
        for value in other.values:
            if value not in combined_values:
                combined_values.append(value)

        return self.__class__(values=combined_values)


@dataclass
class MinTrait(Trait):
    name: ClassVar[str] = "minimum"
    value: Any | ASTNode

    @classmethod
    def from_domain_trait(cls, trait: DomainTrait) -> MinTrait | None:
        if not OrderableTrait.is_orderable(trait.values):
            return None

        min_value = None
        for value in trait.values:
            # Most ASTNodes wont have literal_min defined
            # This is mostly meant for ints and floats
            if not hasattr(value, "literal_min"):
                return None

            if min_value is None:
                min_value = value
                continue

            # When literal_min returns None, that means that one of the inputs wasn't a literal
            min_candidate = value.literal_min(min_value)  # type: ignore
            if min_candidate is None:
                return None

            min_value = value

        return cls(value=min_value)

    def merge(self, other: MinTrait) -> MinTrait:
        if not hasattr(self.value, "literal_min"):
            raise SimileTraitError("Cannot merge MinTrait: values are not comparable")

        min_candidate = self.value.literal_min(other.value)  # type: ignore
        if min_candidate is None:
            raise SimileTraitError("Cannot merge MinTrait: values are not comparable")
        return MinTrait(value=min_candidate)


@dataclass
class MaxTrait(Trait):
    name: ClassVar[str] = "maximum"
    value: Any | ASTNode

    @classmethod
    def from_domain_trait(cls, trait: DomainTrait) -> MaxTrait | None:
        if not OrderableTrait.is_orderable(trait.values):
            return None

        max_value = None
        for value in trait.values:
            # Most ASTNodes wont have literal_max defined
            # This is mostly meant for ints and floats
            if not hasattr(value, "literal_max"):
                return None

            if max_value is None:
                max_value = value
                continue

            # When literal_max returns None, that means that one of the inputs wasn't a literal
            max_candidate = value.literal_max(max_value)  # type: ignore
            if max_candidate is None:
                return None

            max_value = value

        return cls(value=max_value)

    def merge(self, other: MaxTrait) -> MaxTrait:
        if not hasattr(self.value, "literal_max"):
            raise SimileTraitError("Cannot merge MaxTrait: values are not comparable")

        max_candidate = self.value.literal_max(other.value)  # type: ignore
        if max_candidate is None:
            raise SimileTraitError("Cannot merge MaxTrait: values are not comparable")
        return MaxTrait(value=max_candidate)


@dataclass
class SizeTrait(Trait):
    name: ClassVar[str] = "size"
    size: int


@dataclass
class ImmutableTrait(Trait):
    name: ClassVar[str] = "immutable"


@dataclass
class TotalTrait(Trait):
    name: ClassVar[str] = "total"


@dataclass
class EmptyTrait(Trait):
    name: ClassVar[str] = "empty"


@dataclass
class TotalOnDomainTrait(Trait):
    name: ClassVar[str] = "total_on_domain"


@dataclass
class TotalOnRangeTrait(Trait):
    name: ClassVar[str] = "surjective"


@dataclass
class ManyToOneTrait(Trait):
    name: ClassVar[str] = "many_to_one"


@dataclass
class OneToManyTrait(Trait):
    name: ClassVar[str] = "one_to_many"


@dataclass
class GenericBoundTrait(Trait):
    name: ClassVar[str] = "generic_bound"
    bound_types: list[BaseType]
