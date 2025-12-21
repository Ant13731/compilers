from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable, Type, TypeVar

from src.mod.types.error import SimileTypeError
from src.mod.types.traits import Trait, TraitCollection


T = TypeVar("T", bound="BaseType")


# Primitive types
@dataclass(kw_only=True)
class BaseType:
    """Base type for all Simile types."""

    # TODO should traits be a set? we really shouldn't care about order or duplicates...
    trait_collection: TraitCollection = field(default_factory=TraitCollection)

    # Actual type methods
    def cast(self, caster: T, add_trait_collection: TraitCollection | None = None) -> T:
        """Cast the type to a different type."""
        caster = deepcopy(caster)
        # TODO only add traits if the traits make sense to add (ex. no min trait allowed on a StringType)
        # Each type should specify which traits are allowed

        if add_trait_collection is not None:
            caster.trait_collection = self.trait_collection.merge(add_trait_collection)
        return caster

    def _allowed_traits(self) -> list[Type[Trait]]:
        """Return a list of allowed trait types for this type."""
        raise NotImplementedError

    def equals(self, other: BaseType) -> BoolType:
        """Check if this type is equal to another type."""
        raise NotImplementedError

    def not_equals(self, other: BaseType) -> BoolType:
        raise NotImplementedError

    # Helper methods
    # TODO GO THROUGH EACH CHILD METHOD AND CHECK THE TYPECHECKING FUNCTION AGAINST THE FORMAL TYPE SYSTEM IN THE SPEC
    def is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType] | None = None, check_traits: bool = False) -> bool:
        if substitution_mapping is None:
            substitution_mapping = {}
        if check_traits:
            return self._is_eq_type(other, substitution_mapping) and self._is_eq_traits(other)
        else:
            return self._is_eq_type(other, substitution_mapping)

    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        raise NotImplementedError

    def _is_eq_traits(self, other: BaseType) -> bool:
        """Check whether the type would be equal when considering traits."""
        return self.trait_collection == other.trait_collection

    def is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType] | None = None, check_traits: bool = False) -> bool:
        """Check if self is a sub-type of other (in formal type theory, whether self <= other)."""
        if substitution_mapping is None:
            substitution_mapping = {}

        # Reflexive Subtype
        if self.is_eq_type(other, substitution_mapping, check_traits):
            return True

        is_sub_trait = True
        if check_traits:
            is_sub_trait = self._is_sub_traits(other)

        # Sub Top Type
        if isinstance(other, AnyType_):
            return is_sub_trait

        return is_sub_trait and self._is_sub_type(other, substitution_mapping)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        raise NotImplementedError

    def _is_sub_traits(self, other: BaseType) -> bool:
        """Check whether the type is a sub-type when considering traits."""
        raise NotImplementedError

    def replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        """Structurally replace generic types in self according to the provided list of types."""
        new_lst = deepcopy(lst)
        return self._replace_generic_types(new_lst)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        raise NotImplementedError

    @classmethod
    def max_type(cls, types: list[BaseType]) -> BaseType:
        """Return the widest type among the inputs.

        Throws a SimileTypeError if types are incompatible (aside from AnyType_)."""
        widest_type = types[0]
        for type_ in types:
            # Widen type as necessary
            if widest_type.is_sub_type(type_):
                widest_type = type_
            elif not type_.is_sub_type(widest_type):
                raise SimileTypeError(f"Cannot find max (widest) type with incompatible element types: {widest_type} and {type_}")
        return widest_type

    @classmethod
    def min_type(cls, types: list[BaseType]) -> BaseType:
        """Return the widest type among the inputs.

        Throws a SimileTypeError if types are incompatible (aside from NoneType_)."""
        narrowest_type = types[0]
        for type_ in types:
            # Widen type as necessary
            if type_.is_sub_type(narrowest_type):
                narrowest_type = type_
            elif not narrowest_type.is_sub_type(type_):
                raise SimileTypeError(f"Cannot find min (narrowest) type with incompatible element types: {narrowest_type} and {type_}")
        return narrowest_type


# BoolType needs to be here to avoid circular imports
@dataclass
class BoolType(BaseType):
    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, BoolType)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, BoolType)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self

    def not_(self) -> BoolType:
        return BoolType()

    def equivalent(self, other: BaseType) -> BoolType:
        return BoolType()

    def not_equivalent(self, other: BaseType) -> BoolType:
        return BoolType()

    def implies(self, other: BaseType) -> BoolType:
        return BoolType()

    def and_(self, other: BaseType) -> BoolType:
        return BoolType()

    def or_(self, other: BaseType) -> BoolType:
        return BoolType()


@dataclass
class AnyType_(BaseType):

    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, AnyType_)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return False

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self
