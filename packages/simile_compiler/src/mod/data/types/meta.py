from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type, ClassVar

from src.mod.data.types.error import SimileTypeError
from src.mod.data.types.base import BaseType
from src.mod.data.traits.traits import Trait, GenericBoundTrait

if TYPE_CHECKING:
    from src.mod.data.symbol_table.entry import SymbolTableIdentifierEntry, ScopeTableEntry


@dataclass
class AnyType_(BaseType):

    def _is_eq_type(self, other: BaseType) -> bool:
        return isinstance(other, AnyType_)

    def _is_subtype(self, other: BaseType) -> bool:
        return False

    def _populate_mandatory_traits(self) -> None:
        pass


@dataclass
class GenericType(BaseType):
    """Generic types are used primarily for resolving generic procedures/functions into a specific type based on context.

    IDs are only locally valid (i.e., introduced by a procedure argument and used by a procedure's return value).
    """

    # id_: str
    valid_traits: ClassVar[set[Type[Trait]]] = {
        *BaseType.valid_traits,
        GenericBoundTrait,
    }

    def _is_eq_type(self, other: BaseType) -> bool:
        if not isinstance(other, GenericType):
            return False
        # return self.id_ == other.id_ and self.trait_collection.generic_bound_trait == other.trait_collection.generic_bound_trait
        return self.trait_collection.generic_bound_trait == other.trait_collection.generic_bound_trait

    def _is_subtype(self, other: BaseType) -> bool:
        if self.trait_collection.generic_bound_trait is None:
            return False  # effectively the AnyType when its not bound

        if not isinstance(other, GenericType):
            # Comparing generic <= concrete means ALL bound types must be a subtype of the concrete
            for self_bound in self.trait_collection.generic_bound_trait.bound_types:
                if not self_bound.is_subtype(other):
                    return False
            return True

        if other.trait_collection.generic_bound_trait is None:
            return True

        # A generic type is a subtype only if all its bound types are subtypes of at least one of the other's bound types
        for self_bound in self.trait_collection.generic_bound_trait.bound_types:
            for other_bound in other.trait_collection.generic_bound_trait.bound_types:
                if self_bound.is_subtype(other_bound):
                    break
            else:
                return False  # no break occurred, so self_bound is not a subtype of any other_bound
        return True

    def _populate_mandatory_traits(self) -> None:
        pass


@dataclass
class DeferToSymbolTable(BaseType):
    """Types dependent on this will not be resolved until the analysis phase.

    Any type-checking functions called on unresolved types should raise an error."""

    lookup_symbol_entry: SymbolTableIdentifierEntry
    """Entry corresponding to the initial identifier"""

    params: list[BaseType]
    """In case the identifier is a generic type (or otherwise expects parameters)"""

    def _is_eq_type(self, other: BaseType) -> bool:
        raise SimileTypeError("Cannot compare DeferToSymbolTable types before resolution")

    def _is_subtype(self, other: BaseType) -> bool:
        raise SimileTypeError("Cannot compare DeferToSymbolTable types before resolution")

    def _populate_mandatory_traits(self) -> None:
        pass


@dataclass
class ImportedSymbol(BaseType):
    imported_symbol_entry: SymbolTableIdentifierEntry

    def _populate_mandatory_traits(self) -> None:
        pass


@dataclass
class ModuleImports(BaseType):
    """Type to represent importing these objects into the module namespace

    Any type-checking functions called on environments (which is what this dict really is) should raise an error."""

    # Function names and types are held within this scope of the symbol table
    scope: ScopeTableEntry

    def _populate_mandatory_traits(self) -> None:
        pass


@dataclass
class TypeOfType(BaseType):
    type_of: BaseType

    def _populate_mandatory_traits(self) -> None:
        pass
