from __future__ import annotations
from dataclasses import dataclass

from src.mod.types.error import SimileTypeError
from src.mod.types.base import BaseType
from src.mod.types.traits import Trait


# @dataclass
# Moved to trait
# class LiteralType(BaseType):
#     """A literal value of a specific type T."""

#     value: ASTNode  # This shouldn't be a type, rather it should be a value "promoted" to a type - so ASTNode?

#     def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
#         if not isinstance(other, LiteralType):
#             return False
#         return self.value == other.value

#     def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
#         if not isinstance(other, LiteralType):
#             return False
#         # TODO We will need to check for subsets of sets, subtuples, etc.
#         raise NotImplementedError

#     def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
#         return self


@dataclass
class GenericType(BaseType):
    """Generic types are used primarily for resolving generic procedures/functions into a specific type based on context.

    IDs are only locally valid (i.e., introduced by a procedure argument and used by a procedure's return value).
    """

    id_: str

    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if self.id_ not in substitution_mapping:
            substitution_mapping[self.id_] = other
        return substitution_mapping[self.id_] == other

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        try:
            return lst.pop(0)
        except IndexError:
            raise SimileTypeError("Failed to replace generic type value: not enough types provided") from None


@dataclass
class DeferToSymbolTable(BaseType):
    """Types dependent on this will not be resolved until the analysis phase.

    Any type-checking functions called on unresolved types should raise an error."""

    lookup_type: str
    """Identifier to look up in table"""


@dataclass
class ModuleImports(BaseType):
    """Type to represent importing these objects into the module namespace

    Any type-checking functions called on environments (which is what this dict really is) should raise an error."""

    import_objects: dict[str, BaseType]
