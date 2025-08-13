from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
import pathlib
from typing import TypeVar, Generic

from src.mod.ast_.symbol_table_types import (
    SimileType,
    ModuleImports,
    ProcedureTypeDef,
    StructTypeDef,
    EnumTypeDef,
    SimileTypeError,
    BaseSimileType,
    DeferToSymbolTable,
    SetType,
    PairType,
    GenericType,
)
from src.mod.ast_.dataclass_helpers import dataclass_find_and_replace

T = TypeVar("T")


@dataclass
class Environment(Generic[T]):

    previous: Environment[T] | None = None
    """Scope of outer-block environments"""

    table: dict[str, T] = field(default_factory=dict)

    def put(self, key: str, value: T) -> None:
        """Put a key-value pair in the current environment."""
        self.table[key] = value

    def get(self, s: str) -> T | None:
        """Get the value associated with the symbol `s` in the environment."""
        current_env: Environment[T] | None = self
        while current_env is not None:
            if s in current_env.table:
                return current_env.table[s]
            current_env = current_env.previous
        return None

    def get_value(self, v: T) -> list[str]:
        """Get the names of the symbols in the environment that match the given value."""
        current_env: Environment[T] | None = self
        names: list[str] = []
        while current_env is not None:
            for name, value in current_env.table.items():
                if value == v:
                    names.append(name)
            current_env = current_env.previous
        return names


@dataclass
class SymbolTableEnvironment(Environment[SimileType]):

    def put_nested_struct(self, assignment_names: list[str], symbol: SimileType) -> None:
        """Put a symbol in the environment, allowing for nested struct access."""
        if not assignment_names:
            raise SimileTypeError("Cannot insert symbol into symbol table with an empty assignment name list")

        prev_fields: dict[str, SimileType] = {}
        for i, assignment_name in enumerate(assignment_names[:-1]):
            if i == 0:
                current_struct_val = self.get(assignment_name)
                if current_struct_val is None:
                    self.put(assignment_name, StructTypeDef(fields={}))
                    prev_fields = self.table[assignment_name].fields  # type: ignore
                    continue
                if not isinstance(current_struct_val, StructTypeDef):
                    raise SimileTypeError(
                        f"Cannot assign to struct field '{assignment_name}' because it is not a struct (current type: {current_struct_val}) (full expected subfields: {assignment_names})"
                    )
                prev_fields = current_struct_val.fields
                continue

            current_fields = prev_fields.get(assignment_name)
            if current_fields is None:
                prev_fields[assignment_name] = StructTypeDef(fields={})
                prev_fields = prev_fields[assignment_name].fields  # type: ignore
                continue
            if isinstance(current_fields, StructTypeDef):
                prev_fields = current_fields.fields
                continue
            raise SimileTypeError(
                f"Cannot assign to struct field '{assignment_name}' because it is not a struct (current type: {current_fields}) (full expected subfields: {assignment_names})"
            )
        assignment_name = assignment_names[-1]
        current_fields = prev_fields.get(assignment_name)
        if current_fields is None:
            prev_fields[assignment_name] = symbol
        if current_fields != symbol:
            raise SimileTypeError(
                f"Cannot assign to struct field '{assignment_name} (under {assignment_names})' because of conflicting types between existing {current_fields} and new {symbol} values"
            )

    def normalize_deferred_types(self) -> None:
        """Normalize deferred types in the current environment."""

        def normalize_deferred_type(symbol: SimileType) -> SimileType | None:
            if isinstance(symbol, DeferToSymbolTable):
                deferred_type = self.get(symbol.lookup_type)
                if deferred_type is None:
                    raise SimileTypeError(f"Failed to find symbol for {symbol.lookup_type} when normalizing deferred type {symbol}")
                return deferred_type
            return None

        for name, symbol in self.table.items():
            # No dataclass, definitely not a deferred type or parent of deferred type
            if not is_dataclass(symbol):
                continue
            self.put(
                name,
                dataclass_find_and_replace(
                    symbol,
                    normalize_deferred_type,
                ),
            )


PRIMITIVE_TYPES = {
    "int": BaseSimileType.Int,
    "str": BaseSimileType.String,
    "float": BaseSimileType.Float,
    "bool": BaseSimileType.Bool,
    "none": BaseSimileType.None_,
    "ℤ": BaseSimileType.Int,
    "ℕ": BaseSimileType.Nat,
    "ℕ₁": BaseSimileType.PosInt,
}

BUILTIN_FUNCTIONS = {
    # These aren't actually procedures - they are processed as rewrites of relational functions later on,
    # but their types can be better expressed using the procedure notation
    "dom": ProcedureTypeDef(
        {
            "s": SetType(
                PairType(
                    GenericType("L"),
                    GenericType("R"),
                ),
            ),
        },
        GenericType("L"),
    ),
    "ran": ProcedureTypeDef(
        {
            "s": SetType(
                PairType(
                    GenericType("L"),
                    GenericType("R"),
                ),
            ),
        },
        GenericType("R"),
    ),
    "card": ProcedureTypeDef(
        {
            "s": SetType(
                PairType(
                    GenericType("L"),
                    GenericType("R"),
                ),
            ),
        },
        BaseSimileType.PosInt,
    ),
}

STARTING_ENVIRONMENT: SymbolTableEnvironment = SymbolTableEnvironment(
    previous=None,
    table={
        **PRIMITIVE_TYPES,
        **BUILTIN_FUNCTIONS,
    },
)
