from __future__ import annotations
from dataclasses import dataclass, field
import pathlib
from typing import TypeVar

from src.mod.ast_.type_analysis_types import (
    SimileType,
    ModuleImports,
    ProcedureTypeDef,
    StructTypeDef,
    EnumTypeDef,
    SimileTypeError,
    BaseSimileType,
)
from src.mod.scanner import Location
from src.mod import ast_
from src.mod.parser import parse


class ParseImportError(Exception):
    pass


@dataclass
class SymbolInfo:
    type_: SimileType
    defined_at: Location | None = None  # TODO for better error messages


@dataclass
class Environment:
    previous: Environment | None = None
    table: dict[str, SimileType] = field(default_factory=dict)

    def put(self, s: str, symbol: SimileType) -> None:
        self.table[s] = symbol

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

    def get(self, s: str) -> SimileType | None:
        current_env: Environment | None = self
        while current_env is not None:
            if s in current_env.table:
                return current_env.table[s]
            current_env = current_env.previous
        return None


def check_and_add_for_enum(name: str, value: ast_.ASTNode, current_env: Environment) -> tuple[bool, str]:
    """Returns True if the enum was added, False if its name/builtup identifiers already exists. When False, a reason is provided"""
    if not isinstance(value, ast_.Enumeration):
        return False, f"Value for enum '{name}' is not a valid enumeration"

    if name in current_env.table:
        return False, f"Enum '{name}' is already defined in the current scope"

    # Check if all items are identifiers
    items_to_add: set[str] = set()
    for item in value.items:
        if not isinstance(item, ast_.Identifier):
            return False, f"Enum '{name}' contains non-identifier items: {item} (expected all items to be identifiers)"
        if item.name in current_env.table:
            return False, f"Enum '{name}' contains item '{item.name}' which is already defined in the current scope"
        items_to_add.add(item.name)

    # Add the enum to the environment
    current_env.put(name, EnumTypeDef(members=items_to_add))
    return True, ""


def populate_from_assignment(target: ast_.ASTNode, value: ast_.ASTNode, current_env: Environment) -> None:
    match target:
        case ast_.Identifier(name):
            added_enum, reason = check_and_add_for_enum(name, value, current_env)
            if not added_enum:
                current_env.put(target.name, value.get_type)
        case ast_.TypedName(ast_.Identifier(name), explicit_type):
            if current_env.get(name) is not None and current_env.get(name) != explicit_type:
                raise SimileTypeError(f"Type mismatch: cannot assign explicit type {explicit_type} to {current_env.get(name)} (type clashes with a previous definition)")
            if value.get_type != explicit_type.get_type:
                raise SimileTypeError(f"Type mismatch: cannot assign value of type {value.get_type} to explicit type {explicit_type}")

            if isinstance(explicit_type, ast_.Identifier) and explicit_type.name == "enum":  # should look like ... : enum = ...
                added_enum, reason = check_and_add_for_enum(name, value, current_env)
                if not added_enum:
                    raise SimileTypeError(f"Enum assignment failed: {reason}")
            else:
                current_env.put(name, value.get_type)

        case ast_.StructAccess(struct, field):
            assign_names = [field.name]
            while isinstance(struct, ast_.StructAccess):
                assign_names = [struct.field_name.name] + assign_names
                struct = struct.struct
            if not isinstance(struct, ast_.Identifier):
                raise SimileTypeError(f"Invalid struct access for assignment (can only assign to identifiers): {struct}")

            current_env.put_nested_struct(assign_names, value.get_type)
        case ast_.TypedName(ast_.StructAccess(struct, field), explicit_type):
            if value.get_type != explicit_type.get_type:
                raise SimileTypeError(f"Type mismatch: cannot assign value of type {value.get_type} to explicit type {explicit_type}")

            assign_names = [field.name]
            while isinstance(struct, ast_.StructAccess):
                assign_names = [struct.field_name.name] + assign_names
                struct = struct.struct
            if not isinstance(struct, ast_.Identifier):
                raise SimileTypeError(f"Invalid struct access for assignment (can only assign to identifiers): {struct}")

            current_env.put_nested_struct(assign_names, value.get_type)
        case _:
            raise SimileTypeError(f"Unsupported assignment target type: {type(target)} (expected Identifier or StructAccess or TypedName counterparts)")


T = TypeVar("T", bound=ast_.ASTNode)

STARTING_ENVIRONMENT: Environment = Environment(
    previous=None,
    table={
        "int": ast_.BaseSimileType.Int,
        "str": ast_.BaseSimileType.String,
        "float": ast_.BaseSimileType.Float,
        "bool": ast_.BaseSimileType.Bool,
        "none": ast_.BaseSimileType.None_,
        "ℤ": ast_.BaseSimileType.Int,
        "ℕ": ast_.BaseSimileType.Nat,
        "ℕ₁": ast_.BaseSimileType.PosInt,
    },
)


def populate_ast_with_types(ast: T) -> T:
    """Populate the AST with types for static analysis."""
    current_env: Environment | None = STARTING_ENVIRONMENT

    def populate_ast_with_types_aux(node: T) -> None:
        nonlocal current_env
        # Base case to handle environment nesting
        if isinstance(node, ast_.Statements):
            current_env = Environment(previous=current_env)
            node.env = current_env
            #
            for child in node.children():
                populate_ast_with_types_aux(child)  # type: ignore
            current_env = current_env.previous
            return
        if isinstance(node, ast_.Start):
            # skip past start node
            for child in node.children():
                populate_ast_with_types_aux(child)  # type: ignore
            return
        assert current_env is not None, "Current environment should not be None at this point (Start and Statements should be handled separately)"
        # Now we only need to handle adding to the current environment
        match node:
            case ast_.Assignment(target, value):
                populate_from_assignment(target, value, current_env)
            case ast_.StructDef(ast_.Identifier(name), items):
                fields: dict[str, SimileType] = {}
                for item in items:
                    if not isinstance(item.name, ast_.Identifier):
                        raise SimileTypeError(f"Invalid struct field name (must be an identifier): {item.name}")
                    fields[item.name.name] = item.type_.get_type

                current_env.put(
                    name,
                    StructTypeDef(
                        fields=fields,
                    ),
                )
            case ast_.ProcedureDef(ast_.Identifier(name), args, body, return_type):
                arg_types = {}
                for arg in args:
                    if not isinstance(arg.name, ast_.Identifier):
                        raise SimileTypeError(f"Invalid procedure argument name (must be an identifier): {arg.name}")
                    arg_types[arg.name.name] = arg.get_type

                current_env.put(
                    name,
                    ProcedureTypeDef(
                        arg_types=arg_types,
                        return_type=return_type.get_type,
                    ),
                )
            case ast_.Import(module_file_path, import_objects):
                full_module_path = pathlib.Path(module_file_path).resolve(strict=True)
                with open(full_module_path, "r") as f:
                    module_content = f.read()
                module_ast: ast_.Start | list = parse(module_content)
                if isinstance(module_ast, list):
                    raise ParseImportError(
                        f"Module {module_file_path} does not contain a valid Simile module. " f"Expected a single Start node at the top level.\nReceived errors: {module_ast}"
                    )
                if isinstance(module_ast.body, ast_.None_):
                    return
                module_ast_with_types = populate_ast_with_types(module_ast)
                if isinstance(module_ast_with_types.body, ast_.None_):
                    return
                if module_ast_with_types.body.env is None:
                    # Empty parse tree in module file
                    return

                assert isinstance(module_ast_with_types.body.env, Environment), "Module AST body should have an environment"

                match import_objects:
                    case ast_.ImportAll():
                        for name, symbol in module_ast_with_types.body.env.table.items():
                            current_env.put(name, symbol)
                    case ast_.None_():
                        current_env.put(
                            full_module_path.stem,
                            ModuleImports(module_ast_with_types.body.env.table),
                        )
                    case ast_.IdentList(identifiers):
                        identifier_names = []
                        for identifier in identifiers:
                            if not isinstance(identifier, ast_.Identifier):
                                raise SimileTypeError(f"Invalid import identifier (must be an identifier): {identifier}")
                            identifier_names.append(identifier.name)

                        for name, symbol in module_ast_with_types.body.env.table.items():
                            if name in identifier_names:
                                current_env.put(name, symbol)

            case _:
                return

    populate_ast_with_types_aux(ast)
    return ast


# def test_ast_populate():
#     ast = ast_.Start(
#         body=ast_.Statements(
#             items=[
#                 ast_.Assignment(
#                     target=ast_.Identifier("x"),
#                     value=ast_.Int("42"),
#                 ),
#                 ast_.Identifier("x"),
#             ]
#         )
#     )
#     """Test function to populate the AST with types for static analysis."""
#     ast = populate_ast_with_types(ast)
#     print(ast)
#     print(ast.pretty_print())


# test_ast_populate()
