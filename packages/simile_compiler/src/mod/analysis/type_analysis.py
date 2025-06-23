from __future__ import annotations
from dataclasses import dataclass, field
import pathlib

from mod.ast_.type_analysis_types import (
    SimileType,
    ModuleImports,
    FunctionTypeDef,
    StructTypeDef,
    EnumTypeDef,
    SimileTypeError,
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

    def get(self, s: str) -> SimileType | None:
        current_env: Environment | None = self
        while current_env is not None:
            if s in current_env.table:
                return current_env.table[s]
            current_env = current_env.previous
        return None


def populate_ast_with_types(ast: ast_.ASTNode) -> ast_.ASTNode:
    """Populate the AST with types for static analysis."""
    current_env: Environment | None = None

    def populate_ast_with_types_aux(node: ast_.ASTNode) -> None:
        nonlocal current_env
        # Base case to handle environment nesting
        if isinstance(node, ast_.Statements):
            current_env = Environment(previous=current_env)
            node.env = current_env
            #
            for child in node.children():
                populate_ast_with_types_aux(child)
            current_env = current_env.previous
            return
        if isinstance(node, ast_.Start):
            # skip past start node
            for child in node.children():
                populate_ast_with_types_aux(child)
            return
        assert current_env is not None, "Current environment should not be None at this point (Start and Statements should be handled separately)"
        # Now we only need to handle adding to the current environment
        match node:
            case ast_.Assignment(target, value):
                if isinstance(target, ast_.Identifier):
                    # If the target is an identifier, we can add it to the current environment
                    current_env.put(target.name, value.get_type)
                else:  # can only assign to variables for now
                    raise SimileTypeError(f"Unsupported assignment target type: {type(target)} (expected Identifier)")
            case ast_.StructDef(name, items):
                current_env.put(
                    name,
                    StructTypeDef(
                        fields=dict(map(lambda i: (i.name, i.get_type), items)),
                    ),
                )
            case ast_.EnumDef(name, items):
                current_env.put(
                    name,
                    EnumTypeDef(
                        members=list(map(lambda n: n.name, items)),
                    ),
                )
            case ast_.FunctionDef(name, args, body, return_type):
                current_env.put(
                    name,
                    FunctionTypeDef(
                        args=list(map(lambda n: n.get_type, args)),
                        return_type=return_type.get_type,
                    ),
                )
            case ast_.Import(module_file_path, import_objects):
                full_module_path = pathlib.Path(module_file_path).resolve(strict=True)
                with open(full_module_path, "r") as f:
                    module_content = f.read()
                module_ast: ast_.Start | list = parse(module_content, full_module_path)
                if isinstance(module_ast, list):
                    raise ParseImportError(
                        f"Module {module_file_path} does not contain a valid Simile module. " f"Expected a single Start node at the top level.\nReceived errors: {module_ast}"
                    )
                if isinstance(module_ast.body, ast_.None_):
                    return
                module_ast: ast_.Start = populate_ast_with_types(module_ast)
                if module_ast.body.env is None:
                    # Empty parse tree in module file
                    return

                assert isinstance(module_ast.body.env, Environment), "Module AST body should have an environment"

                match import_objects:
                    case ast_.ImportAll():
                        for name, symbol in module_ast.body.env.table.items():
                            current_env.put(name, symbol)
                    case ast_.None_():
                        current_env.put(
                            full_module_path.stem,
                            ModuleImports(module_file_path=module_ast.body.env.table),
                        )
                    case ast_.IdentList(identifiers):
                        identifier_names = [identifier.name for identifier in identifiers]
                        for name, symbol in module_ast.body.env.table.items():
                            if name in identifier_names:
                                current_env.put(name, symbol)

            case _:
                return
