from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
import pathlib
from typing import TypeVar

from src.mod.scanner import Location, KEYWORD_TABLE
from src.mod.parser import parse, ParseError
from src.mod import ast_
from src.mod.ast_.symbol_table_types import (
    SimileType,
    ModuleImports,
    ProcedureTypeDef,
    StructTypeDef,
    EnumTypeDef,
    SimileTypeError,
    BaseSimileType,
)
from src.mod.ast_.symbol_table_env import (
    STARTING_ENVIRONMENT,
    PRIMITIVE_TYPES,
    BUILTIN_FUNCTIONS,
)

RESERVED_KEYWORDS = list(KEYWORD_TABLE.keys())


T = TypeVar("T", bound=ast_.ASTNode)


@dataclass
class ReservedKeywordErr:
    node: ast_.ASTNode
    clashing_keyword: str
    keyword_list_name: str

    def __str__(self) -> str:
        ret = ""
        ret += f"Error: Reserved keyword {self.clashing_keyword} (from keyword list {self.keyword_list_name}) used as identifier within program, "
        ret += f"{self.node.start_location} to {self.node.end_location}\n"
        return ret


def check_clash(node: ast_.ASTNode, name: str) -> ReservedKeywordErr | None:
    if name in PRIMITIVE_TYPES:
        return ReservedKeywordErr(node, name, "STARTING_ENVIRONMENT")
    if name in BUILTIN_FUNCTIONS:
        return ReservedKeywordErr(node, name, "STARTING_ENVIRONMENT")
    if name in RESERVED_KEYWORDS:
        return ReservedKeywordErr(node, name, "RESERVED_KEYWORDS")
    return None


def reserved_keywords_check(ast: T) -> T:

    def traversal_function(node: ast_.ASTNode) -> ReservedKeywordErr | None:
        match node:
            case ast_.Assignment(ast_.Identifier(name), _) | ast_.Assignment(ast_.TypedName(ast_.Identifier(name), _), _):
                return check_clash(node, name)

            case ast_.For(iterable_names, _, _):
                for ident in iterable_names.flatten():
                    if (ret := check_clash(node, ident.name)) is not None:
                        return ret

            case ast_.StructDef(ast_.Identifier(name), _):
                return check_clash(node, name)
            case ast_.ProcedureDef(ast_.Identifier(name), args, _, _):
                if (ret := check_clash(node, name)) is not None:
                    return ret

                for arg in args:
                    assert isinstance(arg.name, ast_.Identifier)
                    if (ret := check_clash(node, arg.name.name)) is not None:
                        return ret
        return None

    errors = list(filter(None, ast_.dataclass_traverse(ast, traversal_function, True, True)))

    if errors:
        raise SimileTypeError(f"Reserved keywords used as identifiers: {'\n'.join(map(str,errors))}")
    return ast
