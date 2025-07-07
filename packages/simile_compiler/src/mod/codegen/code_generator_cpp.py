from functools import singledispatchmethod
from dataclasses import dataclass, field
from typing import ClassVar, Any

from src.mod import ast_
from src.mod.codegen.code_generator_base import CodeGenerator, CodeGeneratorError


CPP_STARTING_ENVIRONMENT: ast_.Environment = ast_.STARTING_ENVIRONMENT
CPP_STARTING_ENVIRONMENT.table = {name: CPPCodeGenerator.type_translator(typ, name) for name, typ in CPP_STARTING_ENVIRONMENT.table.items()}


@dataclass
class CPPCodeGenerator(CodeGenerator):
    ast: ast_.ASTNode
    new_symbol_table: ast_.Environment = field(default=ast_.STARTING_ENVIRONMENT, init=False, repr=False)

    @classmethod
    def type_translator(cls, simile_type: ast_.SimileType, def_name: str = "") -> str:
        """Translate Simile types to C++ types.

        Args:
            simile_type (ast_.SimileType): The Simile type to translate.
            def_name (str, optional): The name of the definition, if applicable (only used for function, enum, and struct definitions). Defaults to "".
        """
        match simile_type:
            case ast_.BaseSimileType.PosInt | ast_.BaseSimileType.Nat:
                return "unsigned long long"
            case ast_.BaseSimileType.Int:
                return "long long"
            case ast_.BaseSimileType.Float:
                return "double"
            case ast_.BaseSimileType.String:
                return "std::string"
            case ast_.BaseSimileType.Bool:
                return "bool"
            case ast_.BaseSimileType.None_:
                return "null"
            case ast_.PairType(left, right):
                return f"std::pair<{cls.type_translator(left)}, {cls.type_translator(right)}>"
            case ast_.SetType(element_type, relation_subtype):
                return f"std::unordered_set<{cls.type_translator(element_type)}>"
            case ast_.StructTypeDef(fields):
                return f"struct {def_name} {{ {''.join(f'{cls.type_translator(field[1])} {field[0]};' for field in fields.items())} }};"
            case ast_.EnumTypeDef(members):
                return f"enum {def_name} {{ {', '.join(members)} }};"
            case ast_.ProcedureTypeDef(arg_types, return_type):
                return f"{cls.type_translator(return_type)} {def_name}({', '.join(cls.type_translator(arg) for arg in arg_types.values())})"
            case ast_.ModuleImports(import_objects):
                raise ValueError(f"Module imports are not supported in C++ code generation yet. Got: {simile_type}")
            case ast_.TypeUnion(types):
                raise ValueError(f"Union types are not supported in C++ code generation. Got: {simile_type}")
            case ast_.DeferToSymbolTable(lookup_type):
                raise ValueError(f"DeferToSymbolTable types are not supported in C++ code generation (they should be resolved). Got: {simile_type}")
        raise ValueError(f"Unsupported Simile type for C++ translation: {simile_type}")

    def preamble(self) -> str:
        """Generate the C++ preamble."""
        return """
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <string>
"""

    def generate(self) -> str:
        if self.ast._env is None:
            raise ValueError("AST environment should have been populated before code generation (see analysis module).")

        return self._generate_code(self.ast)

    @singledispatchmethod
    def _generate_code(self, ast: ast_.ASTNode) -> str:
        """Auxiliary function for generating LLVM code based on the type of AST node. See :func:`generate_llvm_code`."""
        raise NotImplementedError(f"Code generation not implemented for node type: {type(ast)} with value {ast}")

    @_generate_code.register
    def _(self, ast: ast_.Identifier) -> str:
        # should we lookup the type here?
        return ast.name

    @_generate_code.register
    def _(self, ast: ast_.Literal) -> str:
        match ast:
            case ast_.Int(value) | ast_.Float(value):
                return value
            case ast_.String(value):
                return f'"{value}"'
            case ast_.True_():
                return "true"
            case ast_.False_():
                return "false"
            case ast_.None_():
                return "null"

        raise CodeGeneratorError(f"Unsupported literal type for C++ code generation: {type(ast)} with value {ast}")

    @_generate_code.register
    def _(self, ast: ast_.IdentList) -> str:
        return ", ".join(self._generate_code(ident) for ident in ast.identifiers)

    @_generate_code.register
    def _(self, ast: ast_.BinaryOp) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.RelationOp) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.UnaryOp) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.ListOp) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.Enumeration) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.Type_) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.LambdaDef) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.StructAccess) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.Call) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.Image) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.TypedName) -> str:
        return self._generate_code(ast.name)

    @_generate_code.register
    def _(self, ast: ast_.Assignment) -> str:
        return f"{self._generate_code(ast.target)} = {self._generate_code(ast.value)};"

    @_generate_code.register
    def _(self, ast: ast_.Return) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.ControlFlowStmt) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.Statements) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.Else) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.If) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.Elif) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.For) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.While) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.StructDef) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.ProcedureDef) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.Import) -> str: ...
    @_generate_code.register
    def _(self, ast: ast_.Start) -> str: ...
