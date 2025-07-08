from functools import singledispatchmethod
from dataclasses import dataclass, field
from typing import ClassVar, Any

from src.mod import ast_
from src.mod.codegen.code_generator_base import CodeGenerator, CodeGeneratorError, CodeGenEnvironment


@dataclass
class RustCodeGenerator(CodeGenerator):
    ast: ast_.ASTNode
    new_symbol_table: CodeGenEnvironment = field(init=False)

    def __post_init__(self) -> None:
        self.new_symbol_table = CodeGenEnvironment(
            table={name: RustCodeGenerator.type_translator(typ, name) for name, typ in ast_.STARTING_ENVIRONMENT.table.items()},
        )

    @classmethod
    def type_translator(cls, simile_type: ast_.SimileType, def_name: str = "") -> str:
        """Translate Simile types to C++ types.

        Args:
            simile_type (ast_.SimileType): The Simile type to translate.
            def_name (str, optional): The name of the definition, if applicable (only used for function, enum, and struct definitions). Defaults to "".
        """
        match simile_type:
            case ast_.BaseSimileType.PosInt | ast_.BaseSimileType.Nat:
                return "u64"
            case ast_.BaseSimileType.Int:
                return "i64"
            case ast_.BaseSimileType.Float:
                return "f64"
            case ast_.BaseSimileType.String:
                return "String"
            case ast_.BaseSimileType.Bool:
                return "bool"
            case ast_.BaseSimileType.None_:
                return "null"
            case ast_.PairType(left, right):
                return f"std::pair<{cls.type_translator(left)}, {cls.type_translator(right)}>"
            case ast_.SetType(ast_.PairType(left, right), _):
                return f"std::unordered_map<{cls.type_translator(left)}, {cls.type_translator(right)}>"
            case ast_.SetType(element_type, _):
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
