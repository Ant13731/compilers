# Script to generate the AST nodes - this will overwrite ast_generated.py each time it is run.

from __future__ import annotations
from typing import Any


def ast_config_to_str(config: dict[str, Any]) -> str:
    ret_str = ""
    for name, fields_ in config.items():
        if isinstance(fields_, str):
            ret_str += f"""\
@dataclass
class {name}({fields_}):
    pass


"""
        else:
            assert isinstance(fields_, dict), "Fields must be a dictionary or a string reference to a base class"
            ret_str += f"""\
@dataclass
class {name}(ASTNode):
"""
            if len(fields_) == 0:
                ret_str += "    pass\n"
            else:
                for field_name, field_type in fields_.items():
                    ret_str += f"    {field_name}: {field_type}\n"
            ret_str += "\n\n"
    return ret_str


def add_vars_to_str(var_list: dict[str, list[str]]) -> str:
    ret_str = ""
    for var_type, var_names in var_list.items():
        ret_str += f"{var_type} = {' | '.join(var_names)}\n"
    return ret_str


def ast_generator() -> str:
    "Brings all AST nodes into the global namespace. Linter errors within this function can be ignored."

    ret_str = """\
from __future__ import annotations
from dataclasses import dataclass

try:
    from .ast_base import ASTNode
except ImportError:
    from ast_base import ASTNode  # type: ignore


"""
    ast_primitives_config = {
        "Int": {"value": "int"},
        "Float": {"value": "float"},
        "String": {"value": "str"},
        "Bool": {"value": "bool"},
        "None_": {},  # Represents a None literal
        "Identifier": {"name": "str"},
    }
    ret_str += ast_config_to_str(ast_primitives_config)
    op_config_base = {
        "BinaryOp": {"left": "ASTNode", "right": "ASTNode"},
        "UnaryOp": {"value": "ASTNode"},
        "ListOp": {"items": "list[ASTNode]"},
    }
    ret_str += ast_config_to_str(op_config_base)
    op_config = {
        "And": "ListOp",
        "Or": "ListOp",
        "Not": "UnaryOp",
        "Implies": "BinaryOp",
        "RevImplies": "BinaryOp",
        "Equivalent": "BinaryOp",
        "NotEquivalent": "BinaryOp",
        #
        "Add": "BinaryOp",
        "Subtract": "BinaryOp",
        "Multiply": "BinaryOp",
        "Divide": "BinaryOp",
        "Modulus": "BinaryOp",
        # "Remainder": "BinaryOp",
        "Exponent": "BinaryOp",
        "Equal": "BinaryOp",
        "NotEqual": "BinaryOp",
        "LessThan": "BinaryOp",
        "LessThanOrEqual": "BinaryOp",
        "GreaterThan": "BinaryOp",
        "GreaterThanOrEqual": "BinaryOp",
        #
        "Is": "BinaryOp",
        "IsNot": "BinaryOp",
        #
        "In": "BinaryOp",
        "NotIn": "BinaryOp",
        "Union": "BinaryOp",
        "Intersection": "BinaryOp",
        "Difference": "BinaryOp",
        "Subset": "BinaryOp",
        "SubsetEq": "BinaryOp",
        "Superset": "BinaryOp",
        "SupersetEq": "BinaryOp",
        "NotSubset": "BinaryOp",
        "NotSubsetEq": "BinaryOp",
        "NotSuperset": "BinaryOp",
        "NotSupersetEq": "BinaryOp",
        "UnionAll": "ListOp",
        "IntersectionAll": "ListOp",
        "Powerset": "UnaryOp",
        #
        "Maplet": "BinaryOp",
        "RelationOverride": "BinaryOp",
        "Composition": "BinaryOp",
        "CartesianProduct": "BinaryOp",
        "Inverse": "UnaryOp",
        "DomainSubtraction": "BinaryOp",
        "DomainRestriction": "BinaryOp",
        "RangeSubtraction": "BinaryOp",
        "RangeRestriction": "BinaryOp",
        #
        "RelationOp": "BinaryOp",
        "TotalRelationOp": "BinaryOp",
        "SurjectiveRelationOp": "BinaryOp",
        "TotalSurjectiveRelation": "BinaryOp",
        "PartialFunction": "BinaryOp",
        "TotalFunction": "BinaryOp",
        "PartialInjection": "BinaryOp",
        "TotalInjection": "BinaryOp",
        "PartialSurjection": "BinaryOp",
        "TotalSurjection": "BinaryOp",
        "Bijection": "BinaryOp",
        #
        "UpTo": "BinaryOp",
    }
    ret_str += ast_config_to_str(op_config)
    caller_nodes_config_pre = {
        "Arguments": {"items": "list[ASTNode]"},
        "Type_": {"type_": "ASTNode"},
        "Slice": {"items": "list[ASTNode]"},
    }
    ret_str += ast_config_to_str(caller_nodes_config_pre)
    caller_nodes_config = {
        "StructAccess": {"struct": "ASTNode", "field_name": "Identifier"},
        "FunctionCall": {"function_name": "ASTNode", "args": "Arguments"},
        "TypedName": {"name": "Identifier", "type_": "Type_ | None_"},
        "Indexing": {"target": "ASTNode", "index": "ASTNode | Slice | None_"},
    }
    ret_str += ast_config_to_str(caller_nodes_config)
    ret_str += ast_config_to_str({"ArgDef": {"items": "list[TypedName]"}})
    statements_config = {
        "Assignment": {"target": "ASTNode", "value": "ASTNode"},
        "Return": {"value": "ASTNode | None_"},
        "LambdaDef": {"args": "ArgDef", "body": "ASTNode"},
        "Break": {},
        "Continue": {},
        "Pass": {},
    }
    ret_str += ast_config_to_str(statements_config)
    ret_str += ast_config_to_str({"Statements": {"items": "list[ASTNode]"}})
    ret_str += ast_config_to_str({"Else": {"body": "ASTNode | Statements"}})
    compound_statements_config = {
        "If": {"condition": "ASTNode", "body": "ASTNode | Statements", "else_body": "Else | None_"},
        "Elif": {"condition": "ASTNode", "body": "ASTNode | Statements", "else_body": "Elif | Else | None_"},
        "For": {"iterable_names": "list[Identifier]", "iterable": "ASTNode", "body": "ASTNode | Statements"},
        "StructDef": {"name": "Identifier", "items": "list[TypedName]"},
        "EnumDef": {"name": "Identifier", "items": "list[Identifier]"},
        "FunctionDef": {"name": "Identifier", "args": "ArgDef", "body": "ASTNode | Statements", "return_type": "Type_ | None_"},
    }
    ret_str += ast_config_to_str(compound_statements_config)
    ret_str += ast_config_to_str({"KeyPair": {"key": "ASTNode", "value": "ASTNode"}})
    complex_literals_config_base = {
        "SeqLike": {"items": "list[ASTNode]"},
        "MapLike": {"items": "list[KeyPair]"},
    }
    ret_str += ast_config_to_str(complex_literals_config_base)
    complex_literals_config = {
        "SequenceLiteral": "SeqLike",
        "SetLiteral": "SeqLike",
        "BagLiteral": "MapLike",
        "MappingLiteral": "MapLike",
    }
    ret_str += ast_config_to_str(complex_literals_config)
    comprehension_config_base = {
        "SeqLikeComprehension": {"bound_identifiers": "list[Identifier]", "generator": "ASTNode", "mapping_expression": "ASTNode | None_"},
        "MapLikeComprehension": {"bound_identifiers": "list[KeyPair]", "generator": "ASTNode", "mapping_expression": "ASTNode | None_"},
    }
    ret_str += ast_config_to_str(comprehension_config_base)
    comprehension_config = {
        "SequenceComprehension": "SeqLikeComprehension",
        "SetComprehension": "SeqLikeComprehension",
        "BagComprehension": "SeqLikeComprehension",
        "MappingComprehension": "MapLikeComprehension",
    }
    ret_str += ast_config_to_str(comprehension_config)
    ret_str += ast_config_to_str({"Start": {"body": "Statements | None_"}})

    var_defs = {
        "PrimitiveLiteral": ["Int", "Float", "String", "Bool", "None_"],
        "ComplexLiteral": ["SequenceLiteral", "SetLiteral", "BagLiteral", "MappingLiteral"],
        "ComplexComprehension": ["SequenceComprehension", "SetComprehension", "BagComprehension", "MappingComprehension"],
        "PrimaryStmt": ["StructAccess", "FunctionCall", "Indexing", "PrimitiveLiteral", "ComplexLiteral", "ComplexComprehension", "Identifier"],
        "EquivalenceStmt": ["BinaryOp", "UnaryOp", "ListOp", "PrimaryStmt"],
        "Expr": ["EquivalenceStmt", "LambdaDef"],
        "Atom": ["Identifier", "Expr"],
        "SimpleStmt": ["Expr", "Assignment", "Return", "Break", "Continue", "Pass"],
        "CompoundStmt": ["If", "For", "StructDef", "EnumDef", "FunctionDef"],
    }

    ret_str += add_vars_to_str(var_defs)
    return ret_str


if __name__ == "__main__":
    # Overwrite ast_generated file each time
    with open("C:/Users/hunta/Documents/GitHub/compilers/packages/simile_compiler/simile_compiler/ast_generated.py", "w") as f:
        f.write(ast_generator())
