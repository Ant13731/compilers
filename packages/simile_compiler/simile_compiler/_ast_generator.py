# NO LONGER IN USE
# # Script to generate the AST nodes - this will overwrite ast_generated.py each time it is run.

# from __future__ import annotations
# from typing import Any


# def ast_config_to_str(config: dict[str, Any]) -> str:
#     ret_str = ""
#     for name, fields_ in config.items():
#         if isinstance(fields_, dict):
#             ret_str += f"""\
# @dataclass
# class {name}(ASTNode):
# """
#             if len(fields_) == 0:
#                 ret_str += "    pass\n"
#             else:
#                 for field_name, field_type in fields_.items():
#                     ret_str += f"    {field_name}: {field_type}\n"
#             ret_str += "\n\n"
#     return ret_str


# def add_vars_to_str(var_list: dict[str, list[str]]) -> str:
#     ret_str = ""
#     for var_type, var_names in var_list.items():
#         ret_str += f"{var_type} = {' | '.join(var_names)}\n"
#     return ret_str


# def ast_generator() -> str:
#     "Brings all AST nodes into the global namespace. Linter errors within this function can be ignored."

#     ret_str = """\
# from __future__ import annotations
# from dataclasses import dataclass

# try:
#     from .ast_base import (
#         ASTNode,
#         Identifier,
#         BinaryOpType,
#         RelationTypes,
#         UnaryOpType,
#         ListBoolType,
#         BoolQuantifierType,
#         QuantifierType,
#         ControlFlowType,
#         CollectionType,
#     )
# except ImportError:
#     from ast_base import (  # type: ignore
#         ASTNode,
#         Identifier,
#         BinaryOpType,
#         RelationTypes,
#         UnaryOpType,
#         ListBoolType,
#         BoolQuantifierType,
#         QuantifierType,
#         ControlFlowType,
#         CollectionType,
#     )


# """
#     ast_primitives_config = {
#         "Int": {"value": "str"},  # make these int and float? or leave as string for llvm interpretation?
#         "Float": {"value": "str"},
#         "String": {"value": "str"},
#         "True_": {},
#         "False_": {},
#         "None_": {},  # Represents a None literal
#         # "Identifier": {"name": "str"},
#         "IdentList": {"items": "list[Identifier]"},
#     }
#     ret_str += ast_config_to_str(ast_primitives_config)
#     op_config_base = {
#         "BinaryOp": {"left": "ASTNode", "right": "ASTNode", "op_type": "BinaryOpType"},
#         "RelationOp": {"left": "ASTNode", "right": "ASTNode", "op_type": "RelationTypes"},
#         "UnaryOp": {"value": "ASTNode", "op_type": "UnaryOpType"},
#         "ListOp": {"items": "list[ASTNode]", "op_type": "ListBoolType"},
#         "BoolQuantifier": {
#             "bound_identifiers": "IdentList",
#             "predicate": "ASTNode",
#             "op_type": "BoolQuantifierType",
#         },
#         "Quantifier": {
#             "bound_identifiers": "IdentList",
#             "predicate": "ASTNode",
#             "expression": "ASTNode",
#             "op_type": "QuantifierType",
#         },
#         "ControlFlowStmt": {"op_type": "ControlFlowType"},
#         "Enumeration": {"items": "list[ASTNode]", "op_type": "CollectionType"},
#         "Comprehension": {
#             "bound_identifiers": "IdentList",
#             "predicate": "ASTNode",
#             "expression": "ASTNode",
#             "op_type": "CollectionType",
#         },
#     }
#     ret_str += ast_config_to_str(op_config_base)
#     caller_nodes_config_pre = {
#         "Type_": {"type_": "ASTNode"},
#         "Slice": {"items": "list[ASTNode]"},
#         "LambdaDef": {"ident_pattern": "list[ASTNode]", "predicate": "ASTNode", "expression": "ASTNode"},
#     }
#     ret_str += ast_config_to_str(caller_nodes_config_pre)
#     caller_nodes_config = {
#         "StructAccess": {"struct": "ASTNode", "field_name": "Identifier"},
#         "FunctionCall": {"function_name": "ASTNode", "args": "list[ASTNode]"},
#         "TypedName": {"name": "Identifier", "type_": "Type_ | None_"},
#         "Indexing": {"target": "ASTNode", "index": "ASTNode | Slice | None_"},
#     }
#     ret_str += ast_config_to_str(caller_nodes_config)
#     statements_config = {
#         "Assignment": {"target": "ASTNode", "value": "ASTNode"},
#         "Return": {"value": "ASTNode | None_"},
#     }
#     ret_str += ast_config_to_str(statements_config)
#     compound_statements_config = {
#         "Statements": {"items": "list[ASTNode]"},
#         "Else": {"body": "ASTNode | Statements"},
#         "If": {"condition": "ASTNode", "body": "ASTNode | Statements", "else_body": "Elif | Else | None_"},
#         "Elif": {"condition": "ASTNode", "body": "ASTNode | Statements", "else_body": "Elif | Else | None_"},
#         "For": {"iterable_names": "IdentList", "iterable": "ASTNode", "body": "ASTNode | Statements"},
#         "While": {"condition": "ASTNode", "body": "ASTNode | Statements"},
#         "StructDef": {"name": "Identifier", "items": "list[TypedName]"},
#         "EnumDef": {"name": "Identifier", "items": "list[Identifier]"},
#         "FunctionDef": {"name": "Identifier", "args": "list[TypedName]", "body": "ASTNode | Statements", "return_type": "Type_"},
#         "ImportAll": {},
#         "Import": {"module_identifier": "list[Identifier]", "import_objects": "IdentList | None_ | ImportAll"},
#     }
#     ret_str += ast_config_to_str(compound_statements_config)
#     ret_str += ast_config_to_str({"Start": {"body": "Statements | None_"}})

#     var_defs = {
#         "Literal": ["Int", "Float", "String", "True_", "False_", "None_"],
#         "Predicate": ["BoolQuantifier", "BinaryOp", "UnaryOp", "True_", "False_"],
#         "Primary": ["StructAccess", "FunctionCall", "Indexing", "Literal", "Enumeration", "Comprehension", "Identifier"],
#         "Expr": ["LambdaDef", "Quantifier", "Predicate", "BinaryOp", "UnaryOp", "ListOp", "Primary", "Identifier"],
#         "SimpleStmt": ["Expr", "Assignment", "ControlFlowStmt", "Import"],
#         "CompoundStmt": ["If", "For", "StructDef", "EnumDef", "FunctionDef"],
#     }

#     ret_str += add_vars_to_str(var_defs)
#     return ret_str


# if __name__ == "__main__":
#     # Overwrite ast_generated file each time
#     with open("C:/Users/hunta/Documents/GitHub/compilers/packages/simile_compiler/simile_compiler/ast_generated.py", "w") as f:
#         f.write(ast_generator())
