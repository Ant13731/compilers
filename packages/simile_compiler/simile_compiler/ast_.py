from typing import TypeVar, Any

try:
    from .ast_generated import *
except ImportError:
    from ast_generated import *  # type: ignore


# def update_globals_with_ast_nodes(config: dict[str, Any]) -> None:
#     """Dynamically create AST node classes and update the global namespace."""
#     for name, fields_ in config.items():
#         if isinstance(fields_, str):
#             bases = (globals()[fields_],)  # Get the base class from globals
#             actual_fields = {}  # Clear away fields (no extra fields for configs that use a base class)
#         else:
#             assert isinstance(fields_, dict), "Fields must be a dictionary or a string reference to a base class"
#             actual_fields = fields_
#             bases = (ASTNode,)
#         globals()[name] = make_dataclass(name, actual_fields.items(), bases=bases)


# @lambda x: x()
# def ast_generator() -> None:
#     "Brings all AST nodes into the global namespace. Linter errors within this function can be ignored."

#     ast_primitives_config = {
#         "Int": {"value": int},
#         "Float": {"value": float},
#         "String": {"value": str},
#         "Bool": {"value": bool},
#         "None_": {},  # Represents a None literal
#         "Identifier": {"name": str},
#     }
#     update_globals_with_ast_nodes(ast_primitives_config)
#     op_config_base = {
#         "BinaryOp": {"left": ASTNode, "right": ASTNode},
#         "UnaryOp": {"value": ASTNode},
#         "ListOp": {"items": list[ASTNode]},
#     }
#     update_globals_with_ast_nodes(op_config_base)
#     op_config = {
#         "And": "ListOp",
#         "Or": "ListOp",
#         "Not": "UnaryOp",
#         "Implies": "BinaryOp",
#         "RevImplies": "BinaryOp",
#         "Equivalent": "BinaryOp",
#         "NotEquivalent": "BinaryOp",
#         #
#         "Add": "BinaryOp",
#         "Subtract": "BinaryOp",
#         "Multiply": "BinaryOp",
#         "Divide": "BinaryOp",
#         "Modulus": "BinaryOp",
#         # "Remainder": "BinaryOp",
#         "Exponent": "BinaryOp",
#         "Equal": "BinaryOp",
#         "NotEqual": "BinaryOp",
#         "LessThan": "BinaryOp",
#         "LessThanOrEqual": "BinaryOp",
#         "GreaterThan": "BinaryOp",
#         "GreaterThanOrEqual": "BinaryOp",
#         #
#         "Is": "BinaryOp",
#         "IsNot": "BinaryOp",
#         #
#         "In": "BinaryOp",
#         "NotIn": "BinaryOp",
#         "Union": "BinaryOp",
#         "Intersection": "BinaryOp",
#         "Difference": "BinaryOp",
#         "Subset": "BinaryOp",
#         "SubsetEq": "BinaryOp",
#         "Superset": "BinaryOp",
#         "SupersetEq": "BinaryOp",
#         "NotSubset": "BinaryOp",
#         "NotSubsetEq": "BinaryOp",
#         "NotSuperset": "BinaryOp",
#         "NotSupersetEq": "BinaryOp",
#         "UnionAll": "ListOp",
#         "IntersectionAll": "ListOp",
#         "Powerset": "UnaryOp",
#         #
#         "Maplet": "BinaryOp",
#         "RelationOverride": "BinaryOp",
#         "Composition": "BinaryOp",
#         "CartesianProduct": "BinaryOp",
#         "Inverse": "UnaryOp",
#         "DomainSubtraction": "BinaryOp",
#         "DomainRestriction": "BinaryOp",
#         "RangeSubtraction": "BinaryOp",
#         "RangeRestriction": "BinaryOp",
#         #
#         "RelationOp": "BinaryOp",
#         "TotalRelationOp": "BinaryOp",
#         "SurjectiveRelationOp": "BinaryOp",
#         "TotalSurjectiveRelation": "BinaryOp",
#         "PartialFunction": "BinaryOp",
#         "TotalFunction": "BinaryOp",
#         "PartialInjection": "BinaryOp",
#         "TotalInjection": "BinaryOp",
#         "PartialSurjection": "BinaryOp",
#         "TotalSurjection": "BinaryOp",
#         "Bijection": "BinaryOp",
#         #
#         "UpTo": "BinaryOp",
#     }
#     update_globals_with_ast_nodes(op_config)
#     caller_nodes_config_pre = {
#         "Arguments": {"items": list[ASTNode]},
#         "Type_": {"type_": ASTNode},
#         "Slice": {"items": list[ASTNode]},
#     }
#     update_globals_with_ast_nodes(caller_nodes_config_pre)
#     caller_nodes_config = {
#         "StructAccess": {"struct": ASTNode, "field_name": Identifier},  # type: ignore # noqa
#         "FunctionCall": {"function_name": ASTNode, "args": Arguments},  # type: ignore # noqa
#         "TypedName": {"name": Identifier, "type_": Type_ | None_},  # type: ignore # noqa
#         "Indexing": {"target": ASTNode, "index": ASTNode | Slice | None_},  # type: ignore # noqa
#     }
#     update_globals_with_ast_nodes(caller_nodes_config)
#     update_globals_with_ast_nodes({"ArgDef": {"items": list[TypedName]}})  # type: ignore # noqa
#     statements_config = {
#         "Assignment": {"target": ASTNode, "value": ASTNode},
#         "Return": {"value": ASTNode | None_},  # type: ignore # noqa
#         "LambdaDef": {"args": ArgDef, "body": ASTNode},  # type: ignore # noqa
#         "Break": {},
#         "Continue": {},
#         "Pass": {},
#     }
#     update_globals_with_ast_nodes(statements_config)
#     update_globals_with_ast_nodes({"Statements": {"items": list[ASTNode]}})
#     update_globals_with_ast_nodes({"Else": {"body": ASTNode | Statements}})  # type: ignore # noqa
#     compound_statements_config = {
#         "If": {"condition": ASTNode, "body": ASTNode | Statements, "else_body": Else | None_},  # type: ignore # noqa
#         # "Elif": {"condition": ASTNode, "body": ASTNode | Statements, "else_body": Elif | Else | None_},
#         "For": {"iterable_names": list[Identifier], "iterable": ASTNode, "body": ASTNode | Statements},  # type: ignore # noqa
#         "StructDef": {"name": Identifier, "items": list[TypedName]},  # type: ignore # noqa
#         "EnumDef": {"name": Identifier, "items": list[Identifier]},  # type: ignore # noqa
#         "FunctionDef": {"name": Identifier, "args": ArgDef, "body": ASTNode | Statements, "return_type": Type_ | None_},  # type: ignore # noqa
#     }
#     update_globals_with_ast_nodes(compound_statements_config)
#     update_globals_with_ast_nodes({"KeyPair": {"key": ASTNode, "value": ASTNode}})
#     complex_literals_config_base = {
#         "SeqLike": {"items": list[ASTNode]},
#         "MapLike": {"items": list[KeyPair]},  # type: ignore # noqa
#     }
#     update_globals_with_ast_nodes(complex_literals_config_base)
#     complex_literals_config = {
#         "SequenceLiteral": "SeqLike",
#         "SetLiteral": "SeqLike",
#         "BagLiteral": "MapLike",
#         "MappingLiteral": "MapLike",
#     }
#     update_globals_with_ast_nodes(complex_literals_config)
#     comprehension_config_base = {
#         "SeqLikeComprehension": {"bound_identifiers": list[Identifier], "generator": ASTNode, "mapping_expression": ASTNode | None_},  # type: ignore # noqa
#         "MapLikeComprehension": {"bound_identifiers": list[KeyPair], "generator": ASTNode, "mapping_expression": ASTNode | None_},  # type: ignore # noqa
#     }
#     update_globals_with_ast_nodes(comprehension_config_base)
#     comprehension_config = {
#         "SequenceComprehension": "SeqLikeComprehension",
#         "SetComprehension": "SeqLikeComprehension",
#         "BagComprehension": "SeqLikeComprehension",
#         "MappingComprehension": "MapLikeComprehension",
#     }
#     update_globals_with_ast_nodes(comprehension_config)
#     update_globals_with_ast_nodes({"Start": {"body": Statements | None_}})  # type: ignore # noqa

#     PrimitiveLiteral = [Int, Float, String, Bool, None_]  # type: ignore # noqa
#     ComplexLiteral = [SequenceLiteral, SetLiteral, BagLiteral, MappingLiteral]  # type: ignore # noqa
#     ComplexComprehension = [SequenceComprehension, SetComprehension, BagComprehension, MappingComprehension]  # type: ignore # noqa

#     PrimaryStmt = [StructAccess, FunctionCall, Indexing, *PrimitiveLiteral, *ComplexLiteral, *ComplexComprehension, Identifier]  # type: ignore # noqa
#     EquivalenceStmt = [BinaryOp, UnaryOp, ListOp, PrimaryStmt]  # type: ignore # noqa
#     Expr = [*EquivalenceStmt, LambdaDef]  # type: ignore # noqa
#     Atom = [Identifier, *Expr]  # type: ignore # noqa

#     SimpleStmt = [*Expr, Assignment, Return, Break, Continue, Pass]  # type: ignore # noqa
#     CompoundStmt = [If, For, StructDef, EnumDef, FunctionDef]  # type: ignore # noqa


# ListOp should be in the namespace after ast_generator() is called
L = TypeVar("L", bound=ListOp)  # type: ignore # noqa


def flatten_and_join(obj_lst: list[Any], type_: type[L]) -> L:
    """Flatten a list of objects and join them with the given ListOp."""
    flattened_objs = []
    for obj in obj_lst:
        if isinstance(obj, type_):
            flattened_objs += obj.items
        else:
            flattened_objs.append(obj)
    return type_(items=flattened_objs)
