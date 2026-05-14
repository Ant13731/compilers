from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
import pathlib
from typing import Callable, ParamSpec, TypeVar, Protocol, cast
from functools import singledispatch, wraps

from src.mod.pipeline.scanner import Location
from src.mod.pipeline.parser import parse, ParseError
from src.mod.data import ast_
from src.mod.data.symbol_table import SymbolTable
from src.mod.data import types


def type_check(ast: ast_.ASTNode, symbol_table: SymbolTable) -> None:
    # TODO resolve types for assignments and the like
    return None


P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


class TypeJudgementFunction(Protocol[P, R_co]):
    typing_rule_ids: tuple[str, ...]

    # TODO: register each function and typing rule in a central dictionary so we can make sure we have all typing rules implemented!
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...


def typing_rule(*ids: str) -> Callable[[Callable[P, R_co]], TypeJudgementFunction[P, R_co]]:
    def decorator(func: Callable[P, R_co]) -> TypeJudgementFunction[P, R_co]:
        typed_func = cast(TypeJudgementFunction[P, R_co], func)
        typed_func.typing_rule_ids = ids
        return typed_func

    return decorator


@singledispatch
def synthesize_type(ast: ast_.ASTNode, symbol_table: SymbolTable) -> types.BaseType:
    raise NotImplementedError(f"Type checking not implemented for AST node of type {type(ast)} at location {ast.get_location()}")


@typing_rule("Fetch Identifier")
@synthesize_type.register
def _(ast: ast_.Symbol, symbol_table: SymbolTable) -> types.BaseType:
    symbol_info = symbol_table.lookup_symbol(ast.symbol_table_entry.id_, ast.symbol_table_entry.scope)
    if symbol_info.declared_type is None:
        raise types.SimileTypeError(f"Symbol table entry {ast.symbol_table_entry} does not have an assigned type during type resolution", ast)
    return symbol_info.declared_type


@typing_rule("Tuple Enumeration")
@synthesize_type.register
def _(ast: ast_.TupleSymbol, symbol_table: SymbolTable) -> types.BaseType:
    ast_types = [synthesize_type(item, symbol_table) for item in ast.items]
    return types.TupleType(tuple(ast_types))


@typing_rule("")
@synthesize_type.register
def _(ast: ast_.Int, symbol_table: SymbolTable) -> types.BaseType:
    trait_collection = types.TraitCollection()
    trait_collection.set_trait(types.LiteralTrait(ast))
    return types.IntType(trait_collection=trait_collection)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.Float, symbol_table: SymbolTable) -> types.BaseType:
    trait_collection = types.TraitCollection()
    trait_collection.set_trait(types.LiteralTrait(ast))
    return types.FloatType(trait_collection=trait_collection)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.String, symbol_table: SymbolTable) -> types.BaseType:
    trait_collection = types.TraitCollection()
    trait_collection.set_trait(types.LiteralTrait(ast))
    return types.StringType(trait_collection=trait_collection)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.True_, symbol_table: SymbolTable) -> types.BaseType:
    trait_collection = types.TraitCollection()
    trait_collection.set_trait(types.LiteralTrait(ast))
    return types.BoolType(trait_collection=trait_collection)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.False_, symbol_table: SymbolTable) -> types.BaseType:
    trait_collection = types.TraitCollection()
    trait_collection.set_trait(types.LiteralTrait(ast))
    return types.BoolType(trait_collection=trait_collection)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.None_, symbol_table: SymbolTable) -> types.BaseType:
    return types.NoneType_()


@typing_rule("Lambda Expression")
@synthesize_type.register
def _(ast: ast_.LambdaDef, symbol_table: SymbolTable) -> types.BaseType:
    arg_types = {}
    for arg in ast.params.items:
        assert isinstance(arg, ast_.Symbol), "LambdaDef parameters must be Symbols"
        # Symbols assertion implies no shadowing (I_i != I_j)
        arg_types[arg.symbol_table_entry] = synthesize_type(arg, symbol_table)

    assert isinstance(synthesize_type(ast.predicate, symbol_table), types.BoolType), "Predicate of LambdaDef must be of type BoolType"
    return_type = synthesize_type(ast.expression, symbol_table)

    # result = types.RelationType(left=types.TupleType(tuple(arg_types.values())), right=return_type)
    # result.apply_traits_from_relation_operator(ast_.RelationOperator.PARTIAL_FUNCTION)
    return types.ProcedureType(arg_types=arg_types, return_type=return_type)


@typing_rule("Records - Access")
@synthesize_type.register
def _(ast: ast_.RecordAccess, symbol_table: SymbolTable) -> types.BaseType:
    ast_type = synthesize_type(ast.record, symbol_table)
    if isinstance(ast_type, types.RecordType):
        return ast_type.access(ast.field_name.name)
    raise types.SimileTypeError(f"Cannot access field of non-record type {ast_type} in record access", ast)


@typing_rule("Command - Procedure Call", "Relation Operations - Function Call")
@synthesize_type.register
def _(ast: ast_.Call, symbol_table: SymbolTable) -> types.BaseType:
    ast_type = synthesize_type(ast.target, symbol_table)
    if isinstance(ast_type, types.ProcedureType):
        return ast_type.call([synthesize_type(arg, symbol_table) for arg in ast.args])
    if isinstance(ast_type, types.RelationType) and len(ast.args) == 1:
        return ast_type.function_call(synthesize_type(ast.args[0], symbol_table))
    raise types.SimileTypeError(f"Cannot call type {ast_type} (expected procedure or relation)", ast)


@typing_rule("Bag Operations - Image", "Relation Operations - Image")
@synthesize_type.register
def _(ast: ast_.Image, symbol_table: SymbolTable) -> types.BaseType:
    target_type = synthesize_type(ast.target, symbol_table)
    index_type = synthesize_type(ast.index, symbol_table)
    if isinstance(target_type, types.BagType):
        return target_type.image(index_type)
    if isinstance(target_type, types.RelationType):
        return target_type.image(index_type)
    raise types.SimileTypeError(f"Cannot take image of non-relation type {target_type}", ast)


# There should be no types or typed names
# @resolve_type.register
# def _(ast: ast_.Type_, symbol_table: SymbolTable) -> types.BaseType: ...
# @resolve_type.register
# def _(ast: ast_.TypedName, symbol_table: SymbolTable) -> types.BaseType: ...


@typing_rule("Command - return")
@synthesize_type.register
def _(ast: ast_.Return, symbol_table: SymbolTable) -> types.BaseType:
    return synthesize_type(ast.value, symbol_table)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.Assignment, symbol_table: SymbolTable) -> types.BaseType:
    return types.NoneType_()


@typing_rule("Command - break", "Command - continue", "Command - skip")
@synthesize_type.register
def _(ast: ast_.ControlFlowStmt, symbol_table: SymbolTable) -> types.BaseType:
    return types.NoneType_()


@typing_rule()
@synthesize_type.register
def _(ast: ast_.If | ast_.ElseIf | ast_.Else | ast_.For | ast_.While, symbol_table: SymbolTable) -> types.BaseType:
    return types.NoneType_()


@typing_rule()
@synthesize_type.register
def _(ast: ast_.ImportAll | ast_.Import | ast_.Start | ast_.Statements, symbol_table: SymbolTable) -> types.BaseType:
    return types.NoneType_()


@typing_rule()
@synthesize_type.register
def _(ast: ast_.RecordDefSymbol | ast_.ProcedureDefSymbol, symbol_table: SymbolTable) -> types.BaseType:
    return types.NoneType_()


@typing_rule("Binary Boolean Operations")
@synthesize_type.register
def _(ast: ast_.Implies, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.BoolType):
        raise types.SimileTypeError(f"Left operand of implies must be of type BoolType, got {left_type}", ast.left)
    return left_type.implies(right_type)


@typing_rule("Binary Boolean Operations")
@synthesize_type.register
def _(ast: ast_.Equivalent, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.BoolType):
        raise types.SimileTypeError(f"Left operand of equivalent must be of type BoolType, got {left_type}", ast.left)
    return left_type.equivalent(right_type)


@typing_rule("Binary Boolean Operations")
@synthesize_type.register
def _(ast: ast_.NotEquivalent, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.BoolType):
        raise types.SimileTypeError(f"Left operand of not equivalent must be of type BoolType, got {left_type}", ast.left)
    return left_type.not_equivalent(right_type)


@typing_rule("Bag Operations")
@synthesize_type.register
def _(ast: ast_.Add, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType | types.FloatType):
        raise types.SimileTypeError(f"Left operand of add must be of type IntType or FloatType, got {left_type}", ast.left)
    return left_type.add(right_type)


@typing_rule("Bag Operations")
@synthesize_type.register
def _(ast: ast_.Subtract, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType | types.FloatType):
        raise types.SimileTypeError(f"Left operand of subtract must be of type IntType or FloatType, got {left_type}", ast.left)
    return left_type.subtract(right_type)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.Multiply, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType | types.FloatType):
        raise types.SimileTypeError(f"Left operand of multiply must be of type IntType or FloatType, got {left_type}", ast.left)
    return left_type.multiply(right_type)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.Divide, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType | types.FloatType):
        raise types.SimileTypeError(f"Left operand of divide must be of type IntType or FloatType, got {left_type}", ast.left)
    return left_type.divide(right_type)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.IntDivide, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType):
        raise types.SimileTypeError(f"Left operand of int_divide must be of type IntType, got {left_type}", ast.left)
    return left_type.int_divide(right_type)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.Modulo, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType):
        raise types.SimileTypeError(f"Left operand of modulo must be of type IntType, got {left_type}", ast.left)
    return left_type.modulo(right_type)


@typing_rule()
@synthesize_type.register
def _(ast: ast_.Exponent, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType | types.FloatType):
        raise types.SimileTypeError(f"Left operand of exponent must be of type IntType or FloatType, got {left_type}", ast.left)
    return left_type.power(right_type)


@typing_rule("Ordering Operators")
@synthesize_type.register
def _(ast: ast_.LessThan, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType | types.FloatType):
        raise types.SimileTypeError(f"Left operand of less_than must be of type IntType or FloatType, got {left_type}", ast.left)
    return left_type.less_than(right_type)


@typing_rule("Ordering Operators")
@synthesize_type.register
def _(ast: ast_.LessThanOrEqual, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType | types.FloatType):
        raise types.SimileTypeError(f"Left operand of lessThanOrEqual must be of type IntType or FloatType, got {left_type}", ast.left)
    return left_type.less_than_equals(right_type)


@typing_rule("Ordering Operators")
@synthesize_type.register
def _(ast: ast_.GreaterThan, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType | types.FloatType):
        raise types.SimileTypeError(f"Left operand of greaterThan must be of type IntType or FloatType, got {left_type}", ast.left)
    return left_type.greater_than(right_type)


@typing_rule("Ordering Operators")
@synthesize_type.register
def _(ast: ast_.GreaterThanOrEqual, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    if not isinstance(left_type, types.IntType | types.FloatType):
        raise types.SimileTypeError(f"Left operand of greaterThanOrEqual must be of type IntType or FloatType, got {left_type}", ast.left)
    return left_type.greater_than_equals(right_type)


@typing_rule("Equals")
@synthesize_type.register
def _(ast: ast_.Equal, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    return left_type.equals(right_type)


@typing_rule("Equals")
@synthesize_type.register
def _(ast: ast_.NotEqual, symbol_table: SymbolTable) -> types.BaseType:
    left_type = synthesize_type(ast.left, symbol_table)
    right_type = synthesize_type(ast.right, symbol_table)
    return left_type.not_equals(right_type)


# @resolve_type.register
# def _(ast: ast_.Is, symbol_table: SymbolTable) -> types.BaseType:
# @resolve_type.register
# def _(ast: ast_.IsNot, symbol_table: SymbolTable) -> types.BaseType:


@typing_rule("Set Membership")
@synthesize_type.register
def _(ast: ast_.In, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Membership")
@synthesize_type.register
def _(ast: ast_.NotIn, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Operations", "Bag Operations")
@synthesize_type.register
def _(ast: ast_.Union, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Operations", "Bag Operations")
@synthesize_type.register
def _(ast: ast_.Intersection, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Operations")
@synthesize_type.register
def _(ast: ast_.Difference, symbol_table: SymbolTable) -> types.BaseType: ...


@typing_rule("Set Ordering Operations")
@synthesize_type.register
def _(ast: ast_.Subset, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Ordering Operations")
@synthesize_type.register
def _(ast: ast_.SubsetEq, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Ordering Operations")
@synthesize_type.register
def _(ast: ast_.Superset, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Ordering Operations")
@synthesize_type.register
def _(ast: ast_.SupersetEq, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Ordering Operations")
@synthesize_type.register
def _(ast: ast_.NotSubset, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Ordering Operations")
@synthesize_type.register
def _(ast: ast_.NotSubsetEq, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Ordering Operations")
@synthesize_type.register
def _(ast: ast_.NotSuperset, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Ordering Operations")
@synthesize_type.register
def _(ast: ast_.NotSupersetEq, symbol_table: SymbolTable) -> types.BaseType: ...


@typing_rule("Maplet")
@synthesize_type.register
def _(ast: ast_.Maplet, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.RelationOverriding, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.Composition, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Cartesian Product")
@synthesize_type.register
def _(ast: ast_.CartesianProduct, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Numerical Range")
@synthesize_type.register
def _(ast: ast_.Upto, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.Concat, symbol_table: SymbolTable) -> types.BaseType: ...


@typing_rule()
@synthesize_type.register
def _(ast: ast_.DomainSubtraction, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.DomainRestriction, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.RangeSubtraction, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.RangeRestriction, symbol_table: SymbolTable) -> types.BaseType: ...


@typing_rule()
@synthesize_type.register
def _(ast: ast_.Relation, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Total Relation")
@synthesize_type.register
def _(ast: ast_.TotalRelation, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Surjective Relation")
@synthesize_type.register
def _(ast: ast_.SurjectiveRelation, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Total Surjective Relation")
@synthesize_type.register
def _(ast: ast_.TotalSurjectiveRelation, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Partial Function")
@synthesize_type.register
def _(ast: ast_.PartialFunction, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Total Function")
@synthesize_type.register
def _(ast: ast_.TotalFunction, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Partial Injection")
@synthesize_type.register
def _(ast: ast_.PartialInjection, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Total Injection")
@synthesize_type.register
def _(ast: ast_.TotalInjection, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Partial Surjection")
@synthesize_type.register
def _(ast: ast_.PartialSurjection, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Total Surjection")
@synthesize_type.register
def _(ast: ast_.TotalSurjection, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Relation Subtype - Bijection")
@synthesize_type.register
def _(ast: ast_.Bijection, symbol_table: SymbolTable) -> types.BaseType: ...


@typing_rule("Boolean Operations - Negation")
@synthesize_type.register
def _(ast: ast_.Not, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.Negative, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Operations - Powerset")
@synthesize_type.register
def _(ast: ast_.Powerset, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.NonemptyPowerset, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.Inverse, symbol_table: SymbolTable) -> types.BaseType: ...


@typing_rule("Binary Boolean Operations")
@synthesize_type.register
def _(ast: ast_.And, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Binary Boolean Operations")
@synthesize_type.register
def _(ast: ast_.Or, symbol_table: SymbolTable) -> types.BaseType: ...


# @typing_rule("Forall")
# @synthesize_type.register
# def _(ast: ast_.Forall, symbol_table: SymbolTable) -> types.BaseType: ...
# @typing_rule("Exists")
# @synthesize_type.register
# def _(ast: ast_.Exists, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Forall")
@synthesize_type.register
def _(ast: ast_.QualifiedForall, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Exists")
@synthesize_type.register
def _(ast: ast_.QualifiedExists, symbol_table: SymbolTable) -> types.BaseType: ...


# @typing_rule()
# @synthesize_type.register
# def _(ast: ast_.UnionAll, symbol_table: SymbolTable) -> types.BaseType: ...
# @typing_rule()
# @synthesize_type.register
# def _(ast: ast_.IntersectionAll, symbol_table: SymbolTable) -> types.BaseType: ...
# @typing_rule()
# @synthesize_type.register
# def _(ast: ast_.Sum, symbol_table: SymbolTable) -> types.BaseType: ...
# @typing_rule()
# @synthesize_type.register
# def _(ast: ast_.Product, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("General Union")
@synthesize_type.register
def _(ast: ast_.QualifiedUnionAll, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("General Intersection")
@synthesize_type.register
def _(ast: ast_.QualifiedIntersectionAll, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.QualifiedSum, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.QualifiedProduct, symbol_table: SymbolTable) -> types.BaseType: ...


@typing_rule()
@synthesize_type.register
def _(ast: ast_.Break, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.Continue, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.Skip, symbol_table: SymbolTable) -> types.BaseType: ...


@typing_rule()
@synthesize_type.register
def _(ast: ast_.SequenceEnumeration, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.SetEnumeration, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.RelationEnumeration, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.BagEnumeration, symbol_table: SymbolTable) -> types.BaseType: ...


# @typing_rule()
# @synthesize_type.register
# def _(ast: ast_.SequenceComprehension, symbol_table: SymbolTable) -> types.BaseType: ...
# @typing_rule()
# @synthesize_type.register
# def _(ast: ast_.SetComprehension, symbol_table: SymbolTable) -> types.BaseType: ...
# @typing_rule()
# @synthesize_type.register
# def _(ast: ast_.RelationComprehension, symbol_table: SymbolTable) -> types.BaseType: ...
# @typing_rule()
# @synthesize_type.register
# def _(ast: ast_.BagComprehension, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Sequence Comprehension")
@synthesize_type.register
def _(ast: ast_.QualifiedSequenceComprehension, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Set Comprehension")
@synthesize_type.register
def _(ast: ast_.QualifiedSetComprehension, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule()
@synthesize_type.register
def _(ast: ast_.QualifiedRelationComprehension, symbol_table: SymbolTable) -> types.BaseType: ...
@typing_rule("Bag Comprehension")
@synthesize_type.register
def _(ast: ast_.QualifiedBagComprehension, symbol_table: SymbolTable) -> types.BaseType: ...
