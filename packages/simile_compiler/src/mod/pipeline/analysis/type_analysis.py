from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
import pathlib
from typing import Callable, ParamSpec, TypeVar, Protocol, cast, Any, Sequence
from functools import singledispatch, singledispatchmethod, wraps, reduce

from src.mod.pipeline.scanner import Location
from src.mod.pipeline.parser import parse, ParseError
from src.mod.data import ast_
from src.mod.data.symbol_table import SymbolTable
from src.mod.data import types


def type_check(ast: ast_.ASTNode) -> None:
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


@dataclass
class TypeSynthesizer:
    symbol_table: SymbolTable

    def synthesize_type_binary[T](
        self,
        ast: ast_.BinaryOp | ast_.RelationOp,
        expected_left_type: type[T],
        operation_as_type_func: Callable[[T, types.BaseType], types.BaseType],
    ) -> types.BaseType:
        left_type = self.synthesize_type(ast.left, self.symbol_table)
        right_type = self.synthesize_type(ast.right, self.symbol_table)
        if not isinstance(left_type, expected_left_type):
            raise types.SimileTypeError(f"Left operand of {operation_as_type_func.__name__} must be of type {expected_left_type}, got {left_type}", ast.left)
        return operation_as_type_func(left_type, right_type)

    def synthesize_type_binary_right_as_base[T](
        self,
        ast: ast_.BinaryOp | ast_.RelationOp,
        expected_right_type: type[T],
        operation_as_type_func: Callable[[T, types.BaseType], types.BaseType],
    ) -> types.BaseType:
        left_type = self.synthesize_type(ast.left, self.symbol_table)
        right_type = self.synthesize_type(ast.right, self.symbol_table)
        if not isinstance(right_type, expected_right_type):
            raise types.SimileTypeError(f"Right operand of {operation_as_type_func.__name__} must be of type {expected_right_type}, got {right_type}", ast.right)
        return operation_as_type_func(right_type, left_type)

    def synthesize_type_multiple_binary_paths[T](
        self, type_resolution_funcs: Sequence[tuple[ast_.BinaryOp | ast_.RelationOp, type[T] | Any, Callable[[T, types.BaseType], types.BaseType]] | Any]
    ) -> types.BaseType:
        tries: list[types.SimileTypeError] = []
        for ast, expected_left_type, op_as_type_func in type_resolution_funcs:
            try:
                return self.synthesize_type_binary(ast, expected_left_type, op_as_type_func)
            except types.SimileTypeError as e:
                tries.append(e)
        raise types.SimileTypeError(f"All attempted type resolution paths failed. Got: {'\n'.join(map(str,tries))}")

    @singledispatchmethod
    def synthesize_type(self, ast: ast_.ASTNode) -> types.BaseType:
        raise NotImplementedError(f"Type checking not implemented for AST node of type {type(ast)} at location {ast.get_location()}")

    @typing_rule("Fetch Identifier")
    @synthesize_type.register
    def _(self, ast: ast_.Symbol) -> types.BaseType:
        symbol_info = self.symbol_table.lookup_symbol(ast.symbol_table_entry.id_, ast.symbol_table_entry.scope)
        if symbol_info.declared_type is None:
            raise types.SimileTypeError(f"Symbol table entry {ast.symbol_table_entry} does not have an assigned type during type resolution", ast)
        return symbol_info.declared_type

    @typing_rule("Tuple Enumeration")
    @synthesize_type.register
    def _(self, ast: ast_.TupleSymbol) -> types.BaseType:
        ast_types = [self.synthesize_type(item) for item in ast.items]
        return types.TupleType(tuple(ast_types))

    @typing_rule("")
    @synthesize_type.register
    def _(self, ast: ast_.Int) -> types.BaseType:
        trait_collection = types.TraitCollection()
        trait_collection.set_trait(types.LiteralTrait(ast))
        return types.IntType(trait_collection=trait_collection)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Float) -> types.BaseType:
        trait_collection = types.TraitCollection()
        trait_collection.set_trait(types.LiteralTrait(ast))
        return types.FloatType(trait_collection=trait_collection)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.String) -> types.BaseType:
        trait_collection = types.TraitCollection()
        trait_collection.set_trait(types.LiteralTrait(ast))
        return types.StringType(trait_collection=trait_collection)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.True_) -> types.BaseType:
        trait_collection = types.TraitCollection()
        trait_collection.set_trait(types.LiteralTrait(ast))
        return types.BoolType(trait_collection=trait_collection)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.False_) -> types.BaseType:
        trait_collection = types.TraitCollection()
        trait_collection.set_trait(types.LiteralTrait(ast))
        return types.BoolType(trait_collection=trait_collection)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.None_) -> types.BaseType:
        return types.NoneType_()

    @typing_rule("Lambda Expression")
    @synthesize_type.register
    def _(self, ast: ast_.LambdaDef) -> types.BaseType:
        arg_types = {}
        for arg in ast.params.items:
            assert isinstance(arg, ast_.Symbol), "LambdaDef parameters must be Symbols"
            # Symbols assertion implies no shadowing (I_i != I_j)
            arg_types[arg.symbol_table_entry] = self.synthesize_type(arg)

        assert isinstance(self.synthesize_type(ast.predicate), types.BoolType), "Predicate of LambdaDef must be of type BoolType"
        return_type = self.synthesize_type(ast.expression)

        # result = types.RelationType(left=types.TupleType(tuple(arg_types.values())), right=return_type)
        # result.apply_traits_from_relation_operator(ast_.RelationOperator.PARTIAL_FUNCTION)
        return types.ProcedureType(arg_types=arg_types, return_type=return_type)

    @typing_rule("Records - Access")
    @synthesize_type.register
    def _(self, ast: ast_.RecordAccess) -> types.BaseType:
        ast_type = self.synthesize_type(ast.record)
        if isinstance(ast_type, types.RecordType):
            return ast_type.access(ast.field_name.name)
        raise types.SimileTypeError(f"Cannot access field of non-record type {ast_type} in record access", ast)

    @typing_rule("Command - Procedure Call", "Relation Operations - Function Call")
    @synthesize_type.register
    def _(self, ast: ast_.Call) -> types.BaseType:
        ast_type = self.synthesize_type(ast.target)
        if isinstance(ast_type, types.ProcedureType):
            return ast_type.call([self.synthesize_type(arg) for arg in ast.args])
        if isinstance(ast_type, types.RelationType) and len(ast.args) == 1:
            return ast_type.function_call(self.synthesize_type(ast.args[0]))
        raise types.SimileTypeError(f"Cannot call type {ast_type} (expected procedure or relation)", ast)

    @typing_rule("Bag Operations - Image", "Relation Operations - Image")
    @synthesize_type.register
    def _(self, ast: ast_.Image) -> types.BaseType:
        target_type = self.synthesize_type(ast.target)
        index_type = self.synthesize_type(ast.index)
        if isinstance(target_type, types.BagType):
            return target_type.image(index_type)
        if isinstance(target_type, types.RelationType):
            return target_type.image(index_type)
        raise types.SimileTypeError(f"Cannot take image of non-relation type {target_type}", ast)

    # There should be no types or typed names
    # @resolve_type.register
    # def _(self, ast: ast_.Type_) -> types.BaseType: ...
    # @resolve_type.register
    # def _(self, ast: ast_.TypedName) -> types.BaseType: ...

    @typing_rule("Command - return")
    @synthesize_type.register
    def _(self, ast: ast_.Return) -> types.BaseType:
        return self.synthesize_type(ast.value)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Assignment) -> types.BaseType:
        return types.NoneType_()

    @typing_rule("Command - break", "Command - continue", "Command - skip")
    @synthesize_type.register
    def _(self, ast: ast_.ControlFlowStmt) -> types.BaseType:
        return types.NoneType_()

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.If | ast_.ElseIf | ast_.Else | ast_.For | ast_.While) -> types.BaseType:
        return types.NoneType_()

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.ImportAll | ast_.Import | ast_.Start | ast_.Statements) -> types.BaseType:
        return types.NoneType_()

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.RecordDefSymbol | ast_.ProcedureDefSymbol) -> types.BaseType:
        return types.NoneType_()

    @typing_rule("Binary Boolean Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Implies) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.BoolType, types.BoolType.implies)

    @typing_rule("Binary Boolean Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Equivalent) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.BoolType, types.BoolType.equivalent)

    @typing_rule("Binary Boolean Operations")
    @synthesize_type.register
    def _(self, ast: ast_.NotEquivalent) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.BoolType, types.BoolType.not_equivalent)

    @typing_rule("Bag Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Add) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.IntType, types.IntType.add),
                (ast, types.FloatType, types.FloatType.add),
                (ast, types.BagType, types.BagType.bag_add),
            ]
        )

    @typing_rule("Bag Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Subtract) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.IntType, types.IntType.subtract),
                (ast, types.FloatType, types.FloatType.subtract),
                (ast, types.BagType, types.BagType.bag_difference),
            ]
        )

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Multiply) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.IntType, types.IntType.multiply),
                (ast, types.FloatType, types.FloatType.multiply),
            ]
        )

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Divide) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.IntType, types.IntType.divide),
                (ast, types.FloatType, types.FloatType.divide),
            ]
        )

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.IntDivide) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.IntType, types.IntType.int_divide)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Modulo) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.IntType, types.IntType.modulo)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Exponent) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.IntType, types.IntType.power),
                (ast, types.FloatType, types.FloatType.power),
            ]
        )

    @typing_rule("Ordering Operators")
    @synthesize_type.register
    def _(self, ast: ast_.LessThan) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.IntType, types.IntType.less_than),
                (ast, types.FloatType, types.FloatType.less_than),
            ]
        )

    @typing_rule("Ordering Operators")
    @synthesize_type.register
    def _(self, ast: ast_.LessThanOrEqual) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.IntType, types.IntType.less_than_equals),
                (ast, types.FloatType, types.FloatType.less_than_equals),
            ]
        )

    @typing_rule("Ordering Operators")
    @synthesize_type.register
    def _(self, ast: ast_.GreaterThan) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.IntType, types.IntType.greater_than),
                (ast, types.FloatType, types.FloatType.greater_than),
            ]
        )

    @typing_rule("Ordering Operators")
    @synthesize_type.register
    def _(self, ast: ast_.GreaterThanOrEqual) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.IntType, types.IntType.greater_than_equals),
                (ast, types.FloatType, types.FloatType.greater_than_equals),
            ]
        )

    @typing_rule("Equals")
    @synthesize_type.register
    def _(self, ast: ast_.Equal) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.BaseType, types.BaseType.equals)

    @typing_rule("Equals")
    @synthesize_type.register
    def _(self, ast: ast_.NotEqual) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.BaseType, types.BaseType.not_equals)

    # @resolve_type.register
    # def _(self, ast: ast_.Is) -> types.BaseType:
    # @resolve_type.register
    # def _(self, ast: ast_.IsNot) -> types.BaseType:

    @typing_rule("Set Membership")
    @synthesize_type.register
    def _(self, ast: ast_.In) -> types.BaseType:
        return self.synthesize_type_binary_right_as_base(ast, types.SetType, types.SetType.in_)

    @typing_rule("Set Membership")
    @synthesize_type.register
    def _(self, ast: ast_.NotIn) -> types.BaseType:
        return self.synthesize_type_binary_right_as_base(ast, types.SetType, types.SetType.not_in)

    @typing_rule("Set Operations", "Bag Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Union) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.SetType, types.SetType.union),
                (ast, types.BagType, types.BagType.bag_union),
            ]
        )

    @typing_rule("Set Operations", "Bag Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Intersection) -> types.BaseType:
        return self.synthesize_type_multiple_binary_paths(
            [
                (ast, types.SetType, types.SetType.intersection),
                (ast, types.BagType, types.BagType.bag_intersection),
            ]
        )

    @typing_rule("Set Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Difference) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.difference)

    @typing_rule("Set Ordering Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Subset) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.is_subset)

    @typing_rule("Set Ordering Operations")
    @synthesize_type.register
    def _(self, ast: ast_.SubsetEq) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.is_subset_equals)

    @typing_rule("Set Ordering Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Superset) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.is_superset)

    @typing_rule("Set Ordering Operations")
    @synthesize_type.register
    def _(self, ast: ast_.SupersetEq) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.is_superset_equals)

    @typing_rule("Set Ordering Operations")
    @synthesize_type.register
    def _(self, ast: ast_.NotSubset) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.not_is_subset)

    @typing_rule("Set Ordering Operations")
    @synthesize_type.register
    def _(self, ast: ast_.NotSubsetEq) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.not_is_subset_equals)

    @typing_rule("Set Ordering Operations")
    @synthesize_type.register
    def _(self, ast: ast_.NotSuperset) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.not_is_superset)

    @typing_rule("Set Ordering Operations")
    @synthesize_type.register
    def _(self, ast: ast_.NotSupersetEq) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.not_is_superset_equals)

    @typing_rule("Maplet")
    @synthesize_type.register
    def _(self, ast: ast_.Maplet) -> types.BaseType:
        return types.PairType.maplet(self.synthesize_type(ast.left), self.synthesize_type(ast.right))

    @typing_rule("Relation Operations - Overriding")
    @synthesize_type.register
    def _(self, ast: ast_.RelationOverriding) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.RelationType, types.RelationType.overriding)

    @typing_rule("Relation Operations - Composition")
    @synthesize_type.register
    def _(self, ast: ast_.Composition) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.RelationType, types.RelationType.composition)

    @typing_rule("Cartesian Product")
    @synthesize_type.register
    def _(self, ast: ast_.CartesianProduct) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SetType, types.SetType.cartesian_product)

    @typing_rule("Numerical Range")
    @synthesize_type.register
    def _(self, ast: ast_.Upto) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.IntType, types.IntType.upto)

    @typing_rule("Sequence Operations - Concatenation")
    @synthesize_type.register
    def _(self, ast: ast_.Concat) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.SequenceType, types.SequenceType.concat)

    @typing_rule("Relation Operations - Domain Subtraction")
    @synthesize_type.register
    def _(self, ast: ast_.DomainSubtraction) -> types.BaseType:
        return self.synthesize_type_binary_right_as_base(ast, types.RelationType, types.RelationType.domain_subtraction)

    @typing_rule("Relation Operations - Domain Restriction")
    @synthesize_type.register
    def _(self, ast: ast_.DomainRestriction) -> types.BaseType:
        return self.synthesize_type_binary_right_as_base(ast, types.RelationType, types.RelationType.domain_restriction)

    @typing_rule("Relation Operations - Range Subtraction")
    @synthesize_type.register
    def _(self, ast: ast_.RangeSubtraction) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.RelationType, types.RelationType.range_subtraction)

    @typing_rule("Relation Operations - Range Restriction")
    @synthesize_type.register
    def _(self, ast: ast_.RangeRestriction) -> types.BaseType:
        return self.synthesize_type_binary(ast, types.RelationType, types.RelationType.range_restriction)

    def create_relation_type(self, ast: ast_.RelationOp, relation_operator: ast_.RelationOperator | None) -> types.BaseType:
        left_type = self.synthesize_type(ast.left)
        right_type = self.synthesize_type(ast.right)
        relation_type = types.RelationType(left_type, right_type)
        if relation_operator:
            relation_type.apply_traits_from_relation_operator(relation_operator)
        return relation_type

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Relation) -> types.BaseType:
        return self.create_relation_type(ast, None)

    @typing_rule("Relation Subtype - Total Relation")
    @synthesize_type.register
    def _(self, ast: ast_.TotalRelation) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.TOTAL_RELATION)

    @typing_rule("Relation Subtype - Surjective Relation")
    @synthesize_type.register
    def _(self, ast: ast_.SurjectiveRelation) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.SURJECTIVE_RELATION)

    @typing_rule("Relation Subtype - Total Surjective Relation")
    @synthesize_type.register
    def _(self, ast: ast_.TotalSurjectiveRelation) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.TOTAL_SURJECTIVE_RELATION)

    @typing_rule("Relation Subtype - Partial Function")
    @synthesize_type.register
    def _(self, ast: ast_.PartialFunction) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.PARTIAL_FUNCTION)

    @typing_rule("Relation Subtype - Total Function")
    @synthesize_type.register
    def _(self, ast: ast_.TotalFunction) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.TOTAL_FUNCTION)

    @typing_rule("Relation Subtype - Partial Injection")
    @synthesize_type.register
    def _(self, ast: ast_.PartialInjection) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.PARTIAL_INJECTION)

    @typing_rule("Relation Subtype - Total Injection")
    @synthesize_type.register
    def _(self, ast: ast_.TotalInjection) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.TOTAL_INJECTION)

    @typing_rule("Relation Subtype - Partial Surjection")
    @synthesize_type.register
    def _(self, ast: ast_.PartialSurjection) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.PARTIAL_SURJECTION)

    @typing_rule("Relation Subtype - Total Surjection")
    @synthesize_type.register
    def _(self, ast: ast_.TotalSurjection) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.TOTAL_SURJECTION)

    @typing_rule("Relation Subtype - Bijection")
    @synthesize_type.register
    def _(self, ast: ast_.Bijection) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.BIJECTION)

    @typing_rule("Boolean Operations - Negation")
    @synthesize_type.register
    def _(self, ast: ast_.Not) -> types.BaseType:
        val_type = self.synthesize_type(ast.value)
        if not isinstance(val_type, types.BoolType):
            raise types.SimileTypeError(f"Operand of negation must be of type BoolType. Got {val_type}", ast.value)
        return val_type.not_()

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Negative) -> types.BaseType:
        val_type = self.synthesize_type(ast.value)
        if not isinstance(val_type, types.IntType | types.FloatType):
            raise types.SimileTypeError(f"Operand of negative must be of type IntType or BoolType. Got {val_type}", ast.value)
        return val_type.negate()

    @typing_rule("Set Operations - Powerset")
    @synthesize_type.register
    def _(self, ast: ast_.Powerset) -> types.BaseType:
        val_type = self.synthesize_type(ast.value)
        return types.SetType.powerset(val_type)

    # @typing_rule()
    # @synthesize_type.register
    # def _(self, ast: ast_.NonemptyPowerset) -> types.BaseType: ...
    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Inverse) -> types.BaseType:
        val_type = self.synthesize_type(ast.value)
        if not isinstance(val_type, types.RelationType):
            raise types.SimileTypeError(f"Operand of inverse must be of type RelationType. Got {val_type}", ast.value)
        return val_type.inverse()

    @typing_rule("Binary Boolean Operations")
    @synthesize_type.register
    def _(self, ast: ast_.And) -> types.BaseType:
        item_types = []
        for item in ast.items:
            item_type = self.synthesize_type(item)
            if not isinstance(item_type, types.BoolType):
                raise types.SimileTypeError(f"Operand of and_ must be of type BoolType. Got {item_type}", item)
            item_types.append(item_type)
        return reduce(types.BoolType.and_, item_types)

    @typing_rule("Binary Boolean Operations")
    @synthesize_type.register
    def _(self, ast: ast_.Or) -> types.BaseType:
        item_types = []
        for item in ast.items:
            item_type = self.synthesize_type(item)
            if not isinstance(item_type, types.BoolType):
                raise types.SimileTypeError(f"Operand of or_ must be of type BoolType. Got {item_type}", item)
            item_types.append(item_type)
        return reduce(types.BoolType.or_, item_types)

    # @typing_rule("Forall")
    # @synthesize_type.register
    # def _(self, ast: ast_.Forall) -> types.BaseType: ...
    # @typing_rule("Exists")
    # @synthesize_type.register
    # def _(self, ast: ast_.Exists) -> types.BaseType: ...
    @typing_rule("Forall")
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedForall) -> types.BaseType: ...
    @typing_rule("Exists")
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedExists) -> types.BaseType: ...

    # @typing_rule()
    # @synthesize_type.register
    # def _(self, ast: ast_.UnionAll) -> types.BaseType: ...
    # @typing_rule()
    # @synthesize_type.register
    # def _(self, ast: ast_.IntersectionAll) -> types.BaseType: ...
    # @typing_rule()
    # @synthesize_type.register
    # def _(self, ast: ast_.Sum) -> types.BaseType: ...
    # @typing_rule()
    # @synthesize_type.register
    # def _(self, ast: ast_.Product) -> types.BaseType: ...
    @typing_rule("General Union")
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedUnionAll) -> types.BaseType: ...
    @typing_rule("General Intersection")
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedIntersectionAll) -> types.BaseType: ...
    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedSum) -> types.BaseType: ...
    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedProduct) -> types.BaseType: ...

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.SequenceEnumeration) -> types.BaseType:
        item_types = list(map(self.synthesize_type, ast.items))
        return types.SequenceType.enumeration(item_types)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.SetEnumeration) -> types.BaseType:
        item_types = list(map(self.synthesize_type, ast.items))
        return types.SetType.enumeration(item_types)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.RelationEnumeration) -> types.BaseType:
        item_types = list(map(self.synthesize_type, ast.items))
        return types.RelationType.enumeration(item_types)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.BagEnumeration) -> types.BaseType:
        item_types = list(map(self.synthesize_type, ast.items))
        return types.BagType.enumeration(item_types)

    # @typing_rule()
    # @synthesize_type.register
    # def _(self, ast: ast_.SequenceComprehension) -> types.BaseType: ...
    # @typing_rule()
    # @synthesize_type.register
    # def _(self, ast: ast_.SetComprehension) -> types.BaseType: ...
    # @typing_rule()
    # @synthesize_type.register
    # def _(self, ast: ast_.RelationComprehension) -> types.BaseType: ...
    # @typing_rule()
    # @synthesize_type.register
    # def _(self, ast: ast_.BagComprehension) -> types.BaseType: ...
    @typing_rule("Sequence Comprehension")
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedSequenceComprehension) -> types.BaseType: ...
    @typing_rule("Set Comprehension")
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedSetComprehension) -> types.BaseType: ...
    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedRelationComprehension) -> types.BaseType: ...
    @typing_rule("Bag Comprehension")
    @synthesize_type.register
    def _(self, ast: ast_.QualifiedBagComprehension) -> types.BaseType: ...
