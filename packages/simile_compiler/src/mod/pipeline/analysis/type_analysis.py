from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
import pathlib
from typing import Callable, Generic, OrderedDict, ParamSpec, TypeVar, Protocol, cast, Any, Sequence
from functools import singledispatch, singledispatchmethod, wraps, reduce

from src.mod.data.symbol_table.entry import SymbolTableIdentifierEntry
from src.mod.pipeline.analysis.type_resolver import TypeAnnotationResolver
from src.mod.pipeline.scanner import Location
from src.mod.pipeline.parser import parse, ParseError
from src.mod.data import ast_, types, traits
from src.mod.data.symbol_table import SymbolTable
from src.mod.data.types.typing_rule_decorator import typing_rule


def type_check(ast: ast_.ASTNode) -> None:
    # TODO resolve types for assignments and the like
    return None


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
        trait_collection = traits.TraitCollection()
        trait_collection.set_trait(traits.LiteralTrait(ast))
        return types.IntType(trait_collection=trait_collection)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Float) -> types.BaseType:
        trait_collection = traits.TraitCollection()
        trait_collection.set_trait(traits.LiteralTrait(ast))
        return types.FloatType(trait_collection=trait_collection)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.String) -> types.BaseType:
        trait_collection = traits.TraitCollection()
        trait_collection.set_trait(traits.LiteralTrait(ast))
        return types.StringType(trait_collection=trait_collection)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.True_) -> types.BaseType:
        trait_collection = traits.TraitCollection()
        trait_collection.set_trait(traits.LiteralTrait(ast))
        return types.BoolType(trait_collection=trait_collection)

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.False_) -> types.BaseType:
        trait_collection = traits.TraitCollection()
        trait_collection.set_trait(traits.LiteralTrait(ast))
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
        if isinstance(ast, ast_.BuiltinFunctionBase):
            return self.synthesize_type_builtin_function_call(ast)
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

    def create_relation_type(self, ast: ast_.RelationOp, relation_operator: ast_.RelationOperator) -> types.BaseType:
        left_type = self.synthesize_type(ast.left)
        right_type = self.synthesize_type(ast.right)
        relation_type = types.RelationType(left_type, right_type)
        relation_type.apply_traits_from_relation_operator(relation_operator)
        return relation_type

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Relation) -> types.BaseType:
        return self.create_relation_type(ast, ast_.RelationOperator.RELATION)

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

    # @typing_rule("Set Operations - Powerset")
    # @synthesize_type.register
    # def _(self, ast: ast_.Powerset) -> types.BaseType:
    #     val_type = self.synthesize_type(ast.value)
    #     return types.SetType.powerset(val_type)

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
    # @typing_rule("Binds (with generator)", "Binds with OR", "Binds with AND")

    @typing_rule("Quantifier", "Branching Quantifier")
    @synthesize_type.register
    def _(self, ast: ast_.QuantifierBody) -> types.BaseType:
        self._synthesize_type_generator_nest(ast.generators)

        if isinstance(ast.end_or_branch, ast_.ASTNode):
            return_type = self.synthesize_type(ast.end_or_branch)
            return types.QuantificationBodyIntermediary(return_type)

        branch_types: list[types.BaseType] = []
        for branch in ast.end_or_branch:
            branch_type = self.synthesize_type(branch)
            if not isinstance(branch_type, types.QuantificationBodyIntermediary):
                raise types.SimileTypeError(f"Branch of quantifier must be of type QuantificationBodyIntermediary. Got {branch_type}", branch)
            branch_types.append(branch_type.return_type)

        return_type = types.BaseType.max_type(branch_types)
        return types.QuantificationBodyIntermediary(return_type)

    @typing_rule("Generator Nest")
    def _synthesize_type_generator_nest(self, generators: Sequence[ast_.Generator | ast_.IterGenerator]) -> types.BaseType:
        if len(generators) == []:
            return types.NoneType_()

        generator_types = []
        for generator in generators:
            generator_types.append(self.synthesize_type(generator))

        return types.GeneratorIntermediary(types.TupleType(tuple(generator_types)))

    @typing_rule("Generator")
    @synthesize_type.register
    def _(self, ast: ast_.Generator) -> types.BaseType:
        set_type = self.synthesize_type(ast.set_)
        if not isinstance(set_type, types.SetType):
            raise types.SimileTypeError(f"Generator set must be of type SetType. Got {set_type}", ast.set_)

        assert isinstance(ast.identifiers, ast_.SymbolListTypes), "No identifiers allowed at this stage"
        self._structural_match(ast.identifiers, set_type.element_type)

        predicate_type = self.synthesize_type(ast.predicate)
        if not isinstance(predicate_type, types.BoolType):
            raise types.SimileTypeError(f"Predicate of quantifier must be of type BoolType. Got {predicate_type}", ast.predicate)

        return types.GeneratorIntermediary(set_type.element_type)

    @typing_rule("Structural Match", "Structural Match with Tuple")
    def _structural_match(self, identifiers: ast_.SymbolListTypes, element_type) -> None:
        match identifiers:
            case ast_.TupleSymbol(symbols):
                if not isinstance(element_type, types.TupleType):
                    raise types.SimileTypeError(f"Expected element type to be TupleType for structural match with tuple. Got {element_type}", identifiers)
                if len(symbols) != len(element_type.items):
                    raise types.SimileTypeError(
                        f"Expected number of symbols in tuple to match number of elements in tuple type. Got {len(symbols)} symbols and {len(element_type.items)} elements",
                        identifiers,
                    )
                for symbol, elem_type in zip(symbols, element_type.items):
                    self._structural_match(symbol, elem_type)
            case ast_.Symbol(symbol_table_entry):
                symbol_info = self.symbol_table.lookup_symbol(symbol_table_entry.id_, symbol_table_entry.scope)
                if symbol_info.declared_type is not None:
                    raise types.SimileTypeError("Symbol already has a declared type during structural match (expected to be unbound)", identifiers)
                symbol_info.declared_type = element_type
            case _:
                raise types.SimileTypeError("Unreachable state", identifiers)

    @typing_rule("Iter Generator")
    @synthesize_type.register
    def _(self, ast: ast_.IterGenerator) -> types.BaseType:
        generator_type = self.synthesize_type(ast.generator)
        if not isinstance(generator_type, types.GeneratorIntermediary):
            raise types.SimileTypeError(f"Iter generator must be of type GeneratorIntermediary. Got {generator_type}", ast.generator)
        for assignment in ast.assignments:
            self.synthesize_type(assignment)
        return generator_type

    @typing_rule("Iter (body)")
    @synthesize_type.register
    def _(self, ast: ast_.IterBodyEnd) -> types.BaseType:  # typecheck body, then return return_value's type
        self.synthesize_type(ast.body)
        return self.synthesize_type(ast.return_value)

    @typing_rule("Iter")
    @synthesize_type.register
    def _(self, ast: ast_.IterBody) -> types.BaseType:
        self._synthesize_type_generator_nest(ast.generators)

        if isinstance(ast.end_or_branch, ast_.ASTNode):
            return self.synthesize_type(ast.end_or_branch)

        branch_types: list[types.BaseType] = []
        for branch in ast.end_or_branch:
            branch_type = self.synthesize_type(branch)
            branch_types.append(branch_type)

        return_type = types.BaseType.max_type(branch_types)
        return return_type

    @typing_rule("Fold")
    @synthesize_type.register
    def _(self, ast: ast_.Fold) -> types.BaseType:
        self.synthesize_type(ast.accumulator_init)
        accumulator_init_var_type = self.synthesize_type(ast.accumulator_init.target)
        quantifier_body_type = self.synthesize_type(ast.quantifier_body)
        if not isinstance(quantifier_body_type, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of fold must be a quantification body. Got {quantifier_body_type}", ast.quantifier_body)

        fold_return_type = types.BaseType.max_type([accumulator_init_var_type, quantifier_body_type.return_type])
        return types.QuantificationBodyIntermediary(fold_return_type).fold()

    @typing_rule("Forall")
    @synthesize_type.register
    def _(self, ast: ast_.Forall) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of forall must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.forall()

    @typing_rule("Exists")
    @synthesize_type.register
    def _(self, ast: ast_.Exists) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of exists must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.exists()

    @typing_rule("General Union")
    @synthesize_type.register
    def _(self, ast: ast_.UnionAll) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of exists must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.union_all()

    @typing_rule("General Intersection")
    @synthesize_type.register
    def _(self, ast: ast_.IntersectionAll) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of exists must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.intersection_all()

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Sum) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of exists must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.sum()

    @typing_rule()
    @synthesize_type.register
    def _(self, ast: ast_.Product) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of exists must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.product()

    @typing_rule("Set Enumeration")
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
    def _(self, ast: ast_.SequenceComprehension) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of exists must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.sequence_comprehension()

    @typing_rule("Set Comprehension")
    @synthesize_type.register
    def _(self, ast: ast_.SetComprehension) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of exists must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.set_comprehension()

    @typing_rule("Relation Comprehension")  # TODO FIXME?
    @synthesize_type.register
    def _(self, ast: ast_.RelationComprehension) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of exists must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.relation_comprehension()

    @typing_rule("Bag Comprehension")
    @synthesize_type.register
    def _(self, ast: ast_.BagComprehension) -> types.BaseType:
        quantification_body = self.synthesize_type(ast.body)
        if not isinstance(quantification_body, types.QuantificationBodyIntermediary):
            raise types.SimileTypeError(f"Body of exists must be a quantification body. Got {quantification_body}", ast.body)
        return quantification_body.bag_comprehension()

    @singledispatchmethod
    def synthesize_type_builtin_function_call(self, ast: ast_.BuiltinFunctionBase) -> types.BaseType:
        raise NotImplementedError(f"Type synthesis not implemented for builtin function {type(ast)} at location {ast.get_location()}")

    @typing_rule("Built-in - Minimum")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncMin) -> types.BaseType:
        if len(ast.args) != 1:
            raise types.SimileTypeError(f"Minimum function takes exactly 1 argument. Got {len(ast.args)}", ast)

        collection_type = self.synthesize_type(ast.args[0])
        if not isinstance(collection_type, types.SetType):
            raise types.SimileTypeError(f"Argument to minimum must be a set. Got {collection_type}", ast.args[0])
        if collection_type.trait_collection.orderable_trait is None:
            raise types.SimileTypeError(f"Argument to minimum must be of an orderable type. Got {collection_type}", ast.args[0])
        return collection_type.min()

    @typing_rule("Built-in - Mapped Minimum")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncMapMin) -> types.BaseType:
        if len(ast.args) != 2:
            raise types.SimileTypeError(f"Minimum-with-map function takes exactly 2 arguments. Got {len(ast.args)}", ast)

        collection_type = self.synthesize_type(ast.args[0])
        if not isinstance(collection_type, types.SetType):
            raise types.SimileTypeError(f"Argument 1 to mapped minimum must be a set. Got {collection_type}", ast.args[0])

        mapping_func = self.synthesize_type(ast.args[1])
        if not isinstance(mapping_func, types.ProcedureType):
            raise types.SimileTypeError(f"Argument 2 to mapped minimum must be a procedure. Got {mapping_func}", ast.args[1])

        return collection_type.map_min(mapping_func)

    @typing_rule("Built-in - Maximum")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncMax) -> types.BaseType:
        if len(ast.args) != 1:
            raise types.SimileTypeError(f"Maximum function takes exactly 1 argument. Got {len(ast.args)}", ast)

        collection_type = self.synthesize_type(ast.args[0])
        if not isinstance(collection_type, types.SetType):
            raise types.SimileTypeError(f"Argument to maximum must be a set. Got {collection_type}", ast.args[0])
        if collection_type.trait_collection.orderable_trait is None:
            raise types.SimileTypeError(f"Argument to maximum must be of an orderable type. Got {collection_type}", ast.args[0])
        return collection_type.max()

    @typing_rule("Built-in - Mapped Maximum")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncMapMax) -> types.BaseType:
        if len(ast.args) != 2:
            raise types.SimileTypeError(f"Maximum-with-map function takes exactly 2 arguments. Got {len(ast.args)}", ast)

        collection_type = self.synthesize_type(ast.args[0])
        if not isinstance(collection_type, types.SetType):
            raise types.SimileTypeError(f"Argument 1 to mapped maximum must be a set. Got {collection_type}", ast.args[0])

        mapping_func = self.synthesize_type(ast.args[1])
        if not isinstance(mapping_func, types.ProcedureType):
            raise types.SimileTypeError(f"Argument 2 to mapped maximum must be a procedure. Got {mapping_func}", ast.args[1])

        return collection_type.map_max(mapping_func)

    @typing_rule("Built-in - Choice")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncChoice) -> types.BaseType:
        if len(ast.args) != 1:
            raise types.SimileTypeError(f"Choice function takes exactly 1 argument. Got {len(ast.args)}", ast)

        collection_type = self.synthesize_type(ast.args[0])
        if not isinstance(collection_type, types.SetType):
            raise types.SimileTypeError(f"Argument to choice must be a set. Got {collection_type}", ast.args[0])
        return collection_type.choice()

    @typing_rule("Built-in - Domain")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncDom) -> types.BaseType:
        if len(ast.args) != 1:
            raise types.SimileTypeError(f"Domain function takes exactly 1 argument. Got {len(ast.args)}", ast)

        relation_type = self.synthesize_type(ast.args[0])
        if not isinstance(relation_type, types.RelationType):
            raise types.SimileTypeError(f"Argument to domain must be a relation. Got {relation_type}", ast.args[0])
        return relation_type.domain()

    @typing_rule("Built-in - Range")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncRan) -> types.BaseType:
        if len(ast.args) != 1:
            raise types.SimileTypeError(f"Range function takes exactly 1 argument. Got {len(ast.args)}", ast)

        relation_type = self.synthesize_type(ast.args[0])
        if not isinstance(relation_type, types.RelationType):
            raise types.SimileTypeError(f"Argument to range must be a relation. Got {relation_type}", ast.args[0])
        return relation_type.range_()

    @typing_rule("Built-in - Cardinality")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncCard) -> types.BaseType:
        if len(ast.args) != 1:
            raise types.SimileTypeError(f"Cardinality function takes exactly 1 argument. Got {len(ast.args)}", ast)

        collection_type = self.synthesize_type(ast.args[0])
        if not isinstance(collection_type, types.SetType):
            raise types.SimileTypeError(f"Argument to cardinality must be a set. Got {collection_type}", ast.args[0])
        return collection_type.cardinality()

    @typing_rule("Built-in - Bag Size")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncSize) -> types.BaseType:
        if len(ast.args) != 1:
            raise types.SimileTypeError(f"Size function takes exactly 1 argument. Got {len(ast.args)}", ast)

        collection_type = self.synthesize_type(ast.args[0])
        if not isinstance(collection_type, types.BagType):
            raise types.SimileTypeError(f"Argument to size must be a bag. Got {collection_type}", ast.args[0])
        return collection_type.size()

    @typing_rule("Built-in - Sum")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncSum) -> types.BaseType:
        if len(ast.args) != 1:
            raise types.SimileTypeError(f"Sum function takes exactly one argument. Got {len(ast.args)}", ast)

        collection_type = self.synthesize_type(ast.args[0])
        if not isinstance(collection_type, types.SetType):
            raise types.SimileTypeError(f"Argument to sum must be a set. Got {collection_type}", ast.args[0])
        return collection_type.sum()

    @typing_rule("Built-in - Cast")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncCast) -> types.BaseType:
        if len(ast.args) != 2:
            raise types.SimileTypeError(f"Cast function takes exactly 2 arguments. Got {len(ast.args)}", ast)

        caster_type = self.synthesize_type(ast.args[0])
        value_type = self.synthesize_type(ast.args[1])
        return value_type.cast(caster_type)

    @typing_rule("Built-in - Cast With")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncCastWith) -> types.BaseType:
        if len(ast.args) != 3:
            raise types.SimileTypeError(f"Cast-with function takes exactly 3 arguments. Got {len(ast.args)}", ast)

        caster_type = self.synthesize_type(ast.args[0])
        value_type = self.synthesize_type(ast.args[1])
        traits_ast = ast.args[2]
        if not isinstance(traits_ast, ast_.SetEnumeration):
            raise types.SimileTypeError(f"Third argument to cast-with must be a set enumeration of traits. Got {type(traits_ast)}", traits_ast)

        traits = TypeAnnotationResolver.resolve_trait_collection(traits_ast.items, self.symbol_table)
        return value_type.cast(caster_type, traits)

    @typing_rule("Built-in - Print")
    @synthesize_type_builtin_function_call.register
    def _(self, ast: ast_.BuiltinFuncPrint) -> types.BaseType:
        if len(ast.args) != 1:
            raise types.SimileTypeError(f"Print function takes exactly 1 argument. Got {len(ast.args)}", ast)
        self.synthesize_type(ast.args[0])  # Just check that the argument is well-typed
        return types.NoneType_()
