from dataclasses import asdict
from copy import deepcopy

from src.mod.data import ast_
from src.mod.data.symbol_table import SymbolTable, SymbolTableIdentifierEntry
from src.mod.data.types import (
    BaseType,
    BoolType,
    GenericType,
    DeferToSymbolTable,
    StringType,
    IntType,
    FloatType,
    SetType,
    BagType,
    RelationType,
    SequenceType,
    TupleType,
    SimileTypeError,
    TypeOfType,
    ProcedureType,
    RecordType,
    EnumType,
    NoneType_,
    AnyType_,
)
from src.mod.data.traits import (
    Trait,
    TraitCollection,
    OrderableTrait,
    IterableTrait,
    LiteralTrait,
    DomainTrait,
    MinTrait,
    MaxTrait,
    SizeTrait,
    ImmutableTrait,
    TotalOnDomainTrait,
    TotalOnRangeTrait,
    ManyToOneTrait,
    OneToManyTrait,
    EmptyTrait,
    TotalTrait,
    UniqueElementsTrait,
    GenericBoundTrait,
)


class TypeAnnotationResolver:
    RESERVED_KEYWORDS_FOR_TYPES: list[str] = [
        "int",
        "float",
        "string",
        "bool",
        "set",
        "sequence",
        "bag",
        "relation",
        "generic",
        "tuple",
        "type",
        "enum",
        "ℤ",
        "ℕ",
        "ℕ₁",
    ]

    # limited form of the below type synthesizer (which should not need to encounter type annotations)
    # this should be accessible to populate_symbol_table though
    @classmethod
    def resolve_type_annotation(cls, ast: ast_.Type_ | ast_.ASTNode, symbol_table: SymbolTable) -> BaseType:
        resolved_type_params: list[BaseType] = []
        if isinstance(ast, ast_.Type_):
            resolved_type_params = [cls.resolve_type_annotation(type_param, symbol_table) for type_param in ast.generics]
            ast = ast.type_

        match ast:
            # Used while populating the symbol table
            case ast_.TupleLiteral(type_params):
                if len(resolved_type_params) != 0:
                    raise SimileTypeError("Cannot provide parameters to a tuple type")
                resolved_type_params = [cls.resolve_type_annotation(type_param, symbol_table) for type_param in type_params]
                return TupleType(resolved_type_params)
            case ast_.Identifier(symbol_table_name):
                symbol_table_entry = symbol_table.lookup_identifier(symbol_table_name)
                # special case to populate generic type ids with their identifier values as initialized
                declared_type = deepcopy(symbol_table_entry.declared_type)
                if isinstance(declared_type, GenericType):
                    declared_type.add_symbol_info(symbol_table_entry)
                return cls._resolve_generic_type_params(declared_type, resolved_type_params)
            # Used for type synthesis after the symbol table is populated
            case ast_.TupleSymbol(type_params):
                if len(resolved_type_params) != 0:
                    raise SimileTypeError("Cannot provide parameters to a tuple type")
                resolved_type_params = [cls.resolve_type_annotation(type_param, symbol_table) for type_param in type_params]
                return TupleType(resolved_type_params)
            case ast_.Symbol(symbol_table_entry):
                return cls._resolve_generic_type_params(symbol_table_entry.declared_type, resolved_type_params)
            # Special notation for relation types
            # TODO not allowed by parser for now?
            case ast_.RelationOp(left, right, op):
                if len(resolved_type_params) != 0:
                    raise SimileTypeError(f"Infix relation operator type annotation cannot have parameters, got {len(resolved_type_params)}: {resolved_type_params}", ast)
                left_type = cls.resolve_type_annotation(left, symbol_table)
                right_type = cls.resolve_type_annotation(right, symbol_table)
                rel_type = RelationType(left_type, right_type)
                rel_type.apply_traits_from_relation_operator(op)
                return rel_type
            # TODO record types (user identified types) with generics? disallow for now

        raise SimileTypeError(f"Failed to resolve type annotation", ast)

    @classmethod
    def _resolve_generic_type_params(cls, base_type: BaseType, type_params: list[BaseType]) -> BaseType:
        if isinstance(base_type, TupleType):
            raise SimileTypeError(
                f"Cannot apply generic type parameters to a tuple type. Use the tuple literal syntax instead. (got base type {base_type} with params {type_params})"
            )

        if isinstance(base_type, SequenceType):
            if len(type_params) != 1:
                raise SimileTypeError(f"Sequence type annotation must have exactly 1 parameter, got {len(type_params)}: {type_params}")
            return SequenceType(type_params[0])
        if isinstance(base_type, BagType):
            if len(type_params) != 1:
                raise SimileTypeError(f"Bag type annotation must have exactly 1 parameter, got {len(type_params)}: {type_params}")
            return BagType(type_params[0])
        if isinstance(base_type, RelationType):
            if len(type_params) != 2:
                raise SimileTypeError(f"Relation type annotation must have exactly 2 parameters, got {len(type_params)}: {type_params}")
            return RelationType(type_params[0], type_params[1])
        if isinstance(base_type, SetType):
            if len(type_params) != 1:
                raise SimileTypeError(f"Set type annotation must have exactly 1 parameter, got {len(type_params)}: {type_params}")
            return SetType(type_params[0])

        if isinstance(base_type, TypeOfType):
            if len(type_params) != 1:
                # ignore for now
                return base_type
                # raise SimileTypeError(f"TypeOfType type annotation must have exactly 1 parameter, got {len(type_params)}: {type_params}")
            return TypeOfType(type_params[0])
        if isinstance(base_type, ProcedureType):
            if len(type_params) != 2:
                raise SimileTypeError(
                    f"Procedure type annotation must have exactly 2 parameters (multiple arguments get grouped into a tuple), got {len(type_params)}: {type_params}",
                )
            if not isinstance(type_params[0], TupleType):
                raise SimileTypeError(
                    f"Procedure type annotation first parameter must be a tuple of argument types, got {type_params[0]}",
                )
            return ProcedureType(type_params[0], type_params[1])

        if isinstance(base_type, EnumType):
            if len(type_params) > 1:
                raise SimileTypeError(f"Generic type annotation must have either 0 or 1 parameters, got {len(type_params)}: {type_params}")
            if len(type_params) == 1:
                return EnumType(type_params[0])
            return EnumType()

        # TODO record types with generics? disallow for now...
        # Types that expect no generic params
        if len(type_params) == 0:
            return base_type

        raise SimileTypeError(f"Cannot apply generic type parameters to non-generic type: {base_type}")

    @classmethod
    def resolve_trait_annotation(cls, with_clause: ast_.ASTNode, symbol_table: SymbolTable) -> Trait:
        flag_only_traits = [
            OrderableTrait(),
            IterableTrait(),
            ImmutableTrait(),
            TotalOnDomainTrait(),
            TotalOnRangeTrait(),
            ManyToOneTrait(),
            OneToManyTrait(),
            EmptyTrait(),
            TotalTrait(),
            UniqueElementsTrait(),
        ]

        match with_clause:
            case ast_.Identifier(name):
                for trait_type in flag_only_traits:
                    if trait_type.name == name:
                        return trait_type
            # case BinaryOp(left, right, BinaryOperator.EQUAL):
            case ast_.BinaryOp(ast_.Identifier(LiteralTrait.name), right, ast_.BinaryOperator.EQUAL):
                return LiteralTrait(right)
            case ast_.BinaryOp(ast_.Identifier(DomainTrait.name), ast_.Enumeration(items, ast_.CollectionOperator.SET), ast_.BinaryOperator.EQUAL):
                return DomainTrait(items)
            case ast_.BinaryOp(ast_.Identifier(MinTrait.name), right, ast_.BinaryOperator.EQUAL):
                return MinTrait(right)
            case ast_.BinaryOp(ast_.Identifier(MaxTrait.name), right, ast_.BinaryOperator.EQUAL):
                return MaxTrait(right)
            case ast_.BinaryOp(ast_.Identifier(SizeTrait.name), ast_.Int(right), ast_.BinaryOperator.EQUAL):
                return SizeTrait(int(right))
            case ast_.BinaryOp(ast_.Identifier(GenericBoundTrait.name), right, ast_.BinaryOperator.EQUAL):
                generic_bound_type = cls.resolve_type_annotation(right, symbol_table)
                if generic_bound_type is None:
                    raise SimileTypeError(f"Generic bound trait must have a valid type annotation, got None", right)
                return GenericBoundTrait([generic_bound_type])
            # case ast_.Call(ast_.Identifier(RuntimeTrait.name), [trait]):
            #     resolved_trait = cls.resolve_trait_annotation(trait, symbol_table)
            #     if not isinstance(resolved_trait, Trait):
            #         raise SimileTypeError(f"Runtime trait must be applied to a valid trait, got {resolved_trait}", trait)
            #     return RuntimeTrait(resolved_trait)

        raise SimileTypeError(f"Unknown trait in with clause: {with_clause} (failed to convert ASTNode to Trait)", with_clause)

    @classmethod
    def resolve_trait_collection(cls, with_clauses: list[ast_.ASTNode], symbol_table: SymbolTable) -> TraitCollection:
        trait_collection = TraitCollection()
        seen_with_clause_trait_classes: list[type[Trait]] = []
        for clause in with_clauses:
            trait = cls.resolve_trait_annotation(clause, symbol_table)
            if not isinstance(trait, GenericBoundTrait) and any(isinstance(trait, seen_trait) for seen_trait in seen_with_clause_trait_classes):
                raise SimileTypeError(f"Trait {trait} cannot be defined twice. Already seen traits: {seen_with_clause_trait_classes}", clause)
            trait_collection.set_trait(trait)
            seen_with_clause_trait_classes.append(trait.__class__)
        return trait_collection
