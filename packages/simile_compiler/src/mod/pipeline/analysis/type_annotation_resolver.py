from src.mod.data import ast_
from src.mod.data.symbol_table import SymbolTable
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
        "enum",
        "ℤ",
        "ℕ",
        "ℕ₁",
    ]

    # limited form of the below type synthesizer (which should not need to encounter type annotations)
    # this should be accessible to populate_symbol_table though
    @classmethod
    def resolve_type_annotation(cls, ast: ast_.Type_ | ast_.ASTNode | ast_.None_, symbol_table: SymbolTable) -> BaseType:
        params = []
        if isinstance(ast, ast_.Type_):
            params = ast.generics
            ast = ast.type_

        match ast:
            # Built in types
            case ast_.Identifier("int"):
                return IntType()
            case ast_.Identifier("float"):
                return FloatType()
            case ast_.Identifier("string"):
                return StringType()
            case ast_.Identifier("bool"):
                return BoolType()
            case ast_.Identifier("ℤ"):
                return SetType(IntType())
            case ast_.Identifier("ℕ"):
                return SetType(IntType(trait_collection=TraitCollection(min_trait=MinTrait(ast_.Int("0")))))
            case ast_.Identifier("ℕ₁"):
                return SetType(IntType(trait_collection=TraitCollection(min_trait=MinTrait(ast_.Int("1")))))
            case ast_.Identifier("set"):
                if len(params) != 1:
                    raise SimileTypeError(f"Set type annotation must have exactly 1 parameter, got {len(params)}: {params}", ast)
                param_type = cls.resolve_type_annotation(params[0], symbol_table)
                return SetType(param_type)
            case ast_.Identifier("sequence"):
                if len(params) != 1:
                    raise SimileTypeError(f"Sequence type annotation must have exactly 1 parameter, got {len(params)}: {params}", ast)
                param_type = cls.resolve_type_annotation(params[0], symbol_table)
                return SequenceType(param_type)
            case ast_.Identifier("bag"):
                if len(params) != 1:
                    raise SimileTypeError(f"Bag type annotation must have exactly 1 parameter, got {len(params)}: {params}", ast)
                param_type = cls.resolve_type_annotation(params[0], symbol_table)
                return BagType(param_type)
            case ast_.Identifier("relation"):
                if len(params) != 2:
                    raise SimileTypeError(f"Relation type annotation must have exactly 2 parameters, got {len(params)}: {params}", ast)
                left_param_type = cls.resolve_type_annotation(params[0], symbol_table)
                right_param_type = cls.resolve_type_annotation(params[1], symbol_table)
                return RelationType(left_param_type, right_param_type)
            case ast_.Identifier("generic"):
                return GenericType()
            case ast_.Identifier("tuple"):
                params_as_types = list(map(lambda param: cls.resolve_type_annotation(param, symbol_table), params))
                return TupleType(tuple(params_as_types))
            # TODO what about enums?
            # case ast_.Identifier("enum"):

            # User-identified types (presumably)
            case ast_.Identifier(symbol_table_name):
                symbol_table_entry = symbol_table.lookup_identifier_in_current_scope(symbol_table_name)
                params_as_types = list(map(lambda param: cls.resolve_type_annotation(param, symbol_table), params))
                return DeferToSymbolTable(symbol_table_entry, params_as_types)
            # Special notation for relation types
            case ast_.RelationOp(left, right, op):
                if len(params) != 0:
                    raise SimileTypeError(f"Infix relation operator type annotation cannot have parameters, got {len(params)}: {params}", ast)
                left_type = cls.resolve_type_annotation(left, symbol_table)
                right_type = cls.resolve_type_annotation(right, symbol_table)
                rel_type = RelationType(left_type, right_type)
                rel_type.apply_traits_from_relation_operator(op)
                return rel_type

        raise SimileTypeError(f"Failed to resolve type annotation", ast)

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
