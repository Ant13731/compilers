from dataclasses import dataclass, fields
import pathlib
from typing_extensions import OrderedDict
from copy import deepcopy


from src.mod.data import ast_
from src.mod.data.ast_.operators import BinaryOperator
from src.mod.data.symbol_table import SymbolTableError
from src.mod.data.symbol_table import SymbolTable, IdentifierContext, ScopeContext
from src.mod.data.symbol_table.entry import SymbolTableIdentifierEntry
from src.mod.data.types import (
    BaseType,
    BoolType,
    RecordType,
    ProcedureType,
    AnyType_,
    GenericType,
    DeferToSymbolTable,
    ModuleImports,
    NoneType_,
    StringType,
    IntType,
    FloatType,
    SetType,
    EnumType,
    BagType,
    RelationType,
    SequenceType,
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
    TupleType,
    PairType,
    SimileTypeError,
)

from src.mod.pipeline.parser import parse, ParseError


class ParseImportError(Exception):
    pass


def populate_symbol_table(ast: ast_.ASTNode) -> SymbolTable:
    """Populates the symbol table with all identifiers in the AST.
    SIDE EFFECT: Transforms Identifiers within the ast into symbol-table assigned Symbols
    """

    symbol_table = SymbolTable()
    symbol_table.add_scope(ScopeContext.BASE)
    symbol_table_populator = PopulateSymbolTable(symbol_table)
    symbol_table_populator.populate(ast)
    return symbol_table


class TypeAnnotationResolver:
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
            # TODO Built in functions?
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


@dataclass
class PopulateSymbolTable:
    symbol_table: SymbolTable

    def populate(self, ast: ast_.ASTNode) -> ast_.ASTNode:
        new_ast_node, continue_populating = self._populate_aux(ast)
        if new_ast_node is not None:
            ast = new_ast_node

        if not continue_populating:
            return ast

        for f in fields(ast):
            field_value = getattr(ast, f.name)
            if isinstance(field_value, list):
                new_list = []
                for item in field_value:
                    new_list.append(self.populate(item))
                setattr(ast, f.name, new_list)
            else:
                setattr(ast, f.name, self.populate(field_value))
        return ast

    def _populate_aux(self, ast: ast_.ASTNode) -> tuple[ast_.ASTNode | None, bool]:
        """Returns: (Node to replace if not None, whether to continue populating children)"""
        match ast:
            # Scopes
            case ast_.If(condition, body, else_body):
                self.symbol_table.add_scope(ScopeContext.CONDITIONAL)
                _condition = self.populate(condition)
                _body = self.populate(body)
                _else_body = self.populate(else_body)
                self.symbol_table.pop_scope_level()

                assert isinstance(_else_body, ast_.Else | ast_.ElseIf | ast_.None_)
                return ast_.If(_condition, _body, _else_body), False

            case ast_.ElseIf(condition, body, else_body):
                self.symbol_table.add_scope(ScopeContext.CONDITIONAL)
                _condition = self.populate(condition)
                _body = self.populate(body)
                _else_body = self.populate(else_body)
                self.symbol_table.pop_scope_level()

                assert isinstance(_else_body, ast_.Else | ast_.ElseIf | ast_.None_)
                return ast_.ElseIf(_condition, _body, _else_body), False

            case ast_.Else(body):
                self.symbol_table.add_scope(ScopeContext.CONDITIONAL)
                _body = self.populate(body)
                self.symbol_table.pop_scope_level()

                return ast_.Else(_body), False

            case ast_.While(condition, body):
                self.symbol_table.add_scope(ScopeContext.LOOP)
                _condition = self.populate(condition)
                _body = self.populate(body)
                self.symbol_table.pop_scope_level()

                return ast_.While(_condition, _body), False

            case ast_.For(iterable_names, iterable, body):
                assert isinstance(iterable_names, ast_.IdentifierListTypes)

                self.symbol_table.add_scope(ScopeContext.LOOP)
                self._populate_loop_parameters(iterable_names)
                _iterable_symbols = self._convert_identifier_to_symbol(iterable_names)
                _iterable = self.populate(iterable)
                _body = self.populate(body)
                self.symbol_table.pop_scope_level()

                assert isinstance(_iterable_symbols, ast_.TupleSymbol)
                return ast_.For(_iterable_symbols, _iterable, _body), False

            case ast_.QualifiedQuantifier(bound_identifiers, predicate, expression, op_type):
                assert isinstance(bound_identifiers, ast_.IdentifierListTypes)

                self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                self._populate_loop_parameters(bound_identifiers)
                _iterable_symbols = self._convert_identifier_to_symbol(bound_identifiers)
                _predicate = self.populate(predicate)
                _expression = self.populate(expression)
                self.symbol_table.pop_scope_level()

                assert isinstance(_predicate, ast_.ListOp)
                assert isinstance(_iterable_symbols, ast_.TupleSymbol)
                return ast_.QualifiedQuantifier(_iterable_symbols, _predicate, _expression, op_type), False

            case ast_.Quantifier(predicate, expression, op_type):
                self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                unbound_identifiers = self._find_unbound_identifiers(ast)
                self._populate_loop_parameters(unbound_identifiers)
                _iterable_symbols = self._convert_identifier_to_symbol(unbound_identifiers)
                _predicate = self.populate(predicate)
                _expression = self.populate(expression)
                self.symbol_table.pop_scope_level()

                assert isinstance(_predicate, ast_.ListOp)
                assert isinstance(_iterable_symbols, ast_.TupleSymbol)
                return ast_.QualifiedQuantifier(_iterable_symbols, _predicate, _expression, op_type), False

            case ast_.Quantifier3(body, op_type):
                self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                self.symbol_table.pop_scope_level()


            case ast_.Fold(generators, accumulator_init, accumulate_expr):
            case ast_.Iter(generators, accumulator_init, return_identifiers, body):

            case ast_.ProcedureDef(name, params, body, return_type):
                # NOTE: params_dict is populated *after* the procedure type definition since symbols must be added within the scope of the procedure
                params_dict: dict[SymbolTableIdentifierEntry, BaseType] = OrderedDict()
                return_type_ = TypeAnnotationResolver.resolve_type_annotation(return_type, self.symbol_table)
                assert isinstance(return_type_, BaseType), "Procedure return type must have a valid type annotation"
                self.symbol_table.add_symbol(
                    name.name,
                    IdentifierContext.PROCEDURE,
                    ProcedureType(params_dict, return_type_),
                )
                _name_symbol = self._convert_identifier_to_symbol(name)

                self.symbol_table.add_scope(ScopeContext.PROCEDURE)

                _params_symbols: list[ast_.Symbol] = []
                for param in params:
                    if not isinstance(param.name, ast_.Identifier):
                        raise SimileTypeError(f"Invalid procedure parameter name (must be an identifier): {param.name}", param)

                    param_type = TypeAnnotationResolver.resolve_type_annotation(param.type_, self.symbol_table)
                    if not isinstance(param_type, BaseType):
                        raise SimileTypeError(f"Invalid procedure parameter type (must be a valid type): {param.type_}", param)

                    self.symbol_table.add_symbol(
                        param.name.name,
                        IdentifierContext.PROCEDURE_PARAMETER,
                        param_type,
                    )
                    param_symbol = self._convert_identifier_to_symbol(param.name)
                    assert isinstance(param_symbol, ast_.Symbol)
                    params_dict[param_symbol.symbol_table_entry] = param_type
                    _params_symbols.append(param_symbol)

                _body = self.populate(body)

                self.symbol_table.pop_scope_level()
                assert isinstance(_name_symbol, ast_.Symbol)
                return ast_.ProcedureDefSymbol(_name_symbol, _params_symbols, _body), False
            case ast_.RecordDef(ast_.Identifier(name), items):
                fields: dict[str, BaseType] = OrderedDict()
                for item in items:
                    if not isinstance(item.name, ast_.Identifier):
                        raise SimileTypeError(f"Invalid struct field name (must be an identifier): {item.name}", item)
                    field_type = TypeAnnotationResolver.resolve_type_annotation(item.type_, self.symbol_table)
                    assert isinstance(field_type, BaseType)
                    fields[item.name.name] = field_type

                self.symbol_table.add_symbol(
                    name,
                    IdentifierContext.RECORD,
                    RecordType(fields=fields),
                )

                # record_scope = self.symbol_table.add_scope(ScopeContext.RECORD)

                # field_symbols: list[ast_.Symbol] = []
                # for field_name, field_type in fields.items():
                #     self.symbol_table.add_symbol(
                #         field_name,
                #         IdentifierContext.RECORD_FIELD,
                #         field_type,
                #     )

                #     field_symbol = self._convert_identifier_to_symbol(ast_.Identifier(field_name))
                #     assert isinstance(field_symbol, ast_.Symbol)
                #     field_symbols.append(field_symbol)

                # self.symbol_table.pop_scope_level()

                _symbol = self._convert_identifier_to_symbol(ast_.Identifier(name))
                assert isinstance(_symbol, ast_.Symbol)
                return ast_.RecordDefSymbol(_symbol, fields), False
            case ast_.LambdaDef(params, predicate, expression):
                assert isinstance(params, ast_.IdentifierListTypes)

                self.symbol_table.add_scope(ScopeContext.LAMBDA)
                _predicate = self.populate(predicate)
                _expression = self.populate(expression)

                self._populate_loop_parameters(params)
                _param_symbols = self._convert_identifier_to_symbol(params)

                self.symbol_table.pop_scope_level()

                assert isinstance(_param_symbols, ast_.TupleSymbol)
                return ast_.LambdaDef(_param_symbols, _predicate, _expression), False

            # Symbols

            # Dont allow regular assignments to define types
            # case ast_.Assignment(ast_.Identifier(name), value, with_clauses, _):
            #     # TODO if variable is already defined, this could just be a reassignment?
            #     #  Then it would be the responsibility of the type analysis pass to check that the reassignment is valid
            #     # But normal assignment without a type annotation could produce a symbol (ex. typedef)
            #     if not self.symbol_table.does_symbol_exist_in_current_scope(name):
            #         self.symbol_table.add_symbol(
            #             name,
            #             IdentifierContext.VARIABLE,
            #         )
            case ast_.Assignment(ast_.TypedName(ast_.Identifier(name), ast_.Type_(ast_.Identifier("enum"), [])), value, with_clauses, _):
                if not isinstance(value, ast_.Enumeration):
                    raise SimileTypeError(f"Enum type annotation can only be applied to enumeration definitions, got {type(value)}", value)

                members: set[str] = set()
                for _item in value.items:
                    if not isinstance(_item, ast_.Identifier):
                        raise SimileTypeError(f"Invalid enum item name (must be an identifier): {_item}", _item)
                    if self.symbol_table.does_symbol_exist_in_current_scope(_item.name):
                        raise SimileTypeError(f"Enum item name {_item.name} already exists in current scope, cannot be used as enum item name", _item)
                    members.add(_item.name)

                trait_collection = TypeAnnotationResolver.resolve_trait_collection(with_clauses, self.symbol_table)
                self.symbol_table.add_symbol(
                    name,
                    IdentifierContext.ENUM,
                    EnumType(members=members, trait_collection=trait_collection),
                )
                for member in members:
                    symbol_item = self.symbol_table.add_symbol(
                        member,
                        IdentifierContext.ENUM_ITEM,
                    )
                    literal_trait_collection = TraitCollection(literal_trait=LiteralTrait(ast_.Symbol(symbol_item)))
                    symbol_item.declared_type = EnumType(
                        members=members,
                        trait_collection=literal_trait_collection.merge(trait_collection, True),
                    )

            case ast_.Assignment(ast_.TypedName(ast_.Identifier(name), ast_.Type_(ast_.Identifier("type"), [])), value, with_clauses, _):
                trait_collection = TypeAnnotationResolver.resolve_trait_collection(with_clauses, self.symbol_table)
                type_value = TypeAnnotationResolver.resolve_type_annotation(value, self.symbol_table)
                if type_value is None:
                    raise SimileTypeError(f"Type definitions must have a valid type annotation, got None", value)
                type_value.trait_collection = type_value.trait_collection.merge(trait_collection, True)
                self.symbol_table.add_symbol(
                    name,
                    IdentifierContext.TYPE_NAME,
                    type_value,
                )
            case ast_.Assignment(ast_.TypedName(ast_.Identifier(name), declared_type), value, with_clauses, _):
                trait_collection = TypeAnnotationResolver.resolve_trait_collection(with_clauses, self.symbol_table)
                _declared_type = TypeAnnotationResolver.resolve_type_annotation(declared_type, self.symbol_table)
                if _declared_type is None:
                    raise SimileTypeError(f"Variable definitions must have a valid type annotation, got None", declared_type)
                _declared_type.trait_collection = _declared_type.trait_collection.merge(trait_collection, True)
                self.symbol_table.add_symbol(
                    name,
                    IdentifierContext.VARIABLE,
                    _declared_type,
                )
            case ast_.Import(module_file_path, import_objects):
                raise NotImplementedError("Import statements are not yet supported in the symbol table population pass")
                _populate_from_import(self.symbol_table, import_objects, module_file_path)

            # By this point, all identifiers should have been added to the symbol table
            # Replace them with symbol ids corresponding to the symbol table entry
            case ast_.Identifier(_) | ast_.MapletIdentifier(_) | ast_.TupleIdentifier(_):
                return self._convert_identifier_to_symbol(ast), False
        return None, True

    def _convert_identifier_to_symbol(self, ast: ast_.IdentifierListTypes) -> ast_.SymbolListTypes:
        match ast:
            case ast_.Identifier(name):
                symbol_table_entry = self.symbol_table.lookup_identifier_in_current_scope(name)
                return ast_.Symbol(symbol_table_entry)
            case ast_.MapletIdentifier((left, right)):
                _left = self._convert_identifier_to_symbol(left)
                _right = self._convert_identifier_to_symbol(right)
                return ast_.MapletSymbol(_left, _right)
            case ast_.TupleIdentifier(identifiers):
                _identifiers = [self._convert_identifier_to_symbol(ident) for ident in identifiers]
                return ast_.TupleSymbol(tuple(_identifiers))
        raise ValueError(f"Unsupported identifier type: {type(ast)}. This should not happen")

    def _populate_loop_parameters(self, iterable_names: ast_.IdentifierListTypes) -> None:
        if isinstance(iterable_names, ast_.Identifier):
            self.symbol_table.add_symbol(
                iterable_names.name,
                IdentifierContext.LOOP_VARIABLE,
            )
        elif isinstance(iterable_names, ast_.TupleIdentifier):
            for ident in iterable_names.flatten():
                self._populate_loop_parameters(ident)
        else:
            raise SimileTypeError(f"Invalid for loop variable name (must be an identifier, maplet identifier, or tuple identifier): {iterable_names}", iterable_names)

    def _add_generators_to_scope(self, generators: ast_.ListOp | ast_.Generator | ast_.ASTNode, body_or_expr: ast_.ASTNode) -> tuple[ast_.ListOp | ast_.Generator, ast_.ASTNode]:
        # populate predicates per generator
        # nested generators create a new scope, or-separated generators reuse the same parent scope but should be individual scopes
        # (should check for or generators first). Call self.populate on the body as needed
        # (ex. if we have two or-separated generators, their bodies need to be populated in both scopes?)
        match generators:
            case ast_.ListOp(items, ast_.ListOperator.OR):
                for item in items:
                    self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                    populated_generator, new_body_or_expr = self._add_generators_to_scope(item, deepcopy(body_or_expr))
                    self.symbol_table.pop_scope_level()

                return ast_.ListOp([self._add_generators_to_scope(item, body_or_expr) for item in items], ast_.ListOperator.OR), body_or_expr
            case ast_.ListOp(items, ast_.ListOperator.AND):
                deepest_body_or_expr = body_or_expr
                populated_generators = []

                scopes_to_pop = 0
                for item in items:
                    self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                    populated_generator, deepest_body_or_expr = self._add_generators_to_scope(item, deepest_body_or_expr)
                    populated_generators.append(populated_generator)
                    scopes_to_pop += 1
                for _ in range(scopes_to_pop):
                    self.symbol_table.pop_scope_level()

                return ast_.ListOp(populated_generators, ast_.ListOperator.AND), deepest_body_or_expr
            case ast_.Generator(iterable_names, iterable, predicate):
                assert isinstance(iterable_names, ast_.IdentifierListTypes)

                self._populate_loop_parameters(iterable_names)
                _iterable_symbols = self._convert_identifier_to_symbol(iterable_names)
                _iterable = self.populate(iterable)
                _predicate = self.populate(predicate)
                _body_or_expr = self.populate(body_or_expr)
                self.symbol_table.pop_scope_level()
                return ast_.Generator(_iterable_symbols, _iterable, _predicate), _body_or_expr
            case _:
                raise SimileTypeError(f"Invalid generator type (must be a list of generators or a single generator): {generators}", generators)

    def _extract_identifiers_from_generators(self, generators: ast_.ListOp | ast_.Generator) -> list[ast_.]:
        match generators:
            case ast_.ListOp(items, _):
        if isinstance(generators, ast_.ListOp):
            identifiers: list[ast_.IdentifierListTypes] = []
            for generator in generators.items:
                if not isinstance(generator, ast_.Generator):
                    raise SimileTypeError(f"Invalid generator in quantifier (must be a generator): {generator}", generator)
                identifiers.append(self._extract_identifiers_from_generators(generator))
            return ast_.TupleIdentifier(tuple(identifiers))

        if isinstance(generators, ast_.Generator):
            if not isinstance(generators.iterable_names, ast_.IdentifierListTypes):
                raise SimileTypeError(f"Invalid generator iterable names (must be an identifier, maplet identifier, or tuple identifier): {generators.iterable_names}", generators.iterable_names)
            return generators.iterable_names

        raise SimileTypeError(f"Invalid generator type (must be a list of generators or a single generator): {generators}", generators)

    def _find_unbound_identifiers(self, ast: ast_.Quantifier) -> ast_.TupleIdentifier:
        """Finds unbound identifiers in an unqualified quantifier."""
        possible_generators = list(filter(lambda x: x.op_type == ast_.BinaryOperator.IN, ast.predicate.find_all_instances(ast_.BinaryOp)))
        possible_bound_identifiers: list[ast_.IdentifierListTypes] = []
        possible_bound_identifier_names: set[ast_.Identifier] = set()
        for possible_generator in possible_generators:
            if isinstance(possible_generator.left, ast_.IdentifierListTypes):
                possible_bound_identifiers.append(possible_generator.left)
                possible_bound_identifier_names.update(possible_generator.left.flatten())

            if isinstance(possible_generator.left, ast_.BinaryOp):
                left = possible_generator.left.try_cast_maplet_to_maplet_identifier()
                if left is None:
                    continue

                possible_bound_identifiers.append(left)
                possible_bound_identifier_names.update(left.flatten())

        for possible_bound_identifier in possible_bound_identifier_names:
            if not self.symbol_table.does_symbol_exist_in_current_scope(possible_bound_identifier.name):
                possible_bound_identifiers = list(filter(lambda x: not x.contains(possible_bound_identifier), possible_bound_identifiers))

        if not possible_bound_identifiers:
            raise SimileTypeError(
                f"Failed to infer bound variables for quantifier {ast_.ast_to_source(ast)}. "
                "Either the expression is ambiguously overwriting a predefined variable in scope, "
                "or no valid generators are present in the quantification expression. Please explicitly state bound variables",
                ast,
            )

        return ast_.TupleIdentifier(tuple(possible_bound_identifiers))


def _populate_from_import(
    symbol_table: SymbolTable,
    import_objects: ast_.TupleIdentifier | ast_.None_ | ast_.ImportAll,
    module_file_path: str,
) -> None:
    # Read in imported file
    full_module_path = pathlib.Path(module_file_path).resolve(strict=True)
    with open(full_module_path, "r") as f:
        module_content = f.read()

    # Parse module content
    try:
        module_ast: ast_.Start = parse(module_content)
    except ParseError as e:
        raise ParseImportError(f"Module {module_file_path} does not contain a valid Simile module. Expected a single Start node at the top level.") from e

    if isinstance(module_ast.body, ast_.None_):
        return

    # Populate the module AST with types
    module_symbol_table = populate_symbol_table(module_ast)

    # Add module symbols to namespace
    match import_objects:
        case ast_.ImportAll():
            for symbol_table_entry in module_symbol_table.get_top_level_symbols():
                symbol_table.add_symbol(
                    symbol_table_entry.name,
                    symbol_table_entry.context,
                    symbol_table_entry.declared_type,
                )
        case ast_.None_():
            symbol_table.add_symbol(
                full_module_path.stem,
                IdentifierContext.MODULE_IMPORT,
                ModuleImports(module_symbol_table.get_top_level_symbols()),
            )
        case ast_.TupleIdentifier(identifiers):
            identifier_names = []
            for identifier in identifiers:
                if not isinstance(identifier, ast_.Identifier):
                    raise SimileTypeError(f"Invalid import type (must be an identifier): {identifier}", identifier)
                identifier_names.append(identifier.name)

            for symbol_table_entry in module_symbol_table.get_top_level_symbols():
                if symbol_table_entry.name in identifier_names:
                    symbol_table.add_symbol(
                        symbol_table_entry.name,
                        symbol_table_entry.context,
                        symbol_table_entry.declared_type,
                    )
