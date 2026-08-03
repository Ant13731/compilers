from dataclasses import dataclass, fields
import pathlib
from typing_extensions import OrderedDict

from loguru import logger

from src.mod.data import ast_
from src.mod.data.symbol_table import (
    SymbolTable,
    IdentifierContext,
    ScopeContext,
    SymbolTableError,
    SymbolTableIdentifierEntry,
)
from src.mod.data.types import (
    BaseType,
    RecordType,
    ProcedureType,
    ModuleImports,
    ImportedSymbol,
    EnumType,
    SimileTypeError,
    SetType,
)
from src.mod.data.traits import TraitCollection, LiteralTrait
from src.mod.data.standard_library import STANDARD_LIBRARY_FOLDER

from src.mod.data.types.tuple_ import TupleType
from src.mod.pipeline.parser import parse, ParseError
from src.mod.pipeline.analysis.type_annotation_resolver import TypeAnnotationResolver


def populate_symbol_table(ast: ast_.ASTNode) -> SymbolTable:
    """Populates the symbol table with all identifiers in the AST.
    SIDE EFFECT: Transforms Identifiers within the ast into symbol-table assigned Symbols
    """

    if not isinstance(ast, ast_.Start):
        raise SymbolTableError("Cannot populate symbol table because AST is not a Start node")

    # # Add in the standard library
    # if isinstance(ast.body, ast_.Statements):
    #     file_path = (STANDARD_LIBRARY_FOLDER / "standard_library.sim").resolve()
    #     standard_library_import = ast_.Import(file_path, [], ast_.ImportOperator.ALL_NAMES)
    #     ast.body.items.insert(0, standard_library_import)

    symbol_table = SymbolTable()
    symbol_table.add_scope(ScopeContext.BASE)
    symbol_table_populator = PopulateSymbolTable(symbol_table)
    try:
        symbol_table_populator.populate(ast)
    finally:
        logger.debug(symbol_table.debug())
    return symbol_table


@dataclass
class PopulateSymbolTable:
    symbol_table: SymbolTable

    def populate(self, ast: ast_.ASTNode) -> ast_.ASTNode:
        assert isinstance(ast, ast_.ASTNode), f"Gotcha: {ast}"

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
                    if not isinstance(item, ast_.ASTNode):
                        break
                    new_list.append(self.populate(item))
                else:
                    setattr(ast, f.name, new_list)
            elif isinstance(field_value, ast_.ASTNode):
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

            case body if isinstance(body, ast_.QuantifierBody):
                _body = self._populate_loop_parameters_from_generators(body)
                return _body, False

            case ast_.Fold(accumulator_init, body):
                self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                _accumulator_init = self.populate(accumulator_init)
                _body = self.populate(body)
                self.symbol_table.pop_scope_level()

                assert isinstance(_accumulator_init, ast_.Assignment)
                assert isinstance(_body, ast_.QuantifierBody)
                return ast_.Fold(_accumulator_init, _body), False

            case ast_.IterBody(_, _) as iter_:
                return self._populate_iter(iter_), False

            case ast_.ProcedureDef(name, params, body, return_type):
                # NOTE: param_types is populated *after* the procedure type definition since symbols must be added within the scope of the procedure
                param_types: list[BaseType] = []
                return_type_ = TypeAnnotationResolver.resolve_type_annotation(return_type, self.symbol_table)
                assert isinstance(return_type_, BaseType), "Procedure return type must have a valid type annotation"
                self.symbol_table.add_symbol(
                    name.name,
                    IdentifierContext.PROCEDURE,
                    ProcedureType(TupleType(param_types), return_type_),
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
                    param_types.append(param_type)
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

            case ast_.TraitApplication(ast_.Assignment(ast_.TypedName(ast_.Identifier(name), ast_.Type_(ast_.Identifier("enum"), [])), value, is_choice), traits):
                if is_choice:
                    raise SimileTypeError(f"Choice assignment cannot be used with enum definitions", ast)
                if not isinstance(value, ast_.Enumeration):
                    raise SimileTypeError(f"Enum type annotation can only be applied to enumeration definitions, got {type(value)}", value)

                members: set[str] = set()
                for _item in value.items:
                    if not isinstance(_item, ast_.Identifier):
                        raise SimileTypeError(f"Invalid enum item name (must be an identifier): {_item}", _item)
                    if self.symbol_table.does_symbol_exist_in_current_scope(_item.name):
                        raise SimileTypeError(f"Enum item name {_item.name} already exists in current scope, cannot be used as enum item name", _item)
                    members.add(_item.name)

                trait_collection = TypeAnnotationResolver.resolve_trait_collection(traits, self.symbol_table)
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
                _name = self.populate(ast_.Identifier(name))
                _value = self.populate(value)
                return (
                    ast_.TraitApplication(
                        ast_.Assignment(
                            ast_.TypedName(
                                _name,
                                ast_.Type_(ast_.Identifier("enum"), []),
                            ),
                            _value,
                            is_choice,
                        ),
                        traits,
                    ),
                    False,
                )
            case ast_.TraitApplication(ast_.Assignment(ast_.TypedName(ast_.Identifier(name), ast_.Type_(ast_.Identifier("type"), [])), value, is_choice), traits):
                if is_choice:
                    raise SimileTypeError(f"Choice assignment cannot be used with type definitions", ast)
                trait_collection = TypeAnnotationResolver.resolve_trait_collection(traits, self.symbol_table)

                if not isinstance(value, ast_.Image):
                    raise SimileTypeError(f"Type type annotation can only be applied to image definitions (type[<your-type>]), got {type(value)}", value)
                value_as_type_ast = ast_.Type_(value.target, [value.index])

                type_value = TypeAnnotationResolver.resolve_type_annotation(value_as_type_ast, self.symbol_table)
                if type_value is None:
                    raise SimileTypeError(f"Type definitions must have a valid type annotation, got None", value)
                type_value.trait_collection = type_value.trait_collection.merge(trait_collection, True)
                self.symbol_table.add_symbol(
                    name,
                    IdentifierContext.TYPE_NAME,
                    type_value,
                )
                _name = self.populate(ast_.Identifier(name))
                _value = self.populate(value.index)
                _type_symbol = self._convert_identifier_to_symbol(ast_.Identifier("type"))
                return (
                    ast_.TraitApplication(
                        ast_.Assignment(
                            ast_.TypedName(_name, ast_.Type_(_type_symbol, [])),
                            ast_.Type_(_type_symbol, [_value]),
                            is_choice,
                        ),
                        traits,
                    ),
                    False,
                )
            case ast_.TraitApplication(ast_.Assignment(ast_.TypedName(ast_.Identifier(name), declared_type), value, is_choice), traits):
                trait_collection = TypeAnnotationResolver.resolve_trait_collection(traits, self.symbol_table)
                _declared_type = TypeAnnotationResolver.resolve_type_annotation(declared_type, self.symbol_table)
                if _declared_type is None:
                    raise SimileTypeError(f"Variable definitions must have a valid type annotation, got None", declared_type)
                _declared_type.trait_collection = _declared_type.trait_collection.merge(trait_collection, True)
                self.symbol_table.add_symbol(
                    name,
                    IdentifierContext.VARIABLE,
                    _declared_type,
                )
                _name = self.populate(ast_.Identifier(name))
                _value = self.populate(value)
                _declared_type_ast = self.populate(declared_type)
                assert isinstance(_declared_type_ast, ast_.Type_)
                return (
                    ast_.TraitApplication(
                        ast_.Assignment(
                            ast_.TypedName(_name, _declared_type_ast),
                            _value,
                            is_choice,
                        ),
                        traits,
                    ),
                    False,
                )

            case ast_.Assignment(ast_.TypedName(ast_.Identifier(name), ast_.Type_(ast_.Identifier("enum"), [])), value, is_choice_assignment):
                if is_choice_assignment:
                    raise SimileTypeError(f"Choice assignment cannot be used with enum definitions", ast)
                if not isinstance(value, ast_.Enumeration):
                    raise SimileTypeError(f"Enum type annotation can only be applied to enumeration definitions, got {type(value)}", value)

                members = set()
                for _item in value.items:
                    if not isinstance(_item, ast_.Identifier):
                        raise SimileTypeError(f"Invalid enum item name (must be an identifier): {_item}", _item)
                    if self.symbol_table.does_symbol_exist_in_current_scope(_item.name):
                        raise SimileTypeError(f"Enum item name {_item.name} already exists in current scope, cannot be used as enum item name", _item)
                    members.add(_item.name)

                self.symbol_table.add_symbol(
                    name,
                    IdentifierContext.ENUM,
                    EnumType(members=members),
                )
                for member in members:
                    symbol_item = self.symbol_table.add_symbol(
                        member,
                        IdentifierContext.ENUM_ITEM,
                    )
                    literal_trait_collection = TraitCollection(literal_trait=LiteralTrait(ast_.Symbol(symbol_item)))
                    symbol_item.declared_type = EnumType(members=members, trait_collection=literal_trait_collection)

            case ast_.Assignment(ast_.TypedName(ast_.Identifier(name), ast_.Type_(ast_.Identifier("type"), [])), value, is_choice_assignment):
                if not isinstance(value, ast_.Image):
                    raise SimileTypeError(f"Type type annotation can only be applied to image definitions (type[<your-type>]), got {type(value)}", value)
                value_as_type_ast = ast_.Type_(value.target, [value.index])

                type_value = TypeAnnotationResolver.resolve_type_annotation(value_as_type_ast, self.symbol_table)
                if type_value is None:
                    raise SimileTypeError(f"Type definitions must have a valid type annotation, got None", value)
                if is_choice_assignment:
                    raise SimileTypeError(f"Choice assignment cannot be used with type definitions", ast)
                self.symbol_table.add_symbol(
                    name,
                    IdentifierContext.TYPE_NAME,
                    type_value,
                )
                _name = self.populate(ast_.Identifier(name))
                _value = self.populate(value.index)
                _type_symbol = self._convert_identifier_to_symbol(ast_.Identifier("type"))
                return (
                    ast_.Assignment(
                        ast_.TypedName(_name, ast_.Type_(_type_symbol, [])),
                        ast_.Type_(_type_symbol, [_value]),
                        is_choice_assignment,
                    ),
                    False,
                )
            case ast_.Assignment(ast_.TypedName(ast_.Identifier(name), declared_type), value, is_choice_assignment):
                _declared_type = TypeAnnotationResolver.resolve_type_annotation(declared_type, self.symbol_table)
                if _declared_type is None:
                    raise SimileTypeError(f"Variable definitions must have a valid type annotation, got None", declared_type)

                self.symbol_table.add_symbol(
                    name,
                    IdentifierContext.VARIABLE,
                    _declared_type,
                )
            case ast_.Import(module_file_path, names_to_import, import_operator):
                # TODO fix this to allow for modules with different names (import ... as <name>)
                module_file_path = module_file_path.with_suffix(".sim")
                module_name, module_ast = _read_from_path_and_parse(module_file_path)
                module_scope = self.symbol_table.add_scope(ScopeContext.NAMESPACE)
                self.populate(module_ast)
                self.symbol_table.pop_scope_level()

                match import_operator:
                    case ast_.ImportOperator.SPECIFIC_NAMES:
                        for symbol_id in module_scope.declared_symbols:
                            symbol = self.symbol_table.lookup_symbol(symbol_id, module_scope.id_)
                            if symbol.name not in names_to_import:
                                continue
                            self.symbol_table.add_symbol(
                                symbol.name,
                                symbol.context,
                                ImportedSymbol(symbol),
                            )
                    case ast_.ImportOperator.ALL_NAMES:
                        for symbol_id in module_scope.declared_symbols:
                            symbol = self.symbol_table.lookup_symbol(symbol_id, module_scope.id_)
                            self.symbol_table.add_symbol(
                                symbol.name,
                                symbol.context,
                                ImportedSymbol(symbol),
                            )
                    case ast_.ImportOperator.MODULE_NAME:
                        self.symbol_table.add_symbol(
                            module_name,
                            IdentifierContext.MODULE_IMPORT,
                            ModuleImports(module_scope),
                        )
                return None, False

            # By this point, all identifiers should have been added to the symbol table
            # Replace them with symbol ids corresponding to the symbol table entry
            case ast_.Identifier(_) | ast_.MapletIdentifier(_) | ast_.TupleIdentifier(_):
                return self._convert_identifier_to_symbol(ast), False
        return None, True

    def _convert_identifier_to_symbol(self, ast: ast_.IdentifierListTypes) -> ast_.SymbolListTypes:
        match ast:
            case ast_.Identifier(name):
                if name in TypeAnnotationResolver.BUILT_IN_TYPES:
                    return ast_.Symbol(
                        SymbolTableIdentifierEntry(
                            -1,
                            -1,
                            name,
                            TypeAnnotationResolver.BUILT_IN_TYPES[name],
                            IdentifierContext.BUILTIN_TYPE,
                        )
                    )
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

    def _populate_loop_parameters_from_generators(self, quantifier: ast_.QuantifierBody) -> ast_.QuantifierBody:
        match quantifier:
            case ast_.QuantifierBody([], branches) if isinstance(branches, list):
                _branches = []
                for branch in branches:
                    self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                    _branches.append(self._populate_loop_parameters_from_generators(branch))
                    self.symbol_table.pop_scope_level()
                return ast_.QuantifierBody([], _branches)
            case ast_.QuantifierBody([], expr):
                raise SimileTypeError(f"Invalid quantification - expr case {expr} must be accompanied by a generator", quantifier)
            case ast_.QuantifierBody(generators, branches) if isinstance(branches, list):
                _generators = []
                for generator in generators:
                    _generators.append(self._populate_generator(generator))
                _branches = []
                for branch in branches:
                    self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                    _branches.append(self._populate_loop_parameters_from_generators(branch))
                    self.symbol_table.pop_scope_level()
                # Need to pop scope levels added from populate generators
                for _ in _generators:
                    self.symbol_table.pop_scope_level()
                return ast_.QuantifierBody(_generators, _branches)
            case ast_.QuantifierBody(generators, expr):
                assert isinstance(expr, ast_.ASTNode), "Other case caught by earlier match case"
                _generators = []
                for generator in generators:
                    _generators.append(self._populate_generator(generator))
                _expr = self.populate(expr)
                # Need to pop scope levels added from populate generators
                for _ in _generators:
                    self.symbol_table.pop_scope_level()
                return ast_.QuantifierBody(_generators, _expr)
            case _:
                raise SimileTypeError(f"Invalid quantification - must be a list of branches or a single expression", quantifier)

    def _populate_iter(self, iter_: ast_.IterBody) -> ast_.IterBody:
        match iter_:
            case ast_.IterBody([], branches) if isinstance(branches, list):
                _branches = []
                for branch in branches:
                    self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                    _branches.append(self._populate_iter(branch))
                    self.symbol_table.pop_scope_level()
                return ast_.IterBody([], _branches)
            case ast_.IterBody([], iter_body):
                raise SimileTypeError(f"Invalid iter - iter_body case {iter_body} must be accompanied by a generator", iter_)
            case ast_.IterBody(generators, branches) if isinstance(branches, list):
                _generators = []
                for generator in generators:
                    _generators.append(self._populate_iter_generator(generator))
                _branches = []
                for branch in branches:
                    self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
                    _branches.append(self._populate_iter(branch))
                    self.symbol_table.pop_scope_level()
                # Need to pop scope levels added from populate generators
                for _ in _generators:
                    self.symbol_table.pop_scope_level()
                return ast_.IterBody(_generators, _branches)
            case ast_.IterBody(generators, iter_body):
                assert isinstance(iter_body, ast_.ASTNode), "Other case caught by earlier match case"
                _generators = []
                for generator in generators:
                    _generators.append(self._populate_iter_generator(generator))
                _iter_body_return_body = self.populate(iter_body.body)
                _iter_body_return_value = self.populate(iter_body.return_value)
                # Need to pop scope levels added from populate generators
                for _ in _generators:
                    self.symbol_table.pop_scope_level()
                assert isinstance(_iter_body_return_value, ast_.SymbolListTypes)
                return ast_.IterBody(_generators, ast_.IterBodyEnd(_iter_body_return_body, _iter_body_return_value))
            case _:
                raise SimileTypeError(f"Invalid iter - must be a list of branches or a single expression", iter_)

    def _populate_generator(self, generator: ast_.Generator) -> ast_.Generator:
        """Populates a generator with the appropriate symbol table entries.
        ONLY ADDS SCOPE, SCOPE SHOULD BE POPPED AFTER THIS TO ACCOUNT FOR BRANCHING
        """
        self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
        assert isinstance(generator.identifiers, ast_.IdentifierListTypes)
        self._populate_loop_parameters(generator.identifiers)
        _identifier_symbols = self._convert_identifier_to_symbol(generator.identifiers)
        _iterable = self.populate(generator.set_)
        _predicate = self.populate(generator.predicate)
        return ast_.Generator(_identifier_symbols, _iterable, _predicate)

    def _populate_iter_generator(self, iter_generator: ast_.IterGenerator) -> ast_.IterGenerator:
        """Populates a generator with the appropriate symbol table entries.
        ONLY ADDS SCOPE, SCOPE SHOULD BE POPPED AFTER THIS TO ACCOUNT FOR BRANCHING
        """
        self.symbol_table.add_scope(ScopeContext.QUANTIFICATION)
        assert isinstance(iter_generator.generator.identifiers, ast_.IdentifierListTypes)
        self._populate_loop_parameters(iter_generator.generator.identifiers)
        _identifier_symbols = self._convert_identifier_to_symbol(iter_generator.generator.identifiers)
        _iterable = self.populate(iter_generator.generator.set_)
        _assignments: list[ast_.Assignment] = []
        for assignment in iter_generator.assignments:
            _assignment = self.populate(assignment)
            assert isinstance(_assignment, ast_.Assignment)
            _assignments.append(_assignment)
        _predicate = self.populate(iter_generator.generator.predicate)
        return ast_.IterGenerator(ast_.Generator(_identifier_symbols, _iterable, _predicate), _assignments)


def _read_from_path_and_parse(module_file_path: pathlib.Path) -> tuple[str, ast_.Start]:
    # Read in imported file
    try:
        full_module_path = pathlib.Path(module_file_path).resolve(strict=True)
    except Exception as e:
        raise SymbolTableError(f"Failed to parse module for symbol table importing: module {module_file_path} does not exist or is not a file", e) from e

    if not full_module_path.is_file():
        raise SymbolTableError(f"Failed to parse module for symbol table importing: module {module_file_path} is not a file")

    with open(full_module_path, "r") as f:
        module_content = f.read()

    # Parse module content
    try:
        module_ast: ast_.Start = parse(module_content, full_module_path)
    except ParseError as e:
        raise SymbolTableError(
            f"Failed to parse module for symbol table importing: module {module_file_path} does not contain a valid Simile module. Expected a single Start node at the top level."
        ) from e

    return full_module_path.stem, module_ast
