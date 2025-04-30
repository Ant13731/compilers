from ast_ import *
from rewrite_strategy_select_examples.test_items import *


def test_rewrite_strategy_1(ast: BaseAST) -> BaseAST:
    """First pass rewrite strategy.

    Args:
        ast: plain AST as parsed from Lark, no pre-optimization

    Returns:
        BaseAST: AST with all sets rewritten as generators. Sets may be nested
    """
    match ast:
        # Cases we care about
        case Set(elements):
            # Assuming set literal only contains integers for simplicity
            # Convert into set constructor notation
            return SetComprehension(
                elem=expr_preamble(Identifier("x")),
                generator=expr_preamble(
                    In(
                        Identifier("x"),
                        Set(elements),
                    )
                ),
            )
        case Identifier(name):
            if name in symbol_lookup_table:
                return test_rewrite_strategy_1(symbol_lookup_table[name])
            else:
                print(f"Identifier {name} not found in symbol lookup table.")
                return ast
        case SetComprehension(elem, generator):
            if generator.contains(Union) or generator.contains(Intersect) or generator.contains(Difference):
                return SetComprehension(
                    elem=elem,
                    generator=test_rewrite_strategy_1(generator),
                )
            return ast
        # Descend the tree otherwise
        case Start(body):
            new_body = test_rewrite_strategy_1(body)
            return Start(new_body)
        case Statements(statements):
            new_statements = [test_rewrite_strategy_1(statement) for statement in statements]
            return Statements(new_statements)
        case Union(left, right):
            left = test_rewrite_strategy_1(left)
            right = test_rewrite_strategy_1(right)
            return Union(left, right)
        case Intersect(left, right):
            left = test_rewrite_strategy_1(left)
            right = test_rewrite_strategy_1(right)
            return Intersect(left, right)
        case Difference(left, right):
            left = test_rewrite_strategy_1(left)
            right = test_rewrite_strategy_1(right)
            return Difference(left, right)
        case In(left, right):
            right = test_rewrite_strategy_1(right)
            return In(left, right)
        case Expr(expr):
            return Expr(test_rewrite_strategy_1(expr))
        case EquivalenceStmt(stmt):
            return EquivalenceStmt(test_rewrite_strategy_1(stmt))
        case _:
            return ast


def test_rewrite_strategy_1_denest(ast: BaseAST) -> BaseAST:
    """Cleans up the AST after the first rewrite strategy."""
    match ast:
        # Cases we care about
        case SetComprehension(
            Expr(EquivalenceStmt(Identifier(x))),
            Expr(EquivalenceStmt(In(Identifier(y), Expr(EquivalenceStmt(right))))),
        ):
            # For now, take the easy way out in deconstructing the generator
            # since we know all tests use Expr(EquivalenceStmt(In(...))))
            # When we add properties, this will need to be more intelligent
            if right.contains(SetComprehension) and x == y:
                return right

            return ast
        # Descend the tree otherwise
        case Start(body):
            new_body = test_rewrite_strategy_1_denest(body)
            return Start(new_body)
        case Statements(statements):
            new_statements = [test_rewrite_strategy_1_denest(statement) for statement in statements]
            return Statements(new_statements)
        case Union(left, right):
            left = test_rewrite_strategy_1_denest(left)
            right = test_rewrite_strategy_1_denest(right)
            return Union(left, right)
        case Intersect(left, right):
            left = test_rewrite_strategy_1_denest(left)
            right = test_rewrite_strategy_1_denest(right)
            return Intersect(left, right)
        case Difference(left, right):
            left = test_rewrite_strategy_1_denest(left)
            right = test_rewrite_strategy_1_denest(right)
            return Difference(left, right)
        case Expr(expr):
            return Expr(test_rewrite_strategy_1_denest(expr))
        case EquivalenceStmt(stmt):
            return EquivalenceStmt(test_rewrite_strategy_1_denest(stmt))
        case _:
            return ast


def substitute_identifier(ast: BaseAST, identifier: Identifier, replacement: Identifier) -> BaseAST:
    """Substitutes an identifier in the AST with a replacement identifier.

    Naive implementation that should ensure not to overwrite existing identifiers,
    but we need a symbol table for that."""
    match ast:
        case Identifier(name) if name == identifier.value:
            return replacement
        case SetComprehension(elem, generator):
            new_elem = substitute_identifier(elem, identifier, replacement)
            new_generator = substitute_identifier(generator, identifier, replacement)
            return SetComprehension(new_elem, new_generator)
        case Union(left, right):
            new_left = substitute_identifier(left, identifier, replacement)
            new_right = substitute_identifier(right, identifier, replacement)
            return Union(new_left, new_right)
        case Intersect(left, right):
            new_left = substitute_identifier(left, identifier, replacement)
            new_right = substitute_identifier(right, identifier, replacement)
            return Intersect(new_left, new_right)
        case Difference(left, right):
            new_left = substitute_identifier(left, identifier, replacement)
            new_right = substitute_identifier(right, identifier, replacement)
            return Difference(new_left, new_right)
        case And(elems):
            new_elems = [substitute_identifier(elem, identifier, replacement) for elem in elems]
            return And(new_elems)
        case Or(elems):
            new_elems = [substitute_identifier(elem, identifier, replacement) for elem in elems]
            return Or(new_elems)
        case Not(left):
            new_left = substitute_identifier(left, identifier, replacement)
            return Not(new_left)
        case In(left, right):
            new_left = substitute_identifier(left, identifier, replacement)
            new_right = substitute_identifier(right, identifier, replacement)
            return In(new_left, new_right)
        case Start(body):
            new_body = substitute_identifier(body, identifier, replacement)
            return Start(new_body)
        case Statements(statements):
            new_statements = [substitute_identifier(statement, identifier, replacement) for statement in statements]
            return Statements(new_statements)
        case Expr(expr):
            return Expr(substitute_identifier(expr, identifier, replacement))
        case EquivalenceStmt(stmt):
            return EquivalenceStmt(substitute_identifier(stmt, identifier, replacement))
        case _:
            return ast


def test_rewrite_strategy_2(ast: BaseAST) -> BaseAST:
    """Second pass of rewrites assuming `test_rewrite_strategy_1` has run."""
    match ast:
        case Union(
            SetComprehension(
                # For now we only support identifier dummy variables (not functions)
                Expr(EquivalenceStmt(Identifier(l_elem))),
                Expr(EquivalenceStmt(l_generator)),
            ),
            SetComprehension(
                Expr(EquivalenceStmt(Identifier(r_elem))),
                Expr(EquivalenceStmt(r_generator)),
            ),
        ):
            l_generator = substitute_identifier(l_generator, Identifier(l_elem), Identifier("x"))
            r_generator = substitute_identifier(r_generator, Identifier(r_elem), Identifier("x"))
            return SetComprehension(
                expr_preamble(Identifier("x")),
                expr_preamble(Or.flatten_and_join([l_generator, r_generator])),
            )
        case Intersect(
            SetComprehension(
                Expr(EquivalenceStmt(Identifier(l_elem))),
                Expr(EquivalenceStmt(l_generator)),
            ),
            SetComprehension(
                Expr(EquivalenceStmt(Identifier(r_elem))),
                Expr(EquivalenceStmt(r_generator)),
            ),
        ):
            l_generator = substitute_identifier(l_generator, Identifier(l_elem), Identifier("x"))
            r_generator = substitute_identifier(r_generator, Identifier(r_elem), Identifier("x"))
            return SetComprehension(
                expr_preamble(Identifier("x")),
                expr_preamble(And.flatten_and_join([l_generator, r_generator])),
            )
        case Difference(
            SetComprehension(
                Expr(EquivalenceStmt(Identifier(l_elem))),
                Expr(EquivalenceStmt(l_generator)),
            ),
            SetComprehension(
                Expr(EquivalenceStmt(Identifier(r_elem))),
                Expr(EquivalenceStmt(r_generator)),
            ),
        ):
            l_generator = substitute_identifier(l_generator, Identifier(l_elem), Identifier("x"))
            r_generator = substitute_identifier(r_generator, Identifier(r_elem), Identifier("x"))
            # TODO check Not vs NotIn, maybe we dont need the NotIn ASR construct after all
            # We can leave the key words and grammar, but the transformer could just produce Not(In(...)) instead of NotIn
            return SetComprehension(
                expr_preamble(Identifier("x")),
                expr_preamble(And.flatten_and_join([l_generator, Not(r_generator)])),
            )
        # Descend the tree otherwise
        case Union(left, right):
            left = test_rewrite_strategy_2(left)
            right = test_rewrite_strategy_2(right)
            # TODO this might be a hack, but it basically allows us to work from the bottom up
            # in case of nested set operators. Eventually, we want to eliminate all set operators with the first 3 rules
            return test_rewrite_strategy_2(Union(left, right))
        case Intersect(left, right):
            left = test_rewrite_strategy_2(left)
            right = test_rewrite_strategy_2(right)
            return test_rewrite_strategy_2(Intersect(left, right))
        case Difference(left, right):
            left = test_rewrite_strategy_2(left)
            right = test_rewrite_strategy_2(right)
            return test_rewrite_strategy_2(Difference(left, right))
        case Start(body):
            new_body = test_rewrite_strategy_2(body)
            return Start(new_body)
        case Statements(statements):
            new_statements = [test_rewrite_strategy_2(statement) for statement in statements]
            return Statements(new_statements)
        case Expr(expr):
            return Expr(test_rewrite_strategy_2(expr))
        case EquivalenceStmt(stmt):
            return EquivalenceStmt(test_rewrite_strategy_2(stmt))
        case _:
            return ast


def iterable_size(generator: BaseAST) -> int:
    """Returns the expected size of the iterable nested within a generator."""
    match generator:
        case Set(elems):
            return len(elems)
        case _:
            raise ValueError("Could not determine size of iterable (only set literals supported for now)")


def select_generator(generator: BaseAST, elem_identifier: Identifier, is_top_level: bool = True) -> list[tuple[BaseAST, And | Or | None_]]:
    match generator:
        case In(left, right) if left == elem_identifier:
            return [(right, None_())]
        case And(elems):
            # Find smallest set within the and clause
            # For now just assume
            generator_for_this_clause, conditions_for_this_clause = None, None
            for elem in elems:
                match elem:
                    case In(left, right) as candidate_generator if left == elem_identifier:
                        if generator_for_this_clause is not None and iterable_size(generator_for_this_clause) < iterable_size(right):
                            continue

                        generator_for_this_clause = right
                        conditions_for_this_clause_list = [elem for elem in elems if elem != candidate_generator]
                        if len(conditions_for_this_clause_list) == 1:
                            conditions_for_this_clause = conditions_for_this_clause_list[0]
                        else:
                            conditions_for_this_clause = And(conditions_for_this_clause_list)
                    case _:
                        continue
            return [(generator_for_this_clause, conditions_for_this_clause)]

        case Or(elems) if is_top_level:
            generator_list = []
            for elem in elems:
                generator_list += select_generator(elem, elem_identifier, False)
            return generator_list

        case _:
            raise ValueError("Generator must have at least one In statement with the selected element_identifier per top-level or-clause")


def test_rewrite_strategy_3(ast: BaseAST) -> BaseAST:
    """Third pass of rewrites - no set operations should be left at this point"""
    match ast:
        case SetComprehension(Expr(EquivalenceStmt(Identifier(elem) as elem_identifier)), Expr(EquivalenceStmt(generator))):
            generators = select_generator(generator, elem_identifier)
            return ConcreteSet(Identifier(elem), [ConcreteSetLoop(generator, conditions) for (generator, conditions) in generators])
        case Start(body):
            new_body = test_rewrite_strategy_3(body)
            return Start(new_body)
        case Statements(statements):
            new_statements = [test_rewrite_strategy_3(statement) for statement in statements]
            return Statements(new_statements)
        case Expr(expr):
            return Expr(test_rewrite_strategy_3(expr))
        case EquivalenceStmt(stmt):
            return EquivalenceStmt(test_rewrite_strategy_3(stmt))
        case _:
            return ast
