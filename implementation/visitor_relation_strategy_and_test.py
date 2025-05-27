from ast_ import *
from rewrite_strategy_select_examples.test_items import *
from parser import parse

visitor_relation = "card((location ** -1 circ attends ** -1)[{room}])"

visitor_relation_parsed_ast = parse(visitor_relation)


#### TEMP ASTS
@dataclass
class Sum(BaseAST):
    elem: BaseAST
    gen: BaseAST
    pred: BaseAST


@dataclass
class BSetComprehension(BaseAST):
    elem: BaseAST
    gen: BaseAST
    pred: BaseAST


#### TEMP CONCRETE ASTS
@dataclass
class ConcreteRelationLoop(BaseAST):
    generator: In
    nested_generators: list[In]
    conditions: In | And | Or | None_
    action: BaseAST


@dataclass
class ConcreteFold(BaseAST):
    starting_statement: BaseAST
    loop: BaseAST


symbol_lookup_table = {
    "card": lambda gen, pred: Sum(Int(1), gen, pred),
    "location": lambda: Dict(
        list(
            map(
                lambda tup: KeyPair(*tup),
                zip(
                    list(map(lambda x: expr_preamble(Int(x)), range(0, 5))),
                    list(map(lambda x: expr_preamble(Int(x)), range(7, 10))),
                ),
            ),
        ),
    ),
    "attends": lambda: Dict(
        list(
            map(
                lambda tup: KeyPair(*tup),
                zip(
                    list(map(lambda x: expr_preamble(Int(x)), range(7, 10))),
                    list(map(lambda x: expr_preamble(Int(x)), range(15, 17))),
                ),
            ),
        ),
    ),
}

expected_visitor_relation_ast = start_preamble(
    expr_preamble(
        Call(
            Identifier("card"),
            Arguments(
                [
                    expr_preamble(
                        Indexing(
                            expr_preamble(
                                RelationComposition(
                                    Power(Identifier("location"), Neg(Int(1))),
                                    Power(Identifier("attends"), Neg(Int(1))),
                                ),
                            ),
                            EquivalenceStmt(
                                Set(
                                    [
                                        expr_preamble(Identifier("room")),
                                    ]
                                )
                            ),
                        ),
                    ),
                ]
            ),
        ),
    ),
)

expected_set_construction_visitor_relation_ast = start_preamble(
    expr_preamble(
        Sum(
            Int(1),
            And(
                [
                    In(
                        KeyPair(Identifier("c"), Identifier("a")),
                        Identifier("location"),
                    ),
                    In(
                        KeyPair(Identifier("b"), Identifier("c'")),
                        Identifier("attends"),
                    ),
                ]
            ),
            And(
                [
                    expr_preamble(Eq(Identifier("a"), Identifier("room"))),
                    expr_preamble(Eq(Identifier("c"), Identifier("c'"))),
                ]
            ),
        )
    )
)
# expected_final_visitor_relation_ast

assert visitor_relation_parsed_ast == expected_visitor_relation_ast


def to_set_construction_notation(ast: BaseAST) -> BaseAST:
    match ast:
        case Call(Identifier("card"), Arguments([expr])):
            return Sum(Int(1), to_set_construction_notation(expr), None_())
        case Indexing(target, slice):  # Targeted at relational image, missing a lot of checks for that for now
            return BSetComprehension(
                Identifier("b"),
                to_set_construction_notation(
                    In(
                        KeyPair(Identifier("a"), Identifier("b")),
                        target,
                    )
                ),
                to_set_construction_notation(In(Identifier("a"), slice)),
            )
        case In(Identifier(id_), EquivalenceStmt(Set([Expr(EquivalenceStmt(elem))]))):
            return expr_preamble(Eq(Identifier(id_), elem))
        case In(
            KeyPair(Identifier(id1), Identifier(id2)),
            Expr(EquivalenceStmt(RelationComposition(left, right))),
        ):
            intermediate_id1 = Identifier("c")
            intermediate_id2 = Identifier("c'")
            return And(
                list(
                    map(
                        to_set_construction_notation,
                        [
                            In(KeyPair(Identifier(id1), intermediate_id1), left),
                            In(KeyPair(intermediate_id2, Identifier(id2)), right),
                            Expr(EquivalenceStmt(Eq(intermediate_id1, intermediate_id2))),
                        ],
                    )
                )
            )
        case In(
            KeyPair(Identifier(id1), Identifier(id2)),
            Power(rel, Neg(Int(1))),
        ):
            return In(
                KeyPair(Identifier(id2), Identifier(id1)),
                to_set_construction_notation(rel),
            )
        # Descend the tree otherwise
        case Start(body):
            new_body = to_set_construction_notation(body)
            return Start(new_body)
        case Statements(statements):
            new_statements = [to_set_construction_notation(statement) for statement in statements]
            return Statements(new_statements)
        case In(left, right):
            right = to_set_construction_notation(right)
            return In(left, right)
        case Expr(expr):
            return Expr(to_set_construction_notation(expr))
        case EquivalenceStmt(stmt):
            return EquivalenceStmt(to_set_construction_notation(stmt))
        case _:
            return ast


def hack_for_membership_collapse_with_card(ast: BaseAST) -> BaseAST:
    match ast:
        case Sum(Int(1), Expr(EquivalenceStmt(BSetComprehension(elem, gen, pred_))), pred):
            return Sum(Int(1), gen, And([pred_, pred]))  # We can just eliminate the BSet elem since we use a constant
        # Descend the tree otherwise
        case Start(body):
            new_body = hack_for_membership_collapse_with_card(body)
            return Start(new_body)
        case Statements(statements):
            new_statements = [hack_for_membership_collapse_with_card(statement) for statement in statements]
            return Statements(new_statements)
        case In(left, right):
            right = hack_for_membership_collapse_with_card(right)
            return In(left, right)
        case Expr(expr):
            return Expr(hack_for_membership_collapse_with_card(expr))
        case EquivalenceStmt(stmt):
            return EquivalenceStmt(hack_for_membership_collapse_with_card(stmt))
        case _:
            return ast


def bool_manipulation(ast: BaseAST) -> BaseAST:
    match ast:
        case Sum(elem, And(gens), And(preds)):  # do this for other quantifiers, but we just need sum for this example
            new_gens = []
            new_preds = preds
            for gen in gens:
                match gen:
                    case In(left, right):
                        new_gens.append(In(left, right))
                    case _:
                        new_preds.append(gen)

            return Sum(elem, And(new_gens), And(list(filter(lambda x: x != None_(), new_preds))))
        # Descend the tree otherwise
        case Start(body):
            new_body = bool_manipulation(body)
            return Start(new_body)
        case Statements(statements):
            new_statements = [bool_manipulation(statement) for statement in statements]
            return Statements(new_statements)
        case In(left, right):
            right = bool_manipulation(right)
            return In(left, right)
        case Expr(expr):
            return Expr(bool_manipulation(expr))
        case EquivalenceStmt(stmt):
            return EquivalenceStmt(bool_manipulation(stmt))
        case _:
            return ast


assert expected_set_construction_visitor_relation_ast == bool_manipulation(
    hack_for_membership_collapse_with_card(
        to_set_construction_notation(visitor_relation_parsed_ast),
    )
)


# TODO move this to a proper spot, add types, etc.
def flatten(xss):
    return [x for xs in xss for x in xs]


def generated_summation(ast: BaseAST) -> BaseAST:
    match ast:
        case Sum(elem, And(gens), pred):
            return ConcreteFold(
                starting_statement=Assignment(Identifier("c"), Int(0)),
                loop=generated_summation(
                    ConcreteRelationLoop(
                        generator=gens[0],  # Will eventually need to check for proper chains, but since we only have 2 nested loops for now, it works out
                        nested_generators=gens[1:],
                        conditions=pred,
                        action=Assignment(
                            Identifier("c"),
                            Add(Identifier("c"), elem),
                        ),
                    ),
                ),
            )
        case ConcreteRelationLoop(generator, nested_generators, And(conditions), action):
            if nested_generators == []:
                return ast
            # TODO make this a proper part of the AST infrastructure - easy traversal, etc.
            identifiers_bound_by_unlooped_relations = flatten([cond.find_all_instances(Identifier) for cond in nested_generators])
            parent_conditions = []
            nested_conditions = []
            for cond in conditions:
                identifiers = cond.find_all_instances(Identifier)
                if any(map(lambda id_: id_ in identifiers_bound_by_unlooped_relations, identifiers)):
                    nested_conditions.append(cond)
                else:
                    parent_conditions.append(cond)

            return ConcreteRelationLoop(
                generator=generator,
                nested_generators=[],
                conditions=And(parent_conditions),  # For now bring down all conditions
                action=ConcreteRelationLoop(
                    generator=nested_generators[0],
                    nested_generators=nested_generators[1:],
                    conditions=And(nested_conditions),
                    action=action,
                ),
            )
        # Descend the tree otherwise
        case Start(body):
            new_body = generated_summation(body)
            return Start(new_body)
        case Statements(statements):
            new_statements = [generated_summation(statement) for statement in statements]
            return Statements(new_statements)
        case In(left, right):
            right = generated_summation(right)
            return In(left, right)
        case Expr(expr):
            return Expr(generated_summation(expr))
        case EquivalenceStmt(stmt):
            return EquivalenceStmt(generated_summation(stmt))
        case _:
            return ast


print(
    generated_summation(
        bool_manipulation(
            hack_for_membership_collapse_with_card(
                to_set_construction_notation(visitor_relation_parsed_ast),
            )
        )
    ).pretty_print()
)
