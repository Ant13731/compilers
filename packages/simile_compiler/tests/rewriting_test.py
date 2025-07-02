import pytest

from src.mod.ast_ import *
from src.mod.parser import parse
from src.mod.analysis import populate_ast_environments
from src.mod.optimizer import (
    collection_optimizer,
    SET_REWRITE_COLLECTION,
    SetComprehensionConstructionCollection,
    DisjunctiveNormalFormQuantifierPredicateCollection,
    GeneratorSelectionCollection,
    SetCodeGenerationCollection,
)

FULL_TEST = [
    ("card({s · s in {1, 2} | s})",),
    ("card({s · s in {1, 2} or s in {2, 3}| s})",),
    ("card({s · s in {1, 2} and s != 1 or s in {2, 3} and s = 3 | s})",),
    ("card({s · s in {1, 2} and (s != 1 or s != 0) or s in {2, 3} and s = 3 | s})",),
    ("card({s · s in { x + 1 | x in {1, 2} or x in {4}} or s in { x + 2 | x in {2, 3}} | s})",),
    ("card({s,t · s in {1, 2} and t == s + 1 or s in {2, 3} and t == s + 2 | t } and s = 3)",),
    ("card({s | s in {1, 2}})",),
    ("card({s | s in {1, 2} or s in {2, 3}})",),
    ("{s | s in {1, 2}}",),
    ("{s | s in {1, 2} or s in {2, 3}}",),
    (
        """
S = {1,2}
T = {1}
S ∪ T
"""
    ),
    (
        """
S = {1,2}
T = {1}
S ∩ T
"""
    ),
]
TEST_SET_COMPREHENSION = [
    (
        Union(
            SetEnumeration(
                [
                    Int("1"),
                    Int("1"),
                    Int("2"),
                ]
            ),
            SetEnumeration(
                [
                    Int("2"),
                    Int("3"),
                    Int("4"),
                ]
            ),
        ),
    ),
    (
        Union(
            SetEnumeration(
                [
                    Int("1"),
                    Int("1"),
                    Int("2"),
                ]
            ),
            SetEnumeration(
                [
                    Int("2"),
                ]
            ),
        ),
    ),
    (
        Union(
            SetEnumeration(
                [
                    Int("1"),
                    Int("1"),
                    Int("2"),
                ]
            ),
            SetComprehension(
                And(
                    [
                        In(
                            Identifier("x"),
                            SetEnumeration(
                                [
                                    Int("2"),
                                ]
                            ),
                        ),
                    ],
                ),
                Identifier("x"),
            ),
        ),
    ),
]

vars = """
a = True
b = False
c = b
d = a
"""
prepend = [
    Assignment(Identifier("a"), True_()),
    Assignment(Identifier("b"), True_()),
    Assignment(Identifier("c"), Identifier("a")),
    Assignment(Identifier("d"), Identifier("a")),
]
TEST_DNF = list(
    map(
        lambda i: (vars + i[0], Statements(prepend + [i[1]])),
        [
            ("a and b or c and d",),
            ("not a and b or c and (d or (a and (b or c)))",),
        ],
    )
)
TEST_GEN_SEL = [
    (
        SetComprehension(
            And(
                [
                    In(
                        Identifier("x"),
                        SetEnumeration(
                            [
                                Int("1"),
                                Int("2"),
                            ],
                        ),
                    )
                ]
            ),
            Identifier("x"),
        ),
    ),
    (
        SetComprehension(
            And(
                [
                    In(
                        Identifier("x"),
                        SetEnumeration(
                            [
                                Int("1"),
                                Int("2"),
                            ],
                        ),
                    ),
                    Equal(
                        Add(
                            Identifier("x"),
                            Int("1"),
                        ),
                        Identifier("y"),
                    ),
                ]
            ),
            Identifier("y"),
            7,
        ),
    ),
]
# TEST_CODE_GEN = [
#     (),
# ]


class TestRewritingSets:

    @pytest.mark.parametrize("input, expected", FULL_TEST)
    def test_full(self, input: str, expected: ASTNode):
        parsed_input = parse(input)
        assert not isinstance(parsed_input, list), f"Parser should not be throwing errors..., input {input} got {parsed_input}"
        analyzed_input = populate_ast_environments(parsed_input)
        actual = collection_optimizer(analyzed_input, SET_REWRITE_COLLECTION)
        assert actual == expected

    @pytest.mark.parametrize("input, expected", TEST_SET_COMPREHENSION)
    def test_set_comprehension_collection(self, input: ASTNode, expected: ASTNode):
        analyzed_input = populate_ast_environments(input)
        actual = collection_optimizer(analyzed_input, [SetComprehensionConstructionCollection])
        assert actual == expected

    @pytest.mark.parametrize("input, expected", TEST_DNF)
    def test_disjunctive_normal_form_quantifier_predicate_collection(self, input: ASTNode, expected: ASTNode):
        analyzed_input = populate_ast_environments(input)
        actual = collection_optimizer(analyzed_input, [DisjunctiveNormalFormQuantifierPredicateCollection])
        assert actual == expected

    @pytest.mark.parametrize("input, expected", TEST_GEN_SEL)
    def test_generator_selection_collection(self, input: ASTNode, expected: ASTNode):
        analyzed_input = populate_ast_environments(input)
        actual = collection_optimizer(analyzed_input, [GeneratorSelectionCollection])
        assert hasattr(actual, "_selected_generators")
        assert actual._selected_generators == expected

    # @pytest.mark.parametrize("input, expected", TEST_CODE_GEN)
    # def test_set_code_generation_collection(self, input: ASTNode, expected: ASTNode):
    #     analyzed_input = populate_ast_environments(input)
    #     actual = collection_optimizer(analyzed_input, [SetCodeGenerationCollection])
    #     assert actual == expected
