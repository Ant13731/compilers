import pytest
from dataclasses import dataclass
from copy import deepcopy

from src import ast_
from src import analysis


def mkenv(new_env: dict) -> analysis.Environment:
    return analysis.Environment(
        previous=analysis.STARTING_ENVIRONMENT,
        table=new_env,
    )


TEST_ASTS = [
    ast_.Statements(
        [
            ast_.Assignment(
                ast_.Identifier("x"),
                ast_.Int("5"),
            )
        ],
    ),
    ast_.Statements(
        [
            ast_.Assignment(
                ast_.Identifier("test_enum"),
                ast_.Enumeration(
                    [ast_.Identifier("a"), ast_.Identifier("b"), ast_.Identifier("c")],
                    op_type=ast_.CollectionOperator.SET,
                ),
            )
        ],
    ),
    ast_.Statements(
        [
            ast_.StructDef(
                ast_.Identifier("TestStruct"),
                [
                    ast_.TypedName(ast_.Identifier("a"), ast_.Type_(ast_.Identifier("int"))),
                    ast_.TypedName(ast_.Identifier("b"), ast_.Type_(ast_.Identifier("str"))),
                ],
            ),
            ast_.Assignment(
                ast_.Identifier("test_struct"),
                ast_.Call(
                    ast_.Identifier("TestStruct"),
                    [
                        ast_.Int("42"),
                        ast_.String("hello"),
                    ],
                ),
            ),
        ],
    ),
]

TEST_AST_TYPES = list(
    map(
        mkenv,
        [
            {"x": ast_.BaseSimileType.Int},
            {"test_enum": ast_.EnumTypeDef({"a", "b", "c"})},
            {
                "TestStruct": ast_.StructTypeDef({"a": ast_.BaseSimileType.Int, "b": ast_.BaseSimileType.String}),
                "test_struct": ast_.DeferToSymbolTable(lookup_type="TestStruct"),
            },
        ],
    )
)

TEST_ASTS_WITH_TYPES = []
for ast_node, ast_type in zip(TEST_ASTS, TEST_AST_TYPES):
    typed_ast_node = deepcopy(ast_node)
    typed_ast_node.env = ast_type
    TEST_ASTS_WITH_TYPES.append((ast_node, typed_ast_node))


class TestAnalysis:
    @pytest.mark.parametrize("ast_node, typed_ast_node", TEST_ASTS_WITH_TYPES)
    def test_type_analysis(self, ast_node: ast_.ASTNode, typed_ast_node: ast_.ASTNode):
        analyzed_ast = analysis.populate_ast_with_types(ast_node)
        assert analyzed_ast == typed_ast_node
