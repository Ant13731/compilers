import pytest
from dataclasses import dataclass

from simile_compiler import ast_


@dataclass
class TestDummyNode(ast_.ASTNode):
    pass


TEST_CONTAINS_CHILD = [
    (ast_.Identifier("a"), ast_.Identifier("a")),
    (
        ast_.BinaryOp(
            ast_.Identifier("x"),
            ast_.Identifier("y"),
            op_type=ast_.BinaryOpType.ADD,
        ),
        ast_.Identifier("x"),
    ),
    (
        ast_.IdentList([ast_.Identifier("x"), ast_.Identifier("y")]),
        ast_.Identifier("x"),
    ),
    (
        ast_.SetEnumeration(
            [
                ast_.BinaryOp(
                    ast_.Identifier("x"),
                    ast_.Identifier("y"),
                    op_type=ast_.BinaryOpType.ADD,
                ),
                ast_.BinaryOp(
                    ast_.Identifier("a"),
                    ast_.Identifier("b"),
                    op_type=ast_.BinaryOpType.MULTIPLY,
                ),
            ]
        ),
        ast_.Add(TestDummyNode(), TestDummyNode()),
    ),
    (
        ast_.SetEnumeration(
            [
                ast_.BinaryOp(
                    ast_.Identifier("x"),
                    ast_.Identifier("y"),
                    op_type=ast_.BinaryOpType.ADD,
                ),
                ast_.BinaryOp(
                    ast_.Identifier("a"),
                    ast_.Identifier("b"),
                    op_type=ast_.BinaryOpType.MULTIPLY,
                ),
            ]
        ),
        ast_.Multiply(TestDummyNode(), TestDummyNode()),
    ),
]
TEST_NOT_CONTAINS_CHILD = []
TEST_CONTAINS_IDENTIFIER = []
TEST_CHILDREN = []
TEST_LEAF_NODES = []
TEST_NOT_LEAF_NODES = []


class TestASTNode:
    @pytest.mark.parametrize("ast_node", map(lambda x: x[0], TEST_CONTAINS_CHILD))
    def test_contains_self(self, ast_node: ast_.ASTNode):
        # Test if the ASTNode contains itself
        args = [ast_node.__class__]
        if hasattr(ast_node, "op_type"):
            args.append(ast_node.op_type)
        assert ast_node.contains(*args)

    @pytest.mark.parametrize("ast_node, contains", TEST_CONTAINS_CHILD)
    def test_contains_child(self, ast_node: ast_.ASTNode, contains: ast_.ASTNode):
        # Test if the ASTNode contains a specific child node type
        assert ast_node.contains(contains.__class__, contains.op_type)

    # @pytest.mark.parametrize("ast_node, not_contains", TEST_NOT_CONTAINS_CHILD)
    # def test_contains_not_child(self, ast_node: ast_.ASTNode, not_contains: type[ast_.ASTNode]):
    #     # Test if the ASTNode does not contain a specific child node type
    #     assert not ast_node.contains(not_contains)

    # @pytest.mark.parametrize("ast_node, identifier", TEST_CONTAINS_IDENTIFIER)
    # def test_find_all_instances(self, ast_node: ast_.ASTNode, instance: type[ast_.ASTNode], found_instances: list[ast_.ASTNode]):
    #     # Test if the ASTNode finds all instances of a specific type
    #     assert ast_node.find_all_instances(instance) == found_instances

    # @pytest.mark.parametrize("ast_node, children", TEST_CHILDREN)
    # def test_list_children(self, ast_node: ast_.ASTNode, children: list[ast_.ASTNode]):
    #     # Test if the ASTNode lists all children
    #     assert ast_node.list_children() == children

    # @pytest.mark.parametrize("ast_node", TEST_LEAF_NODES)
    # def test_is_leaf(self, ast_node: ast_.ASTNode):
    #     # Test if the ASTNode is a leaf node
    #     assert ast_node.is_leaf()

    # @pytest.mark.parametrize("ast_node", TEST_NOT_LEAF_NODES)
    # def test_is_not_leaf(self, ast_node: ast_.ASTNode):
    #     # Test if the ASTNode is not a leaf node
    #     assert not ast_node.is_leaf()
