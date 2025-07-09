from typing import Any

from src.mod.ast_.ast_node_base import ASTNode, Identifier
from src.mod.ast_.ast_nodes import (
    ListOp,
    Statements,
)


def structurally_equal(self: ASTNode, other: ASTNode) -> bool:
    """Check if two AST nodes are structurally equal (i.e., have the same AST structure aside from variable names)."""

    # there should be a one-to-one mapping of variable names from this node to the other node
    # FIXME USE ENVIRONMENT!!!!!!!!!!!!!Does this need scoping? like envs? probably
    variable_rename_table: dict[str, str] = {}

    def structurally_equal_aux(self_: ASTNode | Any, other_: ASTNode | Any, var_table: dict[str, str]) -> bool:
        if isinstance(self_, Identifier) and isinstance(other_, Identifier):
            # If both are identifiers, check if they are the same or if they can be renamed
            if self_.name in var_table:
                return var_table[self_.name] == other_.name

            # Other name is somewhere in the var table but doesn't correspond to this var...
            if other_.name in var_table.values():
                return False

            # Add to variable rename table
            var_table[self_.name] = other_.name
            return True

        # If not an ASTNode, nothing special to check, just compare values
        if not isinstance(self_, ASTNode):
            return self_ == other_

        # If self is an ast node but other is not, they are not structurally equal
        if not isinstance(other_, ASTNode):
            return False

        # Eliminate superfluous And, Or, wrapped statements, etc.

        if isinstance(self_, (ListOp, Statements)) and len(self_.items) == 1:
            self_ = self_.items[0]
        if isinstance(other_, (ListOp, Statements)) and len(other_.items) == 1:
            other_ = other_.items[0]

        # Call on all other fields
        for self_f, other_f in zip(self_.children(), other_.children()):
            if not structurally_equal_aux(self_f, other_f, var_table):
                return False
        return True

    return structurally_equal_aux(self, other, variable_rename_table)
