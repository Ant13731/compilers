from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.mod.data.ast_.base import ASTNode


class SimileTraitError(Exception):
    """Custom exception for Simile trait errors."""

    def __init__(self, message: str, node: ASTNode | None = None) -> None:
        message = f"SimileTraitError: {message}"
        if node is not None:
            message = f"Error {node.get_location()} (at node {node}): {message}"

        super().__init__(message)
        self.node = node
