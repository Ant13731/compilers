from functools import singledispatchmethod
from dataclasses import dataclass, field
from typing import ClassVar, Any

from src.mod import ast_


class CodeGeneratorError(Exception):
    """Custom exception for code generation errors."""

    pass


@dataclass
class CodeGenerator:
    ast: ast_.ASTNode

    def generate(self) -> Any:
        if self.ast._env is None:
            raise ValueError("AST environment should have been populated before code generation (see analysis module).")

        return self._generate_code(self.ast)

    @singledispatchmethod
    def _generate_code(self, ast: ast_.ASTNode) -> Any:
        """Auxiliary function for generating LLVM code based on the type of AST node. See :func:`generate_llvm_code`."""
        raise NotImplementedError(f"Code generation not implemented for node type: {type(ast)} with value {ast}")

    @_generate_code.register
    def _(self, ast: ast_.Identifier) -> Any: ...
