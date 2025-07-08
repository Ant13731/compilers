from __future__ import annotations
from functools import singledispatchmethod
from dataclasses import dataclass, field
from typing import ClassVar, Any, Generic, TypeVar

from src.mod import ast_

T = TypeVar("T")


class CodeGeneratorError(Exception):
    """Custom exception for code generation errors."""

    pass


@dataclass
class CodeGeneratorEnvironment(Generic[T]):
    previous: CodeGeneratorEnvironment | None = None
    table: dict[str, T] = field(default_factory=dict)

    def put(self, key: str, value: T) -> None:
        self.table[key] = value

    def get(self, s: str) -> T | None:
        current_env: CodeGeneratorEnvironment[T] | None = self
        while current_env is not None:
            if s in current_env.table:
                return current_env.table[s]
            current_env = current_env.previous
        return None


@dataclass
class CodeGenerator:
    ast: ast_.ASTNode

    def generate(self) -> Any:
        if self.ast._env is None:
            raise ValueError("AST environment should have been populated before code generation (see analysis module).")

        return self._generate_code(self.ast)

    def build(self, target_folder_path: str = "packages/simile_compiler/build") -> None:
        raise NotImplementedError("Build method is not implemented. This should be overridden in subclasses.")

    @singledispatchmethod
    def _generate_code(self, ast: ast_.ASTNode) -> Any:
        """Auxiliary function for generating LLVM code based on the type of AST node. See :func:`generate_llvm_code`."""
        raise NotImplementedError(f"Code generation not implemented for node type: {type(ast)} with value {ast}")
