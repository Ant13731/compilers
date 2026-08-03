from __future__ import annotations

import textwrap

import pytest

from src.mod.data import ast_
from src.mod.pipeline.analysis import SemanticAnalysis, semantic_analysis


def prep_test(ast: ast_.ASTNode) -> SemanticAnalysis:
    return semantic_analysis(ast)
