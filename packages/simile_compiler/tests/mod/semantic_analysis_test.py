from __future__ import annotations

import pytest
import pathlib

from src.mod.data import ast_
from src.mod.pipeline.parser import parse
from src.mod.pipeline.analysis import SemanticAnalysis, semantic_analysis
from src.mod.data.standard_library import STANDARD_LIBRARY_FILES


class TestAnalysisHappyPath:
    @pytest.mark.parametrize("standard_library_path", STANDARD_LIBRARY_FILES)
    def test_standard_library_parses(self, standard_library_path: pathlib.Path):
        with open(standard_library_path, "r") as f:
            source = f.read()
        parse(source, standard_library_path)

    @pytest.mark.parametrize("standard_library_path", STANDARD_LIBRARY_FILES)
    def test_standard_library_passes_semantic_analysis(self, standard_library_path: pathlib.Path):
        with open(standard_library_path, "r") as f:
            source = f.read()
        ast = parse(source, standard_library_path)
        analysis = semantic_analysis(ast)
