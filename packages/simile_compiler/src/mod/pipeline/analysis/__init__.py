from src.mod.pipeline.analysis.reserved_keywords import (
    reserved_keywords_check,
    ReservedKeywordErr,
)
from src.mod.pipeline.analysis.normalize_ast import (
    normalize_ast,
    ast_promoter,
    assert_no_parser_only_nodes,
)

from src.mod.pipeline.analysis.type_synthesizer import TypeSynthesizer
from src.mod.pipeline.analysis.type_annotation_resolver import TypeAnnotationResolver
from src.mod.pipeline.analysis.populate_symbol_table import make_symbol_table, PopulateSymbolTable
from src.mod.pipeline.analysis.analysis import SemanticAnalysis, semantic_analysis
