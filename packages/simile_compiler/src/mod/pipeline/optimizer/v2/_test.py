from src.mod.pipeline.optimizer.v2.simrw_parser import *
from src.mod.pipeline.optimizer.v2.simrw import *

SIMRW_FILES = [
    "bag_sugar",
    "bool",
    "comprehension_construction",
    "dnf",
    "rel_sugar",
    "seq_sugar",
    "sum",
]

for simrw_file in SIMRW_FILES:
    rules = convert_simrw_to_rewrite_rules(parse_simrw_file(f"packages/simile_compiler/src/mod/data/rewrite_rules/{simrw_file}.simrw"))
    for rule in rules:
        print(simrw_to_source(rule))
