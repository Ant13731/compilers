import pathlib

SIMRW_FILE_NAMES = [
    "bag_sugar",
    "bool",
    "comprehension_construction",
    "dnf",
    "rel_sugar",
    "seq_sugar",
    "sum",
]

SIMRW_FILES = [pathlib.Path(__file__).resolve().parent / f"{file}.simrw" for file in SIMRW_FILE_NAMES]
