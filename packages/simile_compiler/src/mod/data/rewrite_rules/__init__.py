import pathlib

SIMRW_FOLDER = pathlib.Path(__file__).resolve().parent
SIMRW_FILE_NAMES = [
    "bag_sugar",
    "bool",
    "comprehension_construction",
    "dnf",
    "rel_sugar",
    "seq_sugar",
    "sum",
]

SIMRW_FILES = [SIMRW_FOLDER / f"{file}.simrw" for file in SIMRW_FILE_NAMES]
