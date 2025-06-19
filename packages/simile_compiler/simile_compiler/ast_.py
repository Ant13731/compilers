from typing import Any

try:
    from .ast_types import *  # noqa: F403
except ImportError:
    from ast_types import *  # type: ignore  # noqa: F403


def flatten_and_join(obj_lst: list[Any], type_: ListBoolType) -> ListOp:  # noqa: F405
    """Flatten a list of objects and join them with the given ListOp."""
    flattened_objs = []
    for obj in obj_lst:
        if isinstance(obj, ListOp) and obj.op_type == type_:  # noqa: F405
            flattened_objs += obj.items
        else:
            flattened_objs.append(obj)
    return ListOp(items=flattened_objs, op_type=type_)  # noqa: F405
