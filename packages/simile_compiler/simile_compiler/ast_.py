from typing import TypeVar, Any

try:
    from .ast_generated import *
except ImportError:
    from ast_generated import *  # type: ignore


# ListOp should be in the namespace after ast_generator() is called
L = TypeVar("L", bound=ListOp)  # type: ignore # noqa


def flatten_and_join(obj_lst: list[Any], type_: type[L]) -> L:
    """Flatten a list of objects and join them with the given ListOp."""
    flattened_objs = []
    for obj in obj_lst:
        if isinstance(obj, type_):
            flattened_objs += obj.items
        else:
            flattened_objs.append(obj)
    return type_(items=flattened_objs)
