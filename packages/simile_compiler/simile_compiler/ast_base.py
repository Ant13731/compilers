from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from typing import TypeVar, Callable, Any

T = TypeVar("T")


def dataclass_traverse(
    traversal_target: Any,
    visit: Callable[[Any], T],
    visit_leaves: bool = False,
) -> list[T]:
    accumulator: list[T] = []
    _dataclass_traverse_helper(traversal_target, visit, visit_leaves, accumulator)
    return accumulator


def _dataclass_traverse_helper(
    traversal_target: Any,
    visit: Callable[[Any], T],
    visit_leaves: bool,
    accumulator: list[T],
) -> None:
    assert is_dataclass(traversal_target), "Traversal techniques only work with the dataclass `fields` function"
    accumulator.append(visit(traversal_target))
    for f in fields(traversal_target):
        field_value = getattr(traversal_target, f.name)
        if isinstance(field_value, list):
            for item in field_value:
                if is_dataclass(item):
                    _dataclass_traverse_helper(item, visit, visit_leaves, accumulator)
        elif is_dataclass(field_value):
            _dataclass_traverse_helper(field_value, visit, visit_leaves, accumulator)
        elif visit_leaves:
            accumulator.append(visit(field_value))


@dataclass
class ASTNode:
    """Base class for all AST nodes."""

    def contains(self, node: type[ASTNode]) -> bool:
        """Check if the AST node contains a specific type of node."""
        return any(dataclass_traverse(self, lambda n: isinstance(n, node)))

        # if isinstance(self, node):
        #     return True
        # for f in fields(self):
        #     field_value = getattr(self, f.name)
        #     if isinstance(field_value, list):
        #         for item in field_value:
        #             if isinstance(item, ASTNode) and item.contains(node):
        #                 return True
        #     elif isinstance(field_value, ASTNode) and field_value.contains(node):
        #         return True
        # return False

    def find_all_instances(self, type_: type[ASTNode]) -> list[ASTNode]:
        """Returns a flattened list of all instances of a specific type in the AST.

        Most useful for finding identifiers nested within expressions.
        """
        return list(filter(None, dataclass_traverse(self, lambda n: n if isinstance(n, type_) else None)))
        # if isinstance(self, type_):
        #     return [self]
        # ret = []
        # for f in fields(self):
        #     field_value = getattr(self, f.name)
        #     if isinstance(field_value, list):
        #         for item in field_value:
        #             if isinstance(item, ASTNode):
        #                 ret += item.find_all_instances(type_)
        #     elif isinstance(field_value, ASTNode):
        #         ret += field_value.find_all_instances(type_)
        # return ret

    def pretty_print(
        self,
        ignore_fields: list[str] | None = None,
        indent=2,
    ) -> str:
        if ignore_fields is not None and self.__class__.__name__ in ignore_fields:
            indent_ = ""
            indent -= 2
            ret = ""
        else:
            if isinstance(self, PrimitiveLiteral | Identifier):  # type: ignore # noqa
                if isinstance(self, None_):  # type: ignore # noqa
                    ret = f"{self.__class__.__name__}\n"
                    return ret
                ret = f"{self.__class__.__name__}: {self.value}\n"
                return ret

            ret = f"{self.__class__.__name__}:\n"
            indent_ = indent * " "

        for f in fields(self):
            field_value = getattr(self, f.name)

            if isinstance(field_value, ASTNode):
                if len(fields(self)) == 1:
                    ret += f"{indent_}{field_value.pretty_print(ignore_fields, indent+2)}"
                else:
                    ret += f"{indent_}{f.name}={field_value.pretty_print(ignore_fields,indent + 2)}"
                continue

            if isinstance(field_value, list):
                ret += f"{indent_}[\n"
                for item in field_value:
                    if isinstance(item, ASTNode):
                        ret += f"{indent_}{item.pretty_print(ignore_fields,indent + 2)}"
                    else:
                        ret += f"{indent_}    {item}\n"
                ret += f"{indent_}]\n"
                continue

            ret += f"{indent_}{f.name}**: {field_value}\n"
        return ret


# TODO:
# 1. Finish grammar and parser
# 2. Implement a TRS, based on TRAAT textbook
#
#
# 3. Compare language:
# - want better ease-of-use than python or haskell (maybe rust?)
# - want better performance than hand-optimized C, rust
# - see if our compiler does better than AI generated code
# - Should implement examples in python and haskell, see if they are good enough for testing, maybe find some benchmarks
