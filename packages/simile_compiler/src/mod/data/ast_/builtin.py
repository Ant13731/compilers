from dataclasses import dataclass
from typing import ClassVar

from src.mod.data.ast_.base import ASTNode
from src.mod.data.ast_.common import Call, Identifier, Symbol


class BuiltinFunctionBase(Call):
    target_name: ClassVar[Identifier]

    def __init__(self, args: list[ASTNode]) -> None:
        super().__init__(self.target_name, args=args)


class BuiltinFuncMin(BuiltinFunctionBase):
    target_name = Identifier("min")


class BuiltinFuncMapMin(BuiltinFunctionBase):
    target_name = Identifier("map_min")


class BuiltinFuncMax(BuiltinFunctionBase):
    target_name = Identifier("max")


class BuiltinFuncMapMax(BuiltinFunctionBase):
    target_name = Identifier("map_max")


class BuiltinFuncChoice(BuiltinFunctionBase):
    target_name = Identifier("choice")


class BuiltinFuncDom(BuiltinFunctionBase):
    target_name = Identifier("dom")


class BuiltinFuncRan(BuiltinFunctionBase):
    target_name = Identifier("ran")


class BuiltinFuncCard(BuiltinFunctionBase):
    target_name = Identifier("card")


class BuiltinFuncSize(BuiltinFunctionBase):
    target_name = Identifier("size")


class BuiltinFuncSum(BuiltinFunctionBase):
    target_name = Identifier("sum")


class BuiltinFuncCast(BuiltinFunctionBase):
    target_name = Identifier("cast")


class BuiltinFuncCastWith(BuiltinFunctionBase):
    target_name = Identifier("cast_with")


class BuiltinFuncPrint(BuiltinFunctionBase):
    target_name = Identifier("print")
