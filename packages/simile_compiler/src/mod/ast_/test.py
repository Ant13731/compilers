from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from typing import ClassVar, Self


class InheritedEqMixinParent:

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            print("Other is not an instance of the same class")
            return False
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            self_value = getattr(self, f.name)
            try:
                other_value = getattr(other, f.name)
            except AttributeError:
                print(f"Attribute {f.name} not found in other")
                return False
            if self_value != other_value:
                print(f"Values for {f.name} do not match: {self_value} != {other_value}")
                return False
        return True


class InheritedEqMixinChild:
    parent_mixin: ClassVar[type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.parent_mixin):
            print("Other is not an instance of the parent mixin")
            return False

        for f in fields(self):
            if f.name.startswith("_"):
                continue
            self_value = getattr(self, f.name)
            try:
                other_value = getattr(other, f.name)
            except AttributeError:
                print(f"Attribute {f.name} not found in other")
                return False
            if self_value != other_value:
                print(f"Values for {f.name} do not match: {self_value} != {other_value}")
                return False
        return True


@dataclass(eq=False)
class A(InheritedEqMixinParent):
    a: int


@dataclass
class B(A):
    pass


a = A(1)
b = B(1)
c = B(1)

print(a == b)
print(b == a)
print(b == c)
print(a.__eq__(b))
print(b.__eq__(a))
