from __future__ import annotations
from dataclasses import dataclass, fields, asdict, field
import types
from typing import TypeVar, Generic, Callable, ClassVar, Any, TypeAlias

from egglog import (
    Expr,
    i64Like,
    f64Like,
    BoolLike,
    StringLike,
    function,
)

py_to_rust_type_map = {
    "int": "i64",
    "str": "String",
    "float": "f64",
    "bool": "bool",
    "None": "None",
    "list": "Vec",
    "dict": "HashMap",
    "set": "HashSet",
    "tuple": "Tuple",
    "BaseEggAST": "Id",
}


@dataclass
class BaseEggAST:
    def to_s_expr(self) -> str:
        """Converts the AST node to an S-expression."""
        if len(fields(self)) == 0:
            return f"({self.__class__.__name__})"

        ordered_fields = []
        for field in asdict(self):
            field_value = getattr(self, field)
            if isinstance(field_value, BaseEggAST):
                ordered_fields.append(field_value.to_s_expr())
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, BaseEggAST):
                        ordered_fields.append(item.to_s_expr())
                    else:
                        ordered_fields.append(str(item))
            else:
                ordered_fields.append(str(field_value))
        return f"({self.__class__.__name__} {' '.join(ordered_fields)})"

    @classmethod
    def to_abstract_s_expr(cls) -> str:
        """Converts the AST node to an abstract S-expression (for use in passing to Rust's egg crate)."""

        if len(fields(cls)) == 0:
            return f"({cls.__name__})"

        ordered_field_types = []
        for field in fields(cls):
            field_type = field.type
            if hasattr(field_type, "__name__"):
                type_name = field_type.__name__
                if type_name in py_to_rust_type_map:
                    ordered_field_types.append(py_to_rust_type_map[type_name])
                else:
                    raise TypeError(f"Failed to convert field type {field_type} with name {type_name} to string (not found in mapping py_to_rust_type_map).")
            else:
                raise TypeError(f"Failed to convert field type {field_type} to string (could not find name of field type).")
                # Field type {field_type} is not a valid type for S-expression conversion (failed to convert to string).")
            # else:
            # ordered_field_types.append(str(field_type))
        return f"{cls.__name__}({', '.join(ordered_field_types)})"

    def to_egg(self) -> Expr:
        """Converts the AST node to an egglog expression."""
        raise NotImplementedError(f"to_egg() not implemented for {self.__class__.__name__}")


# PRIMITIVE LITERALS, take in a Token and return an AST node

# Egglog doesnt support generic types yet, so have to do it manually. Otherwise, we could just add this to the inheritance chain
# T = TypeVar("T")
# class EggLiteral(Generic[T]):

#     class Egged(Expr):
#         def __init__(self, value: T) -> None: ...

#     @classmethod
#     def egg_init(cls, value: T) -> Egged:
#         return cls.Egged(value)


@dataclass
class Int(BaseEggAST):
    value: int

    # TODO figure out a way to get rid of the I in EggedI
    # - needed because egglog automatically converts everything to a runtimeexpr, which unfortunately
    #   removes the parent class name (and any and all parameters, methods, properties, etc.) from the object
    class EggedI(Expr):
        def __init__(self, value: i64Like) -> None: ...

    Egged: ClassVar[type] = EggedI

    @classmethod
    def egg_init(cls, value: i64Like) -> Egged:
        return cls.Egged(value)

    def to_egg(self) -> Egged:
        return self.Egged(self.value)


@dataclass
class Float(BaseEggAST):
    value: float

    class EggedF(Expr):
        def __init__(self, value: f64Like) -> None: ...

    Egged: ClassVar[type] = EggedF

    @classmethod
    def egg_init(cls, value: f64Like) -> Egged:
        return cls.Egged(value)

    def to_egg(self) -> Egged:
        return self.Egged(self.value)


@dataclass
class String(BaseEggAST):
    value: str

    class EggedS(Expr):
        def __init__(self, value: StringLike) -> None: ...

    Egged: ClassVar[type] = EggedS

    @classmethod
    def egg_init(cls, value: StringLike) -> Egged:
        return cls.Egged(value)

    def to_egg(self) -> Egged:
        return self.Egged(self.value)


@dataclass
class None_(BaseEggAST):
    class EggedN(Expr):
        def __init__(self) -> None: ...

    Egged: ClassVar[type] = EggedN

    @classmethod
    def egg_init(cls) -> Egged:
        return cls.Egged()

    def to_egg(self) -> Egged:
        return self.Egged()


@dataclass
class Bool(BaseEggAST):
    value: bool

    class EggedB(Expr):
        def __init__(self, value: BoolLike) -> None: ...

    Egged: ClassVar[type] = EggedB

    @classmethod
    def egg_init(cls, value: BoolLike) -> Egged:
        return cls.Egged(value)

    def to_egg(self) -> Egged:
        return self.Egged(self.value)


@dataclass
class Primitive(BaseEggAST):
    # Assume types are already checked by this point?
    # TODO auto detect the type to make it easier to write
    value: Int | Float | Bool | None_ | String

    class EggedP(Expr):
        def __init__(self, value: Int.Egged | Float.Egged | Bool.Egged | None_.Egged | String.Egged) -> None: ...

    Egged: ClassVar[type] = EggedP

    def to_egg(self) -> Egged:
        return self.Egged(self.value.to_egg())


@dataclass
class Identifier(BaseEggAST):
    value: str

    class EggedId(Expr):
        def __init__(self, value: StringLike) -> None: ...

    Egged: ClassVar[type] = EggedId

    @classmethod
    def egg_init(cls, value: StringLike) -> Egged:
        return cls.Egged(value)

    # TODO move this to base class and use for both function like and class like AST nodes
    def to_egg(self) -> Egged:
        return self.Egged(self.value)


# def primitive_class_identifier(value) -> str:
#     class_decl = value.__egg_class_decl__
#     if class_decl == "EggedI":
#         return "Int"


# Operators
@dataclass
class BinOp(BaseEggAST):
    left: BaseEggAST
    right: BaseEggAST

    @classmethod
    def to_egg_func(cls) -> Callable:
        func_prefix = cls.__name__.lower()

        def prim_prim_func(left: Primitive.Egged, right: Primitive.Egged) -> Primitive.Egged: ...

        prim_prim_func.__name__ = f"{func_prefix}_p_p"
        prim_prim_func = function(prim_prim_func)

        def prim_id_func(left: Primitive.Egged, right: Identifier.Egged) -> Primitive.Egged: ...

        prim_id_func.__name__ = f"{func_prefix}_p_id"
        prim_id_func = function(prim_id_func)

        def id_prim_func(left: Identifier.Egged, right: Primitive.Egged) -> Primitive.Egged: ...

        id_prim_func.__name__ = f"{func_prefix}_id_p"
        id_prim_func = function(id_prim_func)

        def id_id_func(left: Identifier.Egged, right: Identifier.Egged) -> Identifier.Egged: ...

        id_id_func.__name__ = f"{func_prefix}_id_id"
        id_id_func = function(id_id_func)

        def func(left: Primitive.Egged | Identifier.Egged, right: Primitive.Egged | Identifier.Egged) -> Expr:
            if left.__egg_class_name__ == "EggedP" and right.__egg_class_name__ == "EggedP":
                return prim_prim_func(left, right)
            if left.__egg_class_name__ == "EggedP" and right.__egg_class_name__ == "EggedId":
                return prim_id_func(left, right)
            if left.__egg_class_name__ == "EggedId" and right.__egg_class_name__ == "EggedP":
                return id_prim_func(left, right)
            if left.__egg_class_name__ == "EggedId" and right.__egg_class_name__ == "EggedId":
                return id_id_func(left, right)
            else:
                raise TypeError(f"Unsupported type(s) for {func_prefix}: {left..__egg_class_name__}, {right.__egg_class_name__}")

        return func

    def to_egg(self) -> Expr:
        return self.to_egg_func()(self.left.to_egg(), self.right.to_egg())


@dataclass
class UnaryOp(BaseEggAST):
    value: BaseEggAST

    @classmethod
    def to_egg_func(cls, func_prefix: str | None = None) -> Callable:
        func_prefix = func_prefix or cls.__name__.lower()

        def primitive_func(value: Primitive.Egged) -> Primitive.Egged: ...

        primitive_func.__name__ = f"{func_prefix}_p"
        primitive_func = function(primitive_func)  # need to manually apply the decorator after dynamic name change

        def identifier_func(value: Identifier.Egged) -> Identifier.Egged: ...

        identifier_func.__name__ = f"{func_prefix}_id"
        identifier_func = function(identifier_func)

        def func(value: Primitive.Egged | Identifier.Egged) -> Expr:
            if value.__egg_class_name__ == "EggedP":
                return primitive_func(value)
            if value.__egg_class_name__ == "EggedId":
                return identifier_func(value)
            else:
                raise TypeError(f"Unsupported type for {func_prefix}: {value.__egg_class_name__}")

        return func

        # raise NotImplementedError(f"to_egg_func() not implemented for {cls.__name__}")

    def to_egg(self) -> Expr:
        return self.to_egg_func()(self.value.to_egg())


class Neg(UnaryOp):
    pass


class Not(UnaryOp):
    pass


# BinOp classes
class Add(BinOp):
    pass



class Sub(BinOp):
    pass



class Mul(BinOp):
    pass



class Div(BinOp):
    pass



class Mod(BinOp):
    pass



class Rem(BinOp):
    pass



class Lt(BinOp):
    pass



class Le(BinOp):
    pass



class Gt(BinOp):
    pass



class Ge(BinOp):
    pass



class Eq(BinOp):
    pass



class Ne(BinOp):
    pass



class In(BinOp):
    pass



class NotIn(BinOp):
    pass



class Is(BinOp):
    pass



class IsNot(BinOp):
    pass


class And(BinOp):
    pass



class Or(BinOp):
    pass



class Implies(BinOp):
    pass



@dataclass
class Power(BaseEggAST):
    left: BaseEggAST
    right: BaseEggAST = field(default_factory=lambda: None_())

    to_egg_func: ClassVar[Callable] = BinOp.to_egg_func
    to_egg: ClassVar[Callable] = BinOp.to_egg


# Calling
@dataclass
class StructAccess(BaseEggAST):
    struct: BaseEggAST
    field_name: str




@dataclass
class Call(BaseEggAST):
    function: BaseEggAST
    arguments: BaseEggAST


@dataclass
class Arguments(BaseEggAST):
    arguments: list[BaseEggAST]


@dataclass
class TypedName(BaseEggAST):
    name: BaseEggAST
    type: BaseEggAST


@dataclass
class Indexing(BaseEggAST):
    target: BaseEggAST
    index: BaseEggAST = field(default_factory=lambda: None_())


@dataclass
class Slice(BaseEggAST):
    slices: list[BaseEggAST]


# Statements
@dataclass
class Assignment(BaseEggAST):
    target: BaseEggAST
    value: BaseEggAST


@dataclass
class ArgDef(BaseEggAST):
    args: list[BaseEggAST]


@dataclass
class Return(BaseEggAST):
    value: BaseEggAST


@dataclass
class LambdaDef(BaseEggAST):
    args: BaseEggAST
    body: BaseEggAST


@dataclass
class Break(BaseEggAST):
    pass


@dataclass
class Continue(BaseEggAST):
    pass


@dataclass
class Type_(BaseEggAST):
    type_: BaseEggAST


# Compound Statements
@dataclass
class If(BaseEggAST):
    condition: BaseEggAST
    body: BaseEggAST
    else_body: BaseEggAST = field(default_factory=lambda: None_())


@dataclass
class Elif(BaseEggAST):
    condition: BaseEggAST
    body: BaseEggAST
    else_body: BaseEggAST = field(default_factory=lambda: None_())


@dataclass
class Else(BaseEggAST):
    body: BaseEggAST


@dataclass
class For(BaseEggAST):
    iterable_names: BaseEggAST
    loop_over: BaseEggAST
    body: BaseEggAST


@dataclass
class Struct(BaseEggAST):
    name: BaseEggAST
    fields: list[BaseEggAST]


@dataclass
class Enum(BaseEggAST):
    name: BaseEggAST
    variants: list[BaseEggAST]


@dataclass
class Func(BaseEggAST):
    name: BaseEggAST
    args: BaseEggAST
    return_type: BaseEggAST
    body: BaseEggAST


# Complex Literals (collections, iterables, etc.)
@dataclass
class IterableNames(BaseEggAST):
    names: list[BaseEggAST]


@dataclass
class SuchThat(BaseEggAST):
    condition: BaseEggAST


@dataclass
class Comprehension(BaseEggAST):
    iterable_names: BaseEggAST
    loop_over: BaseEggAST
    condition: BaseEggAST
    action: BaseEggAST


@dataclass
class KeyPairComprehension(BaseEggAST):
    iterable_names: BaseEggAST
    loop_over: BaseEggAST
    condition: BaseEggAST
    action: BaseEggAST


@dataclass
class Tuple(BaseEggAST):
    elements: list[BaseEggAST]


@dataclass
class List(BaseEggAST):
    elements: list[BaseEggAST]


@dataclass
class Dict(BaseEggAST):
    elements: list[BaseEggAST]


@dataclass
class Set(BaseEggAST):
    elements: list[BaseEggAST]


@dataclass
class KeyPair(BaseEggAST):
    key: BaseEggAST
    value: BaseEggAST


# Top level AST nodes
@dataclass
class Start(BaseEggAST):
    body: BaseEggAST


@dataclass
class Statements(BaseEggAST):
    statements: list[BaseEggAST]
