from dataclasses import dataclass, fields, asdict, field
import types

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
        return f"({cls.__name__} {' '.join(ordered_field_types)})"


# PRIMITIVE LITERALS, take in a Token and return an AST node


@dataclass
class Int(BaseEggAST):
    value: int


@dataclass
class Float(BaseEggAST):
    value: float


@dataclass
class String(BaseEggAST):
    value: str


@dataclass
class None_(BaseEggAST):
    pass


@dataclass
class Bool(BaseEggAST):
    value: bool


@dataclass
class Identifier(BaseEggAST):
    name: str


# Operators
@dataclass
class BinOp(BaseEggAST):
    left: BaseEggAST
    right: BaseEggAST


@dataclass
class UnaryOp(BaseEggAST):
    value: BaseEggAST


class Pos(UnaryOp):
    pass


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
    right: BaseEggAST | None_ = field(default_factory=lambda: None_())


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
    index: BaseEggAST | None_ = field(default_factory=lambda: None_())


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
    value: BaseEggAST | None_


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
    else_body: BaseEggAST | None_ = field(default_factory=lambda: None_())


@dataclass
class Elif(BaseEggAST):
    condition: BaseEggAST
    body: BaseEggAST
    else_body: BaseEggAST | None_ = field(default_factory=lambda: None_())


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
    condition: BaseEggAST | None_
    action: BaseEggAST


@dataclass
class KeyPairComprehension(BaseEggAST):
    iterable_names: BaseEggAST
    loop_over: BaseEggAST
    condition: BaseEggAST | None_
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
    body: BaseEggAST | None_


@dataclass
class Statements(BaseEggAST):
    statements: list[BaseEggAST]
