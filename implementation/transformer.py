from lark import Transformer, Token, Tree

from ast_ import *


class EggASTTransformer(Transformer):

    # PRIMITIVE LITERALS, take in a Token and return an AST node

    def INT(self, token: Token) -> BaseEggAST:
        return Int(int(token.value))

    def FLOAT(self, token: Token) -> BaseEggAST:
        return Float(float(token.value))

    def STRING(self, token: Token) -> BaseEggAST:
        return String(token.value)

    def NONE(self, token: Token) -> BaseEggAST:
        return None_()

    def BOOL(self, token: Token) -> BaseEggAST:
        return Bool(token.value)

    def NAME(self, token: Token) -> BaseEggAST:
        return Identifier(token.value)

    # Operators
    def power(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Power(*tokens)

    def factor(self, tokens: tuple[str, BaseEggAST]) -> BaseEggAST:
        map_symbol_to_ast = {
            "+": Pos,
            "-": Neg,
            "~": Complement,
            r"\powerset": PowerSet,
        }
        if tokens[0] == "-":
            return Neg(tokens[1])
        return tokens[1]

    def term(self, tokens: tuple[BaseEggAST, str, BaseEggAST]) -> BaseEggAST:
        map_symbol_to_ast = {
            "*": Mul,
            "/": Div,
            "%": Mod,
            "//": Rem,
            r"\cap": Intersect,
            r"\ctimes": CartesianProduct,
            r"\circ": RelationComposition,
        }
        return map_symbol_to_ast[tokens[1]](tokens[0], tokens[2])

    def num_and_set_expr(self, tokens: tuple[BaseEggAST, str, BaseEggAST]) -> BaseEggAST:
        map_symbol_to_ast = {
            "+": Add,
            "-": Sub,
            r"\cup": Union,
            r"\setminus": Difference,
        }
        return map_symbol_to_ast[tokens[1]](tokens[0], tokens[2])

    def comp_op(self, tokens: list[Token]) -> str:
        return " ".join([str(token.value) for token in tokens])

    def num_and_set_op(self, tokens: list[Token]) -> str:
        return " ".join([str(token.value) for token in tokens])

    def num_and_set_op_mult(self, tokens: list[Token]) -> str:
        return " ".join([str(token.value) for token in tokens])

    def comparison(self, tokens: tuple[BaseEggAST, str, BaseEggAST]) -> BaseEggAST:
        map_symbol_to_ast = {
            "<": Lt,
            "<=": Le,
            ">": Gt,
            ">=": Ge,
            "==": Eq,
            "!=": Ne,
            "in": In,
            "not in": NotIn,
            "is": Is,
            "is not": IsNot,
            r"\subset": Subset,
            r"\subseteq": SubsetEq,
            r"\supset": Superset,
            r"\supseteq": SupersetEq,
        }
        return map_symbol_to_ast[tokens[1]](tokens[0], tokens[2])

    def negation(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Not(tokens[1])

    def conjunction(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return And(*tokens)

    def disjunction(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Or(*tokens)

    def impl(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Implies(*tokens)

    def rev_impl(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return RevImplies(*tokens)

    def equivalence(self, tokens: tuple[BaseEggAST, str, BaseEggAST]) -> BaseEggAST:
        map_symbol_to_ast = {
            "<==>": Equiv,
            "<!==>": NotEquiv,
        }
        return map_symbol_to_ast[tokens[1]](tokens[0], tokens[2])

    # Calling
    def call(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Call(*tokens)

    def arguments(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Arguments(tokens)

    def typed_name(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return TypedName(*tokens)

    # def slice_or_index(self, tokens: list[str]) -> str:
    #     if len(tokens) == 0:
    #         return ""
    #     return tokens[0]

    def indexing(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Indexing(*tokens)

    def slice(self, tokens: list[None | BaseEggAST]) -> BaseEggAST:
        return Slice(list(map(lambda x: None_() if x is None else x, tokens)))

    # Statements
    def assignment(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Assignment(*tokens)

    def arg_def(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return ArgDef(tokens)

    def return_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        if len(tokens) == 0:
            return Return(None_())
        return Return(*tokens)

    def lambdef(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        print("LAMBDEF", tokens)
        return LambdaDef(*tokens)

    def control_flow_stmt(self, tokens: list[str | BaseEggAST]) -> BaseEggAST:
        print(tokens)
        map_symbol_to_ast = {
            "break": Break,
            "continue": Continue,
            "pass": Pass,
        }
        if isinstance(tokens[0], str) and tokens[0] in map_symbol_to_ast:
            return map_symbol_to_ast[tokens[0]]()
        assert not isinstance(tokens[0], str)  # if this assertion fails, that means we got a string value we didnt expect
        return tokens[0]

    def type_(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Type_(*tokens)

    # Compound Statements
    def if_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return If(*tokens)

    def elif_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Elif(*tokens)

    def else_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Else(*tokens)

    def for_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return For(*tokens)

    def struct_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Struct(tokens[0], tokens[1:])

    def enum_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Enum(tokens[0], tokens[1:])

    def func_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Func(*tokens)

    # Complex Literals (collections, iterables, etc.)

    def iterable_names(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return IterableNames(tokens)

    def such_that(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return SuchThat(*tokens)

    def comp_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        if len(tokens) == 3:
            return Comprehension(tokens[0], tokens[1], None_(), tokens[2])
        return Comprehension(*tokens)

    def key_pair_comp_stmt(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        if len(tokens) == 3:
            return KeyPairComprehension(tokens[0], tokens[1], None_(), tokens[2])
        return KeyPairComprehension(*tokens)

    def tuple_(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Tuple(tokens)

    def list_(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return List(tokens)

    def set_(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Set(tokens)

    def key_pair(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return KeyPair(*tokens)

    def dict_(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Dict(tokens)

    # Top level AST nodes

    def start(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        if len(tokens) == 0:
            return Start(None_())
        return Start(tokens[0])

    def statements(self, tokens: list[BaseEggAST]) -> BaseEggAST:
        return Statements(tokens)
