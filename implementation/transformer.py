from lark import Transformer, Token, Tree

from ast_ import *


class EggASTTransformer(Transformer):
    """The functions in this Transformer class correspond to grammar rules of a lark CFG.

    Some functions are omitted since lark may automatically inline rules with only one child.
    """

    # PRIMITIVE LITERALS, take in a Token and return an AST node

    def INT(self, token: Token) -> Int:
        return Int(int(token.value))

    def FLOAT(self, token: Token) -> Float:
        return Float(float(token.value))

    def STRING(self, token: Token) -> String:
        return String(token.value)

    def NONE(self, token: Token) -> None_:
        return None_()

    def BOOL(self, token: Token) -> Bool:
        return Bool(token.value)

    def NAME(self, token: Token) -> Identifier:
        return Identifier(token.value)

    # Operators
    def power(self, tokens: list[PrimaryStmt | UnaryOp | BinOp]) -> Power:
        return Power(*tokens)

    def factor(self, tokens: tuple[str, PrimaryStmt | UnaryOp | BinOp]) -> UnaryOp:
        map_symbol_to_ast = {
            "+": Pos,
            "-": Neg,
            "~": Complement,
            r"\powerset": PowerSet,
        }
        if tokens[0] in map_symbol_to_ast:
            return map_symbol_to_ast[tokens[0]](tokens[1])
        raise ValueError(f"Unexpected first token in `factor`: {tokens}")

    def term(self, tokens: tuple[PrimaryStmt | UnaryOp | BinOp, str, PrimaryStmt | UnaryOp | BinOp]) -> BinOp:
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

    def num_and_set_expr(self, tokens: tuple[PrimaryStmt | UnaryOp | BinOp, str, PrimaryStmt | UnaryOp | BinOp]) -> BinOp:
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

    def un_op(self, tokens: list[Token]) -> str:
        return " ".join([str(token.value) for token in tokens])

    def comparison(self, tokens: tuple[PrimaryStmt | UnaryOp | BinOp, str, PrimaryStmt | UnaryOp | BinOp]) -> BinOp:
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

    def negation(self, tokens: list[PrimaryStmt | UnaryOp | BinOp]) -> Not:
        return Not(tokens[1])

    def conjunction(self, tokens: list[PrimaryStmt | UnaryOp | BinOp]) -> And:
        return And(tokens)

    def disjunction(self, tokens: list[PrimaryStmt | UnaryOp | BinOp]) -> Or:
        return Or(tokens)

    def impl(self, tokens: list[PrimaryStmt | UnaryOp | BinOp]) -> Implies:
        return Implies(*tokens)

    def rev_impl(self, tokens: list[PrimaryStmt | UnaryOp | BinOp]) -> RevImplies:
        return RevImplies(*tokens)

    def equivalence(self, tokens: tuple[PrimaryStmt | UnaryOp | BinOp, str, PrimaryStmt | UnaryOp | BinOp]) -> EquivalenceStmt:
        if len(tokens) == 1:
            return EquivalenceStmt(tokens[0])
        map_symbol_to_ast = {
            "<==>": Equiv,
            "<!==>": NotEquiv,
        }
        return EquivalenceStmt(map_symbol_to_ast[tokens[1]](tokens[0], tokens[2]))

    # Calling
    def call(self, tokens: tuple[PrimaryStmt, Arguments]) -> Call:
        return Call(*tokens)

    def arguments(self, tokens: list[Expr]) -> Arguments:
        return Arguments(tokens)

    def typed_name(self, tokens: tuple[Identifier, Type_]) -> TypedName:
        return TypedName(*tokens)

    # def slice_or_index(self, tokens: list[str]) -> str:
    #     if len(tokens) == 0:
    #         return ""
    #     return tokens[0]

    def indexing(self, tokens: tuple[PrimaryStmt, Slice | EquivalenceStmt | None_]) -> Indexing:
        return Indexing(*tokens)

    def slice(self, tokens: list[None | EquivalenceStmt]) -> Slice:
        return Slice(list(map(lambda x: None_() if x is None else x, tokens)))

    # Statements
    def expr_stmt(self, tokens: list[EquivalenceStmt | LambdaDef]) -> Expr:
        return Expr(tokens[0])

    def assignment(self, tokens: tuple[TypedName, Expr]) -> Assignment:
        return Assignment(*tokens)

    def arg_def(self, tokens: list[TypedName]) -> ArgDef:
        return ArgDef(tokens)

    def return_stmt(self, tokens: list[Expr]) -> Return:
        if len(tokens) == 0:
            return Return(None_())
        return Return(*tokens)

    def lambdef(self, tokens: tuple[ArgDef, EquivalenceStmt]) -> LambdaDef:
        return LambdaDef(*tokens)

    def control_flow_stmt(self, tokens: list[str | BaseAST]) -> BaseAST:
        map_symbol_to_ast = {
            "break": Break,
            "continue": Continue,
            "pass": Pass,
        }
        if isinstance(tokens[0], str) and tokens[0] in map_symbol_to_ast:
            return map_symbol_to_ast[tokens[0]]()
        assert not isinstance(tokens[0], str)  # if this assertion fails, that means we got a string value we didnt expect
        return tokens[0]

    def type_(self, tokens: list[Expr]) -> Type_:
        return Type_(*tokens)

    # Compound Statements
    def if_stmt(self, tokens: tuple[EquivalenceStmt, SimpleStatement | Statements, Elif | Else | None_]) -> If:
        return If(*tokens)

    def elif_stmt(self, tokens: tuple[EquivalenceStmt, SimpleStatement | Statements, Elif | Else | None_]) -> Elif:
        return Elif(*tokens)

    def else_stmt(self, tokens: tuple[SimpleStatement | Statements]) -> Else:
        return Else(*tokens)

    def for_stmt(self, tokens: tuple[IterableNames, Expr, SimpleStatement | Statements]) -> For:
        return For(*tokens)

    def struct_stmt(self, tokens: list[Identifier | TypedName]) -> Struct:
        return Struct(tokens[0], tokens[1:])  # type: ignore

    def enum_stmt(self, tokens: list[Identifier]) -> Enum:
        return Enum(tokens[0], tokens[1:])

    def func_stmt(self, tokens: tuple[Identifier, ArgDef, Type_, SimpleStatement | Statements]) -> Func:
        return Func(*tokens)

    # Complex Literals (collections, iterables, etc.)

    def iterable_names(self, tokens: list[Identifier]) -> IterableNames:
        return IterableNames(tokens)

    def list_comp(self, tokens: list[Expr]) -> ListComprehension:
        return ListComprehension(*tokens)

    def set_comp(self, tokens: list[Expr]) -> SetComprehension:
        return SetComprehension(*tokens)

    def dict_comp(self, tokens: tuple[KeyPair, Expr]) -> DictComprehension:
        return DictComprehension(*tokens)

    def bag_comp(self, tokens: list[Expr]) -> BagComprehension:
        return BagComprehension(*tokens)

    def tuple_(self, tokens: list[Expr]) -> BaseAST:
        return Tuple(tokens)

    def list_(self, tokens: list[Expr]) -> BaseAST:
        return List(tokens)

    def set_(self, tokens: list[Expr]) -> BaseAST:
        return Set(tokens)

    def key_pair(self, tokens: tuple[Expr, Expr]) -> BaseAST:
        return KeyPair(*tokens)

    def dict_(self, tokens: list[KeyPair]) -> BaseAST:
        return Dict(tokens)

    # Top level AST nodes
    def statements(self, tokens: list[SimpleStatement | CompoundStatement]) -> Statements:
        return Statements(tokens)

    def start(self, tokens: list[Statements]) -> BaseAST:
        if len(tokens) == 0:
            return Start(None_())
        return Start(tokens[0])
