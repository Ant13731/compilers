from lark import Transformer, Token, Tree


class GrammarToEggTransformer(Transformer):
    # basically we need to define the ast for each rule in the grammar as an S-expr so we can convert it to egg for optimization

    # only leaf values need to deal with tokens, higher up the tree will see the realized values
    def INT(self, token: Token) -> str:
        return f"(Int {int(token.value)})"

    def FLOAT(self, token: Token) -> str:
        return f"(Float {float(token.value)})"

    def STRING(self, token: Token) -> str:
        return f"(String {token.value})"

    def NONE(self, token: Token) -> str:
        return f"(None)"

    def BOOL(self, token: Token) -> str:
        return f"(Bool {token.value})"

    def NAME(self, token: Token) -> str:
        return f"(Identifier {token.value})"

    def power(self, tokens: list[str]) -> str:
        return f"(Power {' '.join(tokens)})"

    def factor(self, tokens: list[str]) -> str:
        map_symbol_to_ast = {
            "+": "Pos",
            "-": "Neg",
        }
        if map_symbol_to_ast.get(tokens[0]) == "Pos":
            return tokens[1]
        return f"({map_symbol_to_ast[tokens[0]]} {tokens[1]})"

    def term(self, tokens: list[str]) -> str:
        map_symbol_to_ast = {
            "*": "Mul",
            "/": "Div",
            "%": "Mod",
            "//": "Rem",
        }
        return f"({map_symbol_to_ast[tokens[1]]} {tokens[0]} {tokens[2]})"

    def num_expr(self, tokens: list[str]) -> str:
        map_symbol_to_ast = {
            "+": "Add",
            "-": "Sub",
        }
        return f"({map_symbol_to_ast[tokens[1]]} {tokens[0]} {tokens[2]})"

    def comp_op(self, tokens: list[Token]) -> str:
        return " ".join([str(token.value) for token in tokens])

    def comparison(self, tokens: list[str]) -> str:
        map_symbol_to_ast = {
            "<": "Lt",
            "<=": "Le",
            ">": "Gt",
            ">=": "Ge",
            "==": "Eq",
            "!=": "Ne",
            "in": "In",
            "not in": "NotIn",
            "is": "Is",
            "is not": "IsNot",
        }
        return f"({map_symbol_to_ast[tokens[1]]} {tokens[0]} {tokens[2]})"

    def negation(self, tokens: list[str]) -> str:
        return f"(Not {tokens[1]})"

    def conjunction(self, tokens: list[str]) -> str:
        return f"(And {' '.join(tokens)})"

    def disjunction(self, tokens: list[str]) -> str:
        return f"(Or {' '.join(tokens)})"

    def implication(self, tokens: list[str]) -> str:
        return f"(Implies {' '.join(tokens)})"

    def struct_access(self, tokens: list[str]) -> str:
        return f"(StructAccess {' '.join(tokens)})"

    def call(self, tokens: list[str]) -> str:
        if len(tokens) == 1:
            return f"(Call {tokens[0]})"
        return f"(Call {tokens[0]} {' '.join(tokens[1:])})"

    def arguments(self, tokens: list[str]) -> str:
        if len(tokens) == 0:
            return "(Arguments)"
        return f"(Arguments {' '.join(tokens)})"

    def typed_name(self, tokens: list[str]) -> str:
        return f"(TypedName {tokens[0]} {tokens[1]})"

    def slice_or_index(self, tokens: list[str]) -> str:
        if len(tokens) == 0:
            return ""
        return tokens[0]

    def indexing(self, tokens: list[str]) -> str:
        if len(tokens) == 1:
            return f"(Indexing {tokens[0]})"
        return f"(Indexing {tokens[0]} {' '.join(tokens[1:])})"

    def iterable_names(self, tokens: list[str]) -> str:
        return f"(IterableNames {' '.join(tokens)})"

    def slice(self, tokens: list[str]) -> str:
        # ret = "(Slice "
        # num_colons_seen = 0
        # for token in tokens:
        #     if token == ":":
        #         num_colons_seen += 1
        #         continue
        #     ret += f"(SliceChild {token}) "
        # print(tokens)
        # return ret + ")"
        return f"(Slice {' '.join(list(map(str,tokens)))})"

    # def slice_fst(self, tokens: list[str]) -> str:
    #     return f"(SliceFst {' '.join(tokens)})"
    #     # ret = "(Slice "
    #     # for token in tokens:

    #     # ret += ")"
    #     # # if len(tokens) == 1:
    #     # #     return f"(Slice () {tokens[0]})"
    #     # # print(tokens)
    #     # return tokens

    # def slice_snd(self, tokens: list[str]) -> str:
    #     return f"(SliceSnd {' '.join(tokens)})"

    # def slice_thd(self, tokens: list[str]) -> str:
    #     return f"(SliceThd {tokens[0]})"

    # def slice_snd(self, tokens: list[str]) -> str:
    #     print(tokens)
    #     return f"(SliceSnd {tokens[0]})"

    # def slice_thd(self, tokens: list[str]) -> str:
    #     print(tokens)
    #     return f"(SliceThd {tokens[0]})"

    def such_that(self, tokens: list[str]) -> str:
        return f"(SuchThat {tokens[0]})"

    def comp_stmt(self, tokens: list[str]) -> str:
        print(tokens)
        return f"(CompStmt {' '.join(tokens)})"

    def tuple_(self, tokens: list[str]) -> str:
        return f"(Tuple {' '.join(tokens)})"

    def list_(self, tokens: list[str]) -> str:
        return f"(List {' '.join(tokens)})"

    def set_(self, tokens: list[str]) -> str:
        return f"(Set {' '.join(tokens)})"

    def key_pair(self, tokens: list[str]) -> str:
        return f"(KeyPair {' '.join(tokens)})"

    def dict_(self, tokens: list[str]) -> str:
        return f"(Dict {' '.join(tokens)})"

    def assignment(self, tokens: list[str]) -> str:
        # print(tokens)
        return f"(Assignment {' '.join(tokens)})"

    def arg_def(self, tokens: list[str]) -> str:
        if len(tokens) == 0:
            return "(ArgDef)"
        return f"(ArgDef {' '.join(tokens)})"

    def return_stmt(self, tokens: list[str]) -> str:
        if len(tokens) == 0:
            return "(ReturnStmt)"
        return f"(ReturnStmt {' '.join(tokens)})"

    def lambdef(self, tokens: list[str]) -> str:
        return f"(LambdaDef {' '.join(tokens)})"

    def control_flow_stmt(self, tokens: list[str]) -> str:
        map_symbol_to_ast = {
            "break": "Break",
            "continue": "Continue",
        }
        if tokens[0] in map_symbol_to_ast:
            return f"({map_symbol_to_ast[tokens[0]]})"
        return tokens[0]

    def if_stmt(self, tokens: list[str]) -> str:
        return f"(IfStmt {' '.join(tokens)})"

    def elif_stmt(self, tokens: list[str]) -> str:
        return f"(ElifStmt {' '.join(tokens)})"

    def else_stmt(self, tokens: list[str]) -> str:
        return f"(ElseStmt {' '.join(tokens)})"

    def for_stmt(self, tokens: list[str]) -> str:
        return f"(ForStmt {' '.join(tokens)})"

    def struct_stmt(self, tokens: list[str]) -> str:
        return f"(StructStmt {' '.join(tokens)})"

    def enum_stmt(self, tokens: list[str]) -> str:
        return f"(EnumStmt {' '.join(tokens)})"

    def func_stmt(self, tokens: list[str]) -> str:
        return f"(FuncStmt {' '.join(tokens)})"

    def statements(self, tokens: list[str]) -> str:
        return f"(Statements {' '.join(tokens)})"

    def start(self, tokens: list[str]) -> str:
        return f"(Start {' '.join(tokens)})"

    def type_(self, tokens: list[str]) -> str:
        return f"(Type {' '.join(tokens)})"

    # def assignment(self, tokens: list[Token]) ->
