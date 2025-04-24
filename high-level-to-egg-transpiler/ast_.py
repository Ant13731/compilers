from lark import Transformer, Token, Tree


class GrammarToEggTransformer(Transformer):
    # basically we need to define the ast for each rule in the grammar as an S-expr so we can convert it to egg for optimization

    # only leaf values need to deal with tokens, higher up the tree will see the realized values
    def INT(self, token: Token) -> str:
        return str(int(token.value))

    def FLOAT(self, token: Token) -> str:
        return str(float(token.value))

    def STRING(self, token: Token) -> str:
        return str(token.value)

    def NAME(self, token: Token) -> str:
        return str(token.value)

    def power(self, tokens: list[str]) -> str:
        return f"(Power {' '.join(tokens)})"

    def factor(self, tokens: list[str]) -> str:
        return f"(Factor {' '.join(tokens)})"

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
        return str(tokens[0].value)

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
        return f"(Implies {tokens[0]} {tokens[2]})"

    def iterable_names(self, tokens: list[str]) -> str:
        return f"(IterableNames {' '.join(tokens)})"

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
        print(tokens)
        return f"(Assignment {' '.join(tokens)})"

    def statements(self, tokens: list[str]) -> str:
        return f"(Statements {' '.join(tokens)})"

    def start(self, tokens: list[str]) -> str:
        return f"(Start {' '.join(tokens)})"

    # def assignment(self, tokens: list[Token]) ->
