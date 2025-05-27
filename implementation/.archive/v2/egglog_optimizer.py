from __future__ import annotations
from egglog import *
import ast_
from egglog_types import *

egraph = EGraph()


# @function
# def add(a: Int.Egged, b: Int.Egged) -> Int.Egged: ...


# @function
# def neg1(a: Int.Egged) -> Int.Egged: ...


# expr1 = add(neg1(neg1(Int.egg_init(2))), add(add(Int.egg_init(2), Int.egg_init(3)), Int.egg_init(4)))
# a, b, c = vars_("a b c", Int.Egged)

# egraph.register(expr1)
# res = egraph.run(
#     ruleset(
#         rewrite(add(a, b)).to(add(b, a)),
#         rewrite(add(add(a, b), c)).to(add(a, add(b, c))),
#         rewrite(add(a, add(b, c))).to(add(add(a, b), c)),
#         rewrite(neg1(neg1(a))).to(a),
#     ),
# )

exprs = [Neg(Neg(Neg(Primitive(Int(2))))).to_egg()]
# expr2 = Add(Add(Int(2), Int(3)), Neg(Neg(Int(4)))).to_egg()
# expr3 = Primitive(Int(2)).to_egg()
a, b, c = vars_("a b c", Primitive.Egged)

egraph.register(*exprs)
res = egraph.run(
    ruleset(
        rewrite(Neg.to_egg_func()(Neg.to_egg_func()(a))).to(a),
        # rewrite(Add.to_egg_func()(Add.to_egg_func()(a, b), c)).to(Add.to_egg_func()(a, Add.to_egg_func()(b, c))),
    ),
)
print(res)
for expr in exprs:
    print(egraph.extract(expr))

# def high_level_lang_optimizer():
#     egraph = EGraph()

#     # DataTypes
#     # Primitives

#     # Complex Literals (collections, iterables, etc.)


# high_level_lang_optimizer()
