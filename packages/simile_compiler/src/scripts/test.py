from src.mod import parse
from src.mod import ast_
from src.mod import analysis
from src.mod import collection_optimizer, SET_REWRITE_COLLECTION
from src.mod import RustCodeGenerator, CPPCodeGenerator


# TEST = ast_.Start(
#     ast_.Statements(
#         [
#             ast_.Sum(
#                 ast_.And(
#                     [
#                         ast_.In(
#                             ast_.Identifier("s"),
#                             ast_.SetEnumeration(
#                                 [ast_.Int("1"), ast_.Int("2")],
#                             ),
#                         ),
#                     ]
#                 ),
#                 ast_.Int("1"),
#             ),
#         ]
#     )
# )
# TEST_STR = "card({s · s in {1, 2} | s})"

# print("TEST_STR:", TEST_STR)
# # print("TEST:", TEST.pretty_print())
# parsed_test_str = parse(TEST_STR)
# print(parsed_test_str.body.items[0]._bound_identifiers)
# # print("PARSED TEST_STR:", parsed_test_str.pretty_print())
# analyzed_test = analysis.populate_ast_environments(TEST)
# analyzed_test_str = analysis.populate_ast_environments(parsed_test_str)
# # print("ANALYZED TEST:", analyzed_test.pretty_print())
# # print("ANALYZED PARSED TEST_STR:", analyzed_test_str.pretty_print())
# # comp_constr_test = SetComprehensionConstructionCollection().normalize(analyzed_test)
# comp_constr_test_str1 = SetComprehensionConstructionCollection().normalize(analyzed_test_str)
# comp_constr_test_str2 = DisjunctiveNormalFormQuantifierPredicateCollection().normalize(comp_constr_test_str1)
# comp_constr_test_str3 = SetCodeGenerationCollection().normalize(comp_constr_test_str2)
# # print("COMP CONSTR TEST:", comp_constr_test.pretty_print())
# print(parsed_test_str.body.items[0]._bound_identifiers)
# print(analyzed_test_str.body.items[0]._bound_identifiers)
# print(comp_constr_test_str1.body.items[0]._bound_identifiers)
# print(comp_constr_test_str2.body.items[0]._bound_identifiers)
# # print(comp_constr_test_str3.body.items[0]._bound_identifiers)
# print("COMP CONSTR TEST STR:", comp_constr_test_str1.pretty_print())
# print("COMP CONSTR TEST STR:", comp_constr_test_str2.pretty_print())
# print("COMP CONSTR TEST STR:", comp_constr_test_str3.pretty_print())

# TEST_STR = "card({s · s in {1, 2} or s in {2, 3} | s})"
TEST_STR = "card({1, 2} \\/ {2, 3})"
# TEST_STR = "{1, 2} \\/ {2, 3}"
# len({1, 2} - {2, 3})

print("TEST_STR 2:", TEST_STR)

ast: ast_.ASTNode | list = parse(TEST_STR)
if isinstance(ast, list):
    raise ValueError(f"Expected a single AST, got a list (parsing failed): {ast}")

ast = analysis.populate_ast_environments(ast)
print("PARSED TEST_STR:", ast.pretty_print())
print("PARSED TEST_STR:", ast.pretty_print_algorithmic())

ast = collection_optimizer(ast, SET_REWRITE_COLLECTION)
print("OPTIMIZED TEST_STR:", ast.pretty_print())
print("OPTIMIZED TEST_STR:", ast.pretty_print(print_env=True))
print("OPTIMIZED TEST_STR:", ast.pretty_print_algorithmic())

ast = analysis.populate_ast_environments(ast)
# print("OPTIMIZED TEST_STR:", ast.pretty_print())
# print("OPTIMIZED TEST_STR:", ast.pretty_print(print_env=True))
# print("OPTIMIZED TEST_STR:", ast.pretty_print_algorithmic())

RustCodeGenerator(ast).build()


# TEST_STR_TO_GET_AST = f"""
# counter := 0
# for s in {{1,2}}:
#     expr_var := s
#     counter := counter + 1
# for q in {{2,3}}:
#     expr_var := q
#     if ¬(q ∈ {{1, 2}} ∧ expr_var = q):
#         counter := counter + 1
# """
# ast_to_get = parse(TEST_STR_TO_GET_AST)
# print("TEST_STR_TO_GET_AST:", TEST_STR_TO_GET_AST)
# print("AST TO GET:", ast_to_get.pretty_print())


# print(
#     ast_.structurally_equal(
#         ast,
#         ast_to_get,
#     )
# )


# comp_constr_test_str = SetComprehensionConstructionCollection().normalize(analyzed_test_str)
# print("COMP CONSTR TEST STR 2 1:", comp_constr_test_str.pretty_print())
# comp_constr_test_str = DisjunctiveNormalFormQuantifierPredicateCollection().normalize(comp_constr_test_str)
# print("COMP CONSTR TEST STR 2 2:", comp_constr_test_str.pretty_print())
# comp_constr_test_str = PredicateSimplificationCollection().normalize(comp_constr_test_str)
# print("COMP CONSTR TEST STR 2 3:", comp_constr_test_str.pretty_print())
# comp_constr_test_str = GeneratorSelectionCollection().normalize(comp_constr_test_str)
# print("COMP CONSTR TEST STR 2 4:", comp_constr_test_str.pretty_print())
# # print("COMP CONSTR TEST STR 2:", comp_constr_test_str3.body.items[0]._selected_generators)
# comp_constr_test_str = SetCodeGenerationCollection().normalize(comp_constr_test_str)
# print("COMP CONSTR TEST STR 2 5:", comp_constr_test_str.pretty_print())
# print(parsed_test_str.pretty_print_algorithmic())
# print(comp_constr_test_str.pretty_print_algorithmic())
