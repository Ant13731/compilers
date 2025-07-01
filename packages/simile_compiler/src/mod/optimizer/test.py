from src.mod import parse
from src.mod import ast_
from src.mod import analysis
from src.mod.optimizer.rewrite_collection import RewriteCollection
from src.mod.optimizer.rewrite_collections import (
    SetCodeGenerationCollection,
    SetComprehensionConstructionCollection,
    DisjunctiveNormalFormQuantifierPredicateCollection,
    GeneratorSelectionCollection,
)


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

TEST_STR = "card({s · s in {1, 2} or s in {2, 3} | s})"
print("TEST_STR 2:", TEST_STR)
parsed_test_str = parse(TEST_STR)
analyzed_test_str = analysis.populate_ast_environments(parsed_test_str)
comp_constr_test_str1 = SetComprehensionConstructionCollection().normalize(analyzed_test_str)
print("COMP CONSTR TEST STR 2 1:", comp_constr_test_str1.pretty_print())
comp_constr_test_str2 = DisjunctiveNormalFormQuantifierPredicateCollection().normalize(comp_constr_test_str1)
print("COMP CONSTR TEST STR 2 2:", comp_constr_test_str2.pretty_print())
comp_constr_test_str3 = GeneratorSelectionCollection().normalize(comp_constr_test_str2)
print("COMP CONSTR TEST STR 2 2.1:", comp_constr_test_str3.pretty_print())
print("COMP CONSTR TEST STR 2:", comp_constr_test_str3.body.items[0]._selected_generators)

comp_constr_test_str4 = SetCodeGenerationCollection().normalize(comp_constr_test_str3)
print("COMP CONSTR TEST STR 2 3:", comp_constr_test_str4.pretty_print())
