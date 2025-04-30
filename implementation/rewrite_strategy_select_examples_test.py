"""
Select Rewrite Strategy Examples

Note that this file is independent from the rest of the compiler, intended only for experimentation and testing.
"""

import unittest
from parameterized import parameterized

from ast_ import *
from parser import parse

from rewrite_strategy_select_examples import *


class TestRewriteStrategy(unittest.TestCase):
    @parameterized.expand([(test_str, expected) for test_str, expected in tests_1.items()])
    def test_rewrite_strategy_1(self, test_str, expected):
        ast = parse(test_str)
        res = test_rewrite_strategy_1(ast)
        self.assertEqual(
            res,
            expected,
            f"Test failed for input: {test_str}.\n Expected: {expected.pretty_print()}\n Got: {res.pretty_print()}.",
        )

    @parameterized.expand([(test_str, expected) for test_str, expected in tests_1_denest.items()])
    def test_rewrite_strategy_1_denest(self, test_str, expected):
        ast = parse(test_str)
        res = test_rewrite_strategy_1(ast)
        res = test_rewrite_strategy_1_denest(res)
        self.assertEqual(
            res,
            expected,
            f"Test failed for input: {test_str}.\n Expected: {expected.pretty_print()}\n Got: {res.pretty_print()}.",
        )

    @parameterized.expand([(test_str, expected) for test_str, expected in tests_2.items()])
    def test_rewrite_strategy_2(self, test_str, expected):
        ast = parse(test_str)
        res = test_rewrite_strategy_1(ast)
        res = test_rewrite_strategy_1_denest(res)
        res = test_rewrite_strategy_2(res)
        self.assertEqual(
            res,
            expected,
            f"Test failed for input: {test_str}.\n Expected: {expected.pretty_print()}\n Got: {res.pretty_print()}.",
        )

    @parameterized.expand([(test_str, expected) for test_str, expected in tests_3.items()])
    def test_rewrite_strategy_3(self, test_str, expected):
        ast = parse(test_str)
        res = test_rewrite_strategy_1(ast)
        res = test_rewrite_strategy_1_denest(res)
        res = test_rewrite_strategy_2(res)
        res = test_rewrite_strategy_3(res)
        self.assertEqual(
            res,
            expected,
            f"Test failed for input: {test_str}.\n Expected: {expected.pretty_print()}\n Got: {res.pretty_print()}.",
        )
