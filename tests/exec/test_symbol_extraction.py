from coc.exec.dag import parse_cell_for_symbols

import unittest

class TestAnalyzeCell(unittest.TestCase):
    def test_parse_cell_basic(self):
        code = "x = 5"
        ret = parse_cell_for_symbols(code)
        defined = ret['defined']
        used = ret['used']
        self.assertEqual(defined, {'x'})
        self.assertEqual(used, set())

    def test_parse_cell_function_scope(self):
        code = "def f():\n    x = 5"
        ret = parse_cell_for_symbols(code)
        defined = ret['defined']
        used = ret['used']
        self.assertEqual(defined, {'f'})
        self.assertEqual(used, set())

    def test_parse_cell_import(self):
        code = "import numpy as np"
        ret = parse_cell_for_symbols(code)
        defined = ret['defined']
        used = ret['used']
        self.assertEqual(defined, {'np'})
        self.assertEqual(used, set())

    def test_parse_cell_comprehension_scope(self):
        code = "[x for x in range(10)]"
        ret = parse_cell_for_symbols(code)
        defined = ret['defined']
        used = ret['used']
        self.assertNotIn('x', defined)
        self.assertEqual(used, {'range'})

    def test_parse_cell_usage_in_function(self):
        code = "y = 5\ndef f():\n    print(y)"
        ret = parse_cell_for_symbols(code)
        defined = ret['defined']
        used = ret['used']
        self.assertEqual(defined, {'y', 'f'})
        self.assertEqual(used, {'print'})

    def test_define_and_use(self):
        code = 'x = x + 1'
        ret = parse_cell_for_symbols(code)
        defined = ret['defined']
        used = ret['used']
        self.assertEqual(defined, {'x'})
        self.assertEqual(used, {'x'})

if __name__ == '__main__':
    unittest.main()