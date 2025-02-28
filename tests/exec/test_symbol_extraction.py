from coc.exec.symbol_extraction import parse_cell_for_symbols

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

    # def test_build_dag_sequential(self):
    #     cells = [
    #         "x = 5",
    #         "y = x + 1",
    #         "z = y + x"
    #     ]
    #     expected = {
    #         0: set(),
    #         1: {0},
    #         2: {0, 1}
    #     }
    #     self.assertEqual(build_dependency_dag(cells), expected)

    # def test_build_dag_reassignment(self):
    #     cells = [
    #         "x = 0",
    #         "x = x + 1",
    #         "y = x + 2"
    #     ]
    #     expected = {
    #         0: set(),
    #         1: {0},
    #         2: {1}
    #     }
    #     self.assertEqual(build_dependency_dag(cells), expected)

    # def test_build_dag_import_dependency(self):
    #     cells = [
    #         "import numpy as np",
    #         "data = np.array([1,2,3])"
    #     ]
    #     expected = {
    #         0: set(),
    #         1: {0}
    #     }
    #     self.assertEqual(build_dependency_dag(cells), expected)

if __name__ == '__main__':
    unittest.main()