from coc.exec.dag import build_dependency_dag

import unittest

class TestAnalyzeCell(unittest.TestCase):
    def test_build_dag_sequential(self):
        cells = [
            "x = 5",
            "y = x + 1",
            "z = y + x"
        ]
        expected = {
            0: set(),
            1: {0},
            2: {0, 1}
        }
        self.assertEqual(build_dependency_dag(cells), expected)

    def test_build_dag_reassignment(self):
        cells = [
            "x = 0",
            "x = x + 1",
            "y = x + 2"
        ]
        expected = {
            0: set(),
            1: {0},
            2: {1}
        }
        self.assertEqual(build_dependency_dag(cells), expected)

    def test_build_dag_import_dependency(self):
        cells = [
            "import numpy as np",
            "data = np.array([1,2,3])"
        ]
        expected = {
            0: set(),
            1: {0}
        }
        self.assertEqual(build_dependency_dag(cells), expected)
