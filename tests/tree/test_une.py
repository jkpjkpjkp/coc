import unittest
from unittest.mock import Mock
from coc.exec import Exec
from coc.util import Pair
from coc.tree.une import *
from coc.exec import CONTEXT_FILE

class TestCode(unittest.TestCase):

    def test_successful_execution(self):
        code = Code("print('Hello')", Exec())
        self.assertEqual(code.code, "print('Hello')")
        self.assertEqual(code.output, "Hello\n")
        self.assertEqual(code.error, "")
        self.assertEqual(code.to_pair_of_str(), Pair("print('Hello')", "Hello\n"))

    def test_execution_with_error(self):
        code = Code("raise ValueError('error')", Exec())
        self.assertEqual(code.code, "raise ValueError('error')")
        self.assertEqual(code.output, "")
        self.assertIn("ValueError: error", code.error)
        self.assertEqual(code.to_pair_of_str(), Pair("raise ValueError('error')", ""))

class TestCodeList(unittest.TestCase):
    def test_append_and_to_list_of_pairs(self):
        codelist = CodeList(CONTEXT_FILE)
        codelist.append("print('Hello')")
        self.assertEqual(len(codelist._), 1)
        self.assertEqual(codelist._[0].output, "Hello\n")

        codelist.append("print('World')")
        self.assertEqual(len(codelist._), 2)
        self.assertEqual(codelist._[1].output, "World\n")

        pairs = codelist.to_list_of_pair_of_str()
        self.assertEqual(pairs, [
            Pair("print('Hello')", "Hello\n"),
            Pair("print('World')", "World\n")
        ])

    def test_exclude_errors(self):
        codelist = CodeList(CONTEXT_FILE)
        codelist.append("print('Hello')")
        codelist.append("raise ValueError('error')")
        pairs = codelist.to_list_of_pair_of_str()
        self.assertEqual(pairs, [Pair("print('Hello')", "Hello\n")])

class TestTreeNode(unittest.TestCase):
    def test_initialization(self):
        codelist = CodeList(CONTEXT_FILE)
        node = TreeNode(
            codelist=codelist,
            outputs=[],
            parent=None,
            children=[],
            depth=0
        )
        self.assertEqual(node.codelist, codelist)
        self.assertEqual(node.outputs, [])
        self.assertIsNone(node.parent)
        self.assertEqual(node.children, [])
        self.assertEqual(node.depth, 0)

if __name__ == '__main__':
    unittest.main()