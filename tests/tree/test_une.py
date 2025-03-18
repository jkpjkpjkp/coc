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
        mock_task = Mock()
        codelist = CodeList(CONTEXT_FILE, mock_task)
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
        codelist = CodeList(CONTEXT_FILE, task=None)
        codelist.append("print('Hello')")
        codelist.append("raise ValueError('error')")
        pairs = codelist.to_list_of_pair_of_str()
        self.assertEqual(pairs, [Pair("print('Hello')", "Hello\n")])

class TestTreeNode(unittest.TestCase):
    def test_initialization(self):
        codelist = CodeList(CONTEXT_FILE, task=None)
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

import unittest
from unittest.mock import patch
from coc.tree.une import (
    rollout,
    root_factory,
    force_code_then_answer_at_each_step,
    compare,
    judge_multichoice,
    eval_a_batch,
    TreeNode,
    CodeList,
    Task,
    MAX_DEPTH,
)
from coc.util.misc import fulltask_to_task
from typing import List, Any
from PIL import Image

# Mock Exec class to avoid actual code execution
class MockExec:
    def __init__(self, context_file):
        self.globals = {}
        self.context_file = context_file

    def _run(self, code: str):
        return "dummy output", "", []

    def set_var(self, name, value):
        self.globals[name] = value

    def get_var(self, name):
        return self.globals.get(name)

# Mock LLM class to return predefined responses
class MockLLM:
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0

    def __call__(self, message: str) -> str:
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

# Dummy Task class for testing
class DummyTask(Task):
    pass

class TestTreeFunctions(unittest.TestCase):

    def setUp(self):
        self.task = {
            'query': 'Test query',
            'images': [],
            'choices': ['A', 'B', 'C'],
            'answer': 'A'
        }
        self.max_depth = 2
        self.context_file = "dummy_context.py"

    ### Test for rollout
    def test_rollout(self):
        responses = [
            "```python\nprint(1)\n```",
            "```python\nprint(2)\n```",
            "The answer is \\boxed{42}"
        ]
        mock_llm = MockLLM(responses)
        root = TreeNode(
            codelist=CodeList(context=self.context_file, task=self.task),
            outputs=[],
            parent=None,
            children=[],
            depth=0,
        )
        root.codelist.env = MockExec(self.context_file)
        root.codelist.env.set_var('task', self.task)
        answer, final_node = rollout(self.task, root, mock_llm, max_depth=3, variant='neutral')
        self.assertEqual(answer, "42")
        self.assertEqual(len(root.children), 1)
        child1 = root.children[0]
        self.assertEqual(len(child1.children), 1)
        child2 = child1.children[0]
        self.assertEqual(len(child2.children), 0)
        self.assertEqual(final_node, child2)
        self.assertEqual(len(child2.codelist._), 2)
        self.assertEqual(child2.codelist._[0].code, "print(1)")
        self.assertEqual(child2.codelist._[1].code, "print(2)")

    ### Test for rollout with immediate answer
    def test_rollout_answer_first(self):
        responses = ["The answer is \\boxed{100}"]
        mock_llm = MockLLM(responses)
        root = TreeNode(
            codelist=CodeList(context=self.context_file, task=self.task),
            outputs=[],
            parent=None,
            children=[],
            depth=0,
        )
        root.codelist.env = MockExec(self.context_file)
        root.codelist.env.set_var('task', self.task)
        answer, final_node = rollout(self.task, root, mock_llm, max_depth=3, variant='neutral')
        self.assertEqual(answer, "100")
        self.assertEqual(final_node, root)
        self.assertEqual(len(root.children), 0)

    ### Test for rollout reaching max_depth
    def test_rollout_max_depth(self):
        responses = ["```python\nprint(1)\n```"] * 3 + ["The answer is \\boxed{42}"]
        mock_llm = MockLLM(responses)
        root = TreeNode(
            codelist=CodeList(context=self.context_file, task=self.task),
            outputs=[],
            parent=None,
            children=[],
            depth=0,
        )
        root.codelist.env = MockExec(self.context_file)
        root.codelist.env.set_var('task', self.task)
        answer, final_node = rollout(self.task, root, mock_llm, max_depth=3, variant='neutral')
        self.assertEqual(answer, "42")
        node = root
        for _ in range(3):
            self.assertEqual(len(node.children), 1)
            node = node.children[0]

    ### Test for root_factory
    def test_root_factory(self):
        root = root_factory(self.task)
        self.assertIsInstance(root, TreeNode)
        self.assertEqual(root.codelist.env.get_var('task'), self.task)
        self.assertEqual(root.outputs, [])
        self.assertIsNone(root.parent)
        self.assertEqual(root.children, [])
        self.assertEqual(root.depth, 0)

    ### Test for force_code_then_answer_at_each_step
    def test_force_code_then_answer_at_each_step(self):
        class ForceMockLLM:
            def __init__(self, max_depth):
                self.call_count = 0
                self.max_depth = max_depth
    
            def __call__(self, message):
                self.call_count += 1
                if self.call_count <= self.max_depth:
                    return "```python\nprint({})\n```".format(self.call_count)
                return "The answer is \\boxed{42}"
    
        mock_llm = ForceMockLLM(self.max_depth)
        with patch('coc.tree.une.rollout', side_effect=lambda task, node, llm, max_depth, variant: 
                  (None, node) if variant == 'force code' else ("42", node)):
            ret, node = force_code_then_answer_at_each_step(self.task, self.max_depth)
            self.assertEqual(ret, "42")

    ### Test for compare
    @patch('coc.tool.vqa.gemini_as_llm')
    def test_compare(self, mock_gemini):
        mock_gemini.side_effect = lambda message: "True" if "output: 42" in message and "answer: 42" in message else "False"
        self.assertTrue(compare("42", "42"))
        self.assertFalse(compare("43", "42"))

    ### Test for judge_multichoice
    @patch('coc.tool.vqa.gemini_as_llm')
    def test_judge_multichoice(self, mock_gemini):
        mock_gemini.side_effect = lambda message: "True" if "output (cannot see the choices): A" in message and "answer (correct choice is): A" in message else "False"
        self.assertTrue(judge_multichoice("A", ["A", "B", "C"], "A"))
        self.assertFalse(judge_multichoice("B", ["A", "B", "C"], "A"))

    ### Test for eval_a_batch
    @patch('coc.tree.une.force_code_then_answer_at_each_step')
    @patch('coc.tree.une.judge_multichoice')
    def test_eval_a_batch(self, mock_judge, mock_force):
        batch = [
            {'query': 'Test 1', 'question': 'Test 1', 'choices': ['A', 'B', 'C'], 'answer': 'A', 'images': []},
            {'query': 'Test 2', 'question': 'Test 2', 'choices': ['X', 'Y', 'Z'], 'answer': 'Y', 'images': []},
        ]
        mock_force.side_effect = [("A", None), ("Z", None)]
        mock_judge.side_effect = [True, False]
    
        correct, total = eval_a_batch(batch)
        self.assertEqual(correct, 1)
        self.assertEqual(total, 2)

if __name__ == '__main__':
    unittest.main()
