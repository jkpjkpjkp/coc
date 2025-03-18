from coc.exec import CONTEXT_FILE
import unittest
import copy

class TestExec(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT_FILE)

    def test_exec(self):
        result, error, _ = self.exec._run('print("hello")')
        self.assertEqual(result, 'hello\n')
        self.assertEqual(error, '')

from coc.data.muir import LoadMuir
from coc.exec.mod import Exec
class TestTask(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT_FILE)
        self.data_loader = LoadMuir('Counting')

    def test_task(self):
        task = next(self.data_loader.convert_to_tasks())
        self.exec.set_var('tsk', task)
        result, error, _ = self.exec._run('print(tsk["task_type"])')
        self.assertIn('counting', result.lower())
        self.assertEqual(error, '')

        vqa, error, _ = self.exec._run('print(glm(tsk["images"][0], "What is in the picture?"))')
        self.assertTrue('mitten' in vqa or 'glove' in vqa)
        self.assertEqual(error, '')

class TestGPTOutput(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT_FILE)
        self.data_loader = LoadMuir('Ordering')

    def test_gpt_output(self):
        task = next(self.data_loader.convert_to_tasks())
        self.exec.set_var('tsk', task)
        code_str = '''
  from PIL import Image
  import numpy as np

  #   def glm(image: np.ndarray, question: str) -> str:
  #       ...

  image1 = tsk["images"][0]
  image2 = tsk["images"][1]
  dense_caption1 = glm(image1, "Generate dense caption for the image")
  dense_caption2 = glm(image2, "Generate dense caption for the image")

  dense_caption1, dense_caption2

  print(dense_caption1, dense_caption2)
  '''
        result, error, _ = self.exec._run(code_str)
        self.assertIn('simpson', result.lower())
        self.assertEqual(error, '')

class TestClone(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT_FILE)
        self.exec.set_var('x', 42)

    def test_clone(self):
        cloned_exec = copy.deepcopy(self.exec)
        self.assertEqual(cloned_exec.globals['x'], 42)

        cloned_exec.set_var('y', 100)
        self.assertIn('y', cloned_exec.globals)
        self.assertNotIn('y', self.exec.globals)

        cloned_exec._run('x += y')
        self.assertEqual(cloned_exec.globals['x'], 142)
        self.assertEqual(self.exec.globals['x'], 42)

        self.exec._run('y = 200\nx += y')
        self.assertEqual(self.exec.globals['x'], 242)
        self.assertEqual(cloned_exec.globals['x'], 142)

class TestError(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT_FILE)

    def test_error(self):
        # Test syntax error
        result, error, _ = self.exec._run('print("hello"')
        self.assertEqual(result, '')
        self.assertIn('SyntaxError', error)

        # Test runtime error
        result, error, _ = self.exec._run('print(undefined_variable)')
        self.assertEqual(result, '')
        self.assertIn('NameError', error)

        # Test division by zero error
        result, error, _ = self.exec._run('print(1/0)')
        self.assertEqual(result, '')
        self.assertIn('ZeroDivisionError', error)

class TestBbox(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT_FILE)

    def test_bbox_sub(self):
        result, error, _ = self.exec._run(
'''
box = Bbox(
    box=[0,0,1,1],
    score=1.0,
    label='white',
)
print(box['box'])
''')
        self.assertEqual(result, '[0, 0, 1, 1]\n')
        self.assertEqual(error, '')

    def test_bbox_dot(self):
        result, error, _ = self.exec._run(
'''
box = Bbox(
    box=[0,0,1,1],
    score=1.0,
    label='white',
)
print(box.box)
''')
        self.assertEqual(result, '')
        self.assertNotEqual(error, '')

class TestRunWithList(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT_FILE)

    def test_run_with_list_input(self):
        """Test that _run accepts a list of code strings"""
        # Test with a list of code strings
        code_list = [
            "x = 5",
            "y = 10",
            "print(f'Sum: {x + y}')"
        ]
        
        stdout, stderr, _ = self.exec._run(code_list)
        
        # Check that all code blocks were executed
        self.assertEqual(stdout.strip(), "Sum: 15")
        self.assertEqual(stderr, "")
        self.assertEqual(self.exec.globals['x'], 5)
        self.assertEqual(self.exec.globals['y'], 10)

    def test_run_with_mixed_list(self):
        """Test that _run handles a list with both string and non-string elements"""
        # Create a test image
        test_image = None
        try:
            from PIL import Image
            test_image = Image.new('RGB', (10, 10), color='red')
        except ImportError:
            # Skip if PIL is not available
            self.skipTest("PIL not available")
        
        # Test with a mixed list (code strings and an image)
        code_list = [
            "a = 100",
            test_image,  # Non-string element
            "b = 200",
            "print(f'Product: {a * b}')"
        ]
        
        stdout, stderr, _ = self.exec._run(code_list)
        
        # Check that string elements were executed and non-string elements were ignored
        self.assertEqual(stdout.strip(), "Product: 20000")
        self.assertEqual(stderr, "")
        self.assertEqual(self.exec.globals['a'], 100)
        self.assertEqual(self.exec.globals['b'], 200)

    def test_run_empty_list(self):
        """Test that _run handles an empty list gracefully"""
        # Test with an empty list
        stdout, stderr, _ = self.exec._run([])
        
        # Check that execution was handled gracefully
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")

    def test_run_list_with_syntax_error(self):
        """Test that _run handles syntax errors in list items properly"""
        # Test with a list containing syntax errors
        code_list = [
            "valid_code = 42",
            "this is not valid Python",
            "print('This should not execute')"
        ]
        
        stdout, stderr, _ = self.exec._run(code_list)
        
        # Check that the error was captured
        self.assertEqual(stdout, "")
        self.assertIn("SyntaxError", stderr)
        
        # Check that the valid code before the error was executed
        self.assertEqual(self.exec.globals.get('valid_code'), 42)

if __name__ == '__main__':
    unittest.main()