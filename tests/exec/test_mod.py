from coc.exec import CONTEXT_FILE as CONTEXT
import unittest

class TestExec(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT)

    def test_exec(self):
        result, error = self.exec._run('print("hello")')
        self.assertEqual(result, 'hello\n')
        self.assertEqual(error, '')

from coc.data.muir import LoadMuir
from coc.exec.mod import Exec
class TestTask(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT)
        self.data_loader = LoadMuir('Counting')

    def test_task(self):
        task = next(self.data_loader.convert_to_tasks())
        self.exec.add_var('tsk', task)
        result, error = self.exec._run('print(tsk["task_type"])')
        self.assertIn('counting', result.lower())
        self.assertEqual(error, '')

        vqa, error = self.exec._run('print(glm(tsk["images"][0], "What is in the picture?"))')
        self.assertTrue('mitten' in vqa or 'glove' in vqa)
        self.assertEqual(error, '')

class TestGPTOutput(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT)
        self.data_loader = LoadMuir('Ordering')

    def test_gpt_output(self):
        task = next(self.data_loader.convert_to_tasks())
        self.exec.add_var('tsk', task)
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
        result, error = self.exec._run(code_str)
        self.assertIn('simpson', result.lower())
        self.assertEqual(error, '')

class TestClone(unittest.TestCase):
    def setUp(self):
        self.exec = Exec(CONTEXT)
        self.exec.add_var('x', 42)

    def test_clone(self):
        cloned_exec = self.exec.clone()
        self.assertEqual(cloned_exec.globals['x'], 42)

        cloned_exec.add_var('y', 100)
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
        self.exec = Exec(CONTEXT)

    def test_error(self):
        # Test syntax error
        result, error = self.exec._run('print("hello"')
        self.assertEqual(result, '')
        self.assertIn('SyntaxError', error)

        # Test runtime error
        result, error = self.exec._run('print(undefined_variable)')
        self.assertEqual(result, '')
        self.assertIn('NameError', error)

        # Test division by zero error
        result, error = self.exec._run('print(1/0)')
        self.assertEqual(result, '')
        self.assertIn('ZeroDivisionError', error)

if __name__ == '__main__':
    unittest.main()