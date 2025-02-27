

import unittest

class TestExec(unittest.TestCase):
    def setUp(self):
        self.exec = Exec('coc/exec/context.py')

    def test_exec(self):
        result = self.exec._run('print("hello")')
        self.assertEqual(result, 'hello\n')

from coc.data.muir import LoadMuir
from coc.exec.mod import Exec
class TestTask(unittest.TestCase):
    def setUp(self):
        self.exec = Exec('coc/exec/context.py')
        self.data_loader = LoadMuir('Counting')

    def test_task(self):
        task = next(self.data_loader.convert_to_tasks())
        self.exec.add_var('tsk', task)
        result = self.exec._run('print(tsk["task_type"])')
        self.assertIn('counting', result.lower())

        vqa = self.exec._run('print(vqa(tsk["images"][0], "What is in the picture?"))')
        self.assertTrue('mitten' in vqa or 'glove' in vqa)

class TestGPTOutput(unittest.TestCase):
    def setUp(self):
        self.exec = Exec('coc/exec/context.py')
        self.data_loader = LoadMuir('Ordering')

    def test_gpt_output(self):
        task = next(self.data_loader.convert_to_tasks())
        self.exec.add_var('tsk', task)
        code_str = '''
  from PIL import Image
  import numpy as np

  #   def vqa(image: np.ndarray, question: str) -> str:
  #       ...

  image1 = tsk["images"][0]
  image2 = tsk["images"][1]
  dense_caption1 = vqa(image1, "Generate dense caption for the image")
  dense_caption2 = vqa(image2, "Generate dense caption for the image")

  dense_caption1, dense_caption2

  print(dense_caption1, dense_caption2)
  '''
        result = self.exec._run(code_str)
        self.assertIn('simpson', result.lower())

if __name__ == '__main__':
    unittest.main()