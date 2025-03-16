import unittest
from unittest.mock import MagicMock
from PIL import Image
from coc.tool.vqa.glm import GLM

class TestVQA(unittest.TestCase):
    def setUp(self):
        self.image_path = 'data/sample/onions.jpg'
        self.image = Image.open(self.image_path)
        self.question = 'What is in the image?'
        self.vqa = GLM()

    def test_vqa_run(self):
        result = self.vqa._run(self.image, self.question)
        self.vqa._run(self.image, self.question)
        self.assertIn('onion', result.lower())

if __name__ == '__main__':
    unittest.main()