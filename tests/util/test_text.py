from coc.util.text import extract_code
import unittest

class TestExtractCode(unittest.TestCase):
    def test_no_code_blocks(self):
        response = "No code here."
        self.assertEqual(extract_code(response), [])

    def test_single_code_block(self):
        response = "```python\nprint('hello')\n```"
        self.assertEqual(extract_code(response), ["print('hello')"])

    def test_multiple_code_blocks(self):
        response = "```python\nprint(1)\n```\n```python\nprint(2)\n```"
        self.assertEqual(extract_code(response), ["print(1)", "print(2)"])

    def test_code_with_surrounding_text(self):
        response = "Text before```python\ndef foo(): pass\n```Text after"
        self.assertEqual(extract_code(response), ["def foo(): pass"])

    def test_empty_input(self):
        response = ""
        self.assertEqual(extract_code(response), [])

    def test_code_block_with_whitespace(self):
        response = "```python\n   x = 5   \n```"
        self.assertEqual(extract_code(response), ["x = 5"])

    def test_non_python_code_blocks(self):
        response = "```python\nprint('py')\n```\n```javascript\nconsole.log('js')\n```"
        self.assertEqual(extract_code(response), ["print('py')"])

if __name__ == '__main__':
    unittest.main()