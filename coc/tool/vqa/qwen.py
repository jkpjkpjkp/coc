"""qwen 2.5 vl 72B.

best open-weights model, outperforms 4o and gemini2 on various benchmarks.
called through dashscope api.
"""
import os
from openai import OpenAI

os.environ['DASHSCOPE_API_KEY'] = 'sk-3a88a7f6a9264d54b62eeb4181192248'

class Qwen25VL:
    """not tested yet.
    """
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )
        self.model_name = ('qwen-vl-max-latest', 'qwen-vl-plus-latest')[1]

    def _run(self, image, text):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': 'You are a helpful assistant.'}
                    ],
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {'url': image},
                        },
                        {'type': 'text', 'text': text},
                    ],
                },
            ],
        )
        return completion.choices[0].message.content

def demo():
    """Demo of qwen 2.5 vl 72B."""
    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )

    img_url = (
        'https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/'
        '20241022/emyrja/dog_and_girl.jpeg'
    )

    completion = client.chat.completions.create(
        model=('qwen-vl-max-latest', 'qwen-vl-plus-latest')[1],
        messages=[
            {
                'role': 'system',
                'content': [
                    {'type': 'text', 'text': 'You are a helpful assistant.'}
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': img_url
                        },
                    },
                    {'type': 'text', 'text': '图中描绘的是什么景象?'},
                ],
            },
        ],
    )

    print(completion.choices[0].message.content)
