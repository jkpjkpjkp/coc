"""qwen 2.5 vl 72B.

best open-weights model, outperforms 4o and gemini2 on various benchmarks.
called through dashscope api.
"""
import os
from openai import OpenAI
import coc.secret
import base64
from PIL import Image
from io import BytesIO

class Qwen25VL:
    """not tested yet.
    """
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )
        self.model_name = ('qwen-vl-max-latest', 'qwen-vl-plus-latest')[1]

    def pil_to_base64(self, image):
        # Create an in-memory byte stream
        buffered = BytesIO()
        # Save the image to the stream in a specific format (e.g., PNG or JPEG)
        image.save(buffered, format="PNG")
        # Get the byte data
        img_bytes = buffered.getvalue()
        # Encode to base64 and decode to a UTF-8 string
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        # Add the data URI prefix (required for some APIs)
        return f"data:image/png;base64,{base64_string}"

    def _run(self, image, text):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self.pil_to_base64(image)}  # Use the base64 string here
                        },
                        {"type": "text", "text": text}
                    ]
                },
            ]
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


if __name__ == '__main__':
    from PIL import Image
    qwen = Qwen25VL()
    image = Image.open('data/sample/img3.png')
    # convert to rgb
    image = image.convert('RGB')
    ret = qwen._run(image, 'what is the yellow thing in the middle?')
    print(ret)