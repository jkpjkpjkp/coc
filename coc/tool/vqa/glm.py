import cv2
import numpy as np

from langchain.tools import BaseTool
from typing import List, Literal

from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatZhipuAI
from coc.config import TEMPERATURE


import base64
from PIL.Image import Image as Img
class GLM(BaseTool):
    name: str = 'VQA'
    description: str = (
        'performs VQA on an image. '
        'Args: image: Img, question: str. '
    )
    mllm: ChatZhipuAI

    def __init__(self, variant: Literal['flash', 'plus'] = 'plus'):
        mllm = ChatZhipuAI(
            model='glm-4v-' + variant,
            temperature=TEMPERATURE,
        )
        super().__init__(mllm=mllm)

    def _run(self, image: Img, question: str) -> str:
        """Perform VQA (Visual Question Answering) using ChatZhipuAI.

        `image` should be an Img with RGB channels.
        `question` is the question string to be asked about the image content.
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Img):
            image = np.array(image.convert('RGB'))
        # Encode the NumPy array as a JPEG image in memory
        success, buffer = cv2.imencode('.jpg', image[:, :, ::-1])  # BGR→RGB if needed
        if not success:
            raise RuntimeError('Failed to encode image to buffer')

        # Convert the buffer to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Create the message using LangChain's message types
        message = HumanMessage(
            content=[
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': img_base64
                    }
                },
                {
                    'type': 'text',
                    'text': question
                }
            ]
        )

        response = self.mllm.invoke([message])
        return response.content

    def _run_with_multiple_images(self, images: List[np.ndarray], question: str) -> List[str]:
        results = []
        for image in images:
            if isinstance(image, Img):
                image = np.array(image.convert('RGB'))
            success, buffer = cv2.imencode('.jpg', image[:, :, ::-1])  # BGR→RGB if needed
            if not success:
                raise RuntimeError('Failed to encode image to buffer')
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            results.append(img_base64)

        message = HumanMessage(
            content=[
                *[{
                    'type': 'image_url',
                    'image_url': {
                        'url': result
                    }
                } for result in results],
                {
                    'type': 'text',
                    'text': question
                }
            ]
        )
        try:
            response = self.mllm.invoke([message])
        except Exception as e:
            print(f'Error invoking MLLM: {str(e)}')
            raise RuntimeError(f'MLLM invocation failed: {str(e)}')

        if not response or not hasattr(response, 'content'):
            print(f'Invalid response from MLLM: {response}')
            raise RuntimeError('MLLM returned invalid response')

        return response.content


if __name__ == '__main__':
    from PIL import Image
    image_path = 'data/sample/onions.jpg'
    image = Image.open(image_path)
    question = 'What is in the image?'
    vqa = GLM()
    print(vqa._run(image, question))