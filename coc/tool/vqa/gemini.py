import cv2
import numpy as np

from langchain.tools import BaseTool
from typing import List, Literal

from langchain_core.messages import HumanMessage

import os, sys
from coc.config import TEMPERATURE
from langchain_openai import ChatOpenAI
import coc.secret

import base64
from PIL.Image import Image as Img
from PIL import Image
class Gemini(BaseTool):
    name: str = 'VQA'
    description: str = (
        'performs VQA on an image. '
        'Args: image: Img, question: str. '
    )
    mllm: ChatOpenAI

    def __init__(self):
        mllm = ChatOpenAI(
            model='gemini-2.0-pro-exp-02-05',
            openai_api_key=os.environ['GEMINI_API_KEY'],
            openai_api_base=os.environ['GEMINI_BASE_URL'],
            max_tokens=8192,
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


if __name__ == '__main__':
    glm_tool = Gemini()
    response = glm_tool._run(Image.open("data/sample/img3.png"), "Describe this image")
    print(response)