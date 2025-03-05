import cv2
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from langchain.tools import BaseTool
from typing import List, Literal, Dict

from langchain_core.messages import HumanMessage

import os, sys

# Configure logging
log_dir = Path("data/log/gemini")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / f"gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
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

    def _encode_image(self, image: Img) -> str:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Img):
            image = np.array(image.convert('RGB'))
        # Log image dimensions
        logger.info(f"Processing image with dimensions: {image.shape}")

        # Encode the NumPy array as a JPEG image in memory
        success, buffer = cv2.imencode('.jpg', image[:, :, ::-1])  # BGR→RGB if needed
        if not success:
            logger.error("Failed to encode image to buffer")
            raise RuntimeError('Failed to encode image to buffer')

        encoded = base64.b64encode(buffer).decode('utf-8')
        # Log first 10 chars of base64 as identifier
        logger.info(f"Image encoded (first 10 chars): {encoded[:10]}...")
        return encoded

    def _run(self, image: Img, question: str) -> str:
        """Perform VQA (Visual Question Answering) using Gemini.

        `image` should be an Img with RGB channels.
        `question` is the question string to be asked about the image content.
        """
        logger.info(f"VQA request received. Question: {question}")
        img_base64 = self._encode_image(image)
        # Convert PIL Image to numpy array if needed
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
        logger.info(f"VQA response: {response.content}")
        return response.content

    def _run_multiimage(self, images: List[Img], question: str) -> str:
        """Perform VQA (Visual Question Answering) using Gemini with multiple images.

        `images` should be a list of Img objects with RGB channels.
        `question` is the question string to be asked about the image content.
        """
        logger.info(f"Multi-image VQA request received. Question: {question}")
        logger.info(f"Number of images: {len(images)}")
        # Prepare content list for the message
        content = []

        # Process each image in the list
        for image in images:
            img_base64 = self._encode_image(image)
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': img_base64
                }
            })

        # Add the question at the end
        content.append({
            'type': 'text',
            'text': question
        })

        # Create the message using LangChain's message types
        message = HumanMessage(content=content)

        response = self.mllm.invoke([message])
        logger.info(f"Multi-image VQA response: {response.content}")
        return response.content

    def run_freestyle(self, prompt: List[str]):
        pass



    def invoke(self, question: str) -> str:
        return self.mllm.invoke(question).content


if __name__ == '__main__':
    glm_tool = Gemini()
    response = glm_tool._run_multiimage([Image.open("data/sample/img1.png"), Image.open("data/sample/img2.png"), Image.open("data/sample/img3.png")],  "Describe this sequence of action")
    print(response)
