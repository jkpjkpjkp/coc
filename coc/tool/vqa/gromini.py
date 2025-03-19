import os
import base64
import cv2
import numpy as np
from PIL import Image
from typing import List, Union
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from coc.secret import GEMINI_BROKERS
# Type alias for PIL Image
Img = Image.Image

# Default configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 45.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MODEL_NAME = 'gemini-2.0-pro-exp-02-05'
DEFAULT_BROKER_POOL = GEMINI_BROKERS
DEFAULT_LOG_FILE = 'data/log/big_gemini.log'

# Exceptions to retry on (customize based on API behavior)
RETRYABLE_EXCEPTIONS = (Exception,)  # Broad for now; refine with specific exceptions if known

# Configure logging
logger = logging.getLogger(__name__)

class Gemini:
    """A Visual Question Answering (VQA) tool using the Gemini model family."""
    name: str = 'VQA'
    description: str = 'Performs VQA on an image. Args: image: Img, question: str.'

    # Instance variables with defaults
    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        broker_pool: List[List[str]] = DEFAULT_BROKER_POOL,
        model_name: str = DEFAULT_MODEL_NAME,
        log_file: str = DEFAULT_LOG_FILE,
        **kwargs
    ):
        """Initialize the Gemini VQA tool."""
        # Assign instance variables
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.broker_pool = broker_pool if broker_pool is not None else DEFAULT_BROKER_POOL
        self.model_name = model_name
        self.log_file = log_file
        self.current_broker_index = 0

        # Set up logging
        logging.basicConfig(filename=self.log_file, level=logging.INFO)

        # Validate broker pool
        if not self.broker_pool:
            raise ValueError("Broker pool cannot be empty")

        # Initialize MLLM with the first broker
        self.mllm = self._create_mllm_instance()

        logger.info(
            f"Gemini initialized with retry settings: max_retries={self.max_retries}, "
            f"base_delay={self.base_delay}s, max_delay={self.max_delay}s, "
            f"backoff_factor={self.backoff_factor}"
        )
        logger.info(f"Using broker pool with {len(self.broker_pool)} brokers, starting at index {self.current_broker_index}")

    def _create_mllm_instance(self) -> ChatOpenAI:
        """Create a ChatOpenAI instance using the current broker."""
        api_key, base_url = self.broker_pool[self.current_broker_index]
        return ChatOpenAI(
            model=self.model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_tokens=8192,
            temperature=0.7,  # Replace with TEMPERATURE if defined elsewhere
        )

    def _try_next_broker(self) -> bool:
        """Switch to the next broker in the pool and update the MLLM instance."""
        if len(self.broker_pool) <= 1:
            logger.warning("No alternative brokers available")
            return False
        self.current_broker_index = (self.current_broker_index + 1) % len(self.broker_pool)
        logger.info(f"Switched to broker index {self.current_broker_index}")
        self.mllm = self._create_mllm_instance()
        return True

    def _encode_image(self, image: Img) -> str:
        """Convert a PIL Image to a base64-encoded JPEG string."""
        if not isinstance(image, Img):
            raise TypeError("Image must be a PIL Image object")
        # Ensure RGB format
        image_rgb = image.convert('RGB')
        # Convert to numpy array and encode
        success, buffer = cv2.imencode('.jpg', np.array(image_rgb)[:, :, ::-1])  # RGB to BGR for OpenCV
        if not success:
            raise RuntimeError("Failed to encode image")
        encoded = base64.b64encode(buffer).decode('utf-8')
        logger.info(f"Image encoded (first 10 chars): {encoded[:10]}...")
        return encoded

    def _run(self, image: Img, question: str) -> str:
        """Perform VQA on a single image with broker pool support."""
        logger.info(f"VQA request: {question}")
        for attempt in range(len(self.broker_pool)):
            try:
                return self._run_with_retry(image, question)
            except Exception as e:
                logger.error(f"Broker {self.current_broker_index} failed: {str(e)}")
                if attempt < len(self.broker_pool) - 1:
                    self._try_next_broker()
                else:
                    raise RuntimeError(f"All brokers exhausted. Last error: {str(e)}")

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=DEFAULT_BASE_DELAY, min=0, max=DEFAULT_MAX_DELAY),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} due to {retry_state.outcome.exception()}"
        )
    )
    def _run_with_retry(self, image: Img, question: str) -> str:
        """Internal VQA method with retry logic for a single image."""
        img_base64 = self._encode_image(image)
        # Format as data URL for the model
        img_url = f"data:image/jpeg;base64,{img_base64}"
        message = HumanMessage(
            content=[
                {'type': 'image_url', 'image_url': {'url': img_url}},
                {'type': 'text', 'text': question}
            ]
        )
        response = self.mllm.invoke([message])
        logger.info(f"VQA response: {response.content}")
        return response.content

    def _run_multiimage(self, images: List[Img], question: str) -> str:
        """Perform VQA on multiple images with broker pool support."""
        logger.info(f"Multi-image VQA request: {question}, images: {len(images)}")
        for attempt in range(len(self.broker_pool)):
            try:
                return self._run_multiimage_with_retry(images, question)
            except Exception as e:
                logger.error(f"Broker {self.current_broker_index} failed: {str(e)}")
                if attempt < len(self.broker_pool) - 1:
                    self._try_next_broker()
                else:
                    raise RuntimeError(f"All brokers exhausted. Last error: {str(e)}")

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=DEFAULT_BASE_DELAY, min=0, max=DEFAULT_MAX_DELAY),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} due to {retry_state.outcome.exception()}"
        )
    )
    def _run_multiimage_with_retry(self, images: List[Img], question: str) -> str:
        """Internal VQA method with retry logic for multiple images."""
        content = []
        for img in images:
            img_base64 = self._encode_image(img)
            img_url = f"data:image/jpeg;base64,{img_base64}"
            content.append({'type': 'image_url', 'image_url': {'url': img_url}})
        content.append({'type': 'text', 'text': question})
        message = HumanMessage(content=content)
        response = self.mllm.invoke([message])
        logger.info(f"Multi-image VQA response: {response.content}")
        return response.content

    def run_freestyle(self, prompt: List[Union[str, Img]]) -> str:
        """Handle flexible VQA input with strings and images."""
        logger.info(f"Freestyle VQA request: {prompt}")
        for attempt in range(len(self.broker_pool)):
            try:
                return self._run_freestyle_with_retry(prompt)
            except Exception as e:
                logger.error(f"Broker {self.current_broker_index} failed: {str(e)}")
                if attempt < len(self.broker_pool) - 1:
                    self._try_next_broker()
                else:
                    raise RuntimeError(f"All brokers exhausted. Last error: {str(e)}")

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=DEFAULT_BASE_DELAY, min=0, max=DEFAULT_MAX_DELAY),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} due to {retry_state.outcome.exception()}"
        )
    )
    def _run_freestyle_with_retry(self, prompt: List[Union[str, Img]]) -> str:
        """Internal freestyle VQA method with retry logic."""
        text_parts = []
        images = []
        for item in prompt:
            if isinstance(item, Img):
                images.append(item)
            elif os.path.isfile(item):
                try:
                    with Image.open(item) as img:
                        img.verify()
                        images.append(img.copy())  # Copy to avoid file handle issues
                except Exception:
                    text_parts.append(item)
            else:
                text_parts.append(item)
        
        final_text = "\n".join(text_parts) if text_parts else "Please analyze this image"
        if not images:
            message = HumanMessage(content=[{'type': 'text', 'text': final_text}])
            response = self.mllm.invoke([message])
            return response.content
        elif len(images) == 1:
            return self._run_with_retry(images[0], final_text)
        else:
            return self._run_multiimage_with_retry(images, final_text)

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=DEFAULT_BASE_DELAY, min=0, max=DEFAULT_MAX_DELAY),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} due to {retry_state.outcome.exception()}"
        )
    )
    def invoke(self, question: str) -> str:
        """Invoke the LLM with a text-only query."""
        logger.info(f"Text-only invoke: {question}")
        message = HumanMessage(content=[{'type': 'text', 'text': question}])
        response = self.mllm.invoke([message])
        logger.info(f"Response: {response.content}")
        return response.content