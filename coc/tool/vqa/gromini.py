import os
import base64
import cv2
import numpy as np
from PIL import Image
from typing import List, Union
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from coc.util.logging import get_logger
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

# Get logger instance
logger = get_logger(__name__)

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
        # Store configuration
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.model_name = model_name
        self.log_file = log_file
        self.broker_pool = broker_pool
        self.current_broker_index = 0
        self.kwargs = kwargs
        
        # Set up specialized logger with the specific log file
        self.logger = get_logger(__name__, os.path.basename(self.log_file))
        
        # Create the mllm instance with the initial broker
        self.mllm = self._create_mllm_instance()
        
        # Log the broker pool information
        self.logger.info(
            f"Gemini initialized with {len(broker_pool)} brokers, "
            f"max_retries={max_retries}, base_delay={base_delay}, "
            f"max_delay={max_delay}, backoff_factor={backoff_factor}"
        )
        self.logger.info(f"Using broker pool with {len(self.broker_pool)} brokers, starting at index {self.current_broker_index}")

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
        """Try to switch to the next available broker."""
        # If there's only one broker, there's no point trying another
        if len(self.broker_pool) <= 1:
            self.logger.warning("No alternative brokers available")
            return False
            
        # Move to the next broker in the pool
        self.current_broker_index = (self.current_broker_index + 1) % len(self.broker_pool)
        self.mllm = self._create_mllm_instance()
        self.logger.info(f"Switched to broker index {self.current_broker_index}")
        return True
        
    def _encode_image(self, image: Img) -> str:
        """Encode an image to base64 for API request."""
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            
        # Encode the NumPy array as a JPEG image in memory
        success, buffer = cv2.imencode('.jpg', image[:, :, ::-1])  # BGR→RGB conversion
        if not success:
            raise RuntimeError('Failed to encode image to buffer')
            
        encoded = base64.b64encode(buffer).decode('utf-8')
        # Log first 10 chars of base64 as identifier
        self.logger.info(f"Image encoded (first 10 chars): {encoded[:10]}...")
        return encoded
        
    def _run(self, image: Img, question: str) -> str:
        """Perform VQA using Gemini with a single image."""
        self.logger.info(f"VQA request: {question}")
        try:
            return self._run_with_retry(image, question)
        except Exception as e:
            self.logger.error(f"Broker {self.current_broker_index} failed: {str(e)}")
            if self._try_next_broker():
                return self._run(image, question)  # Retry with new broker
            raise  # Re-raise if no more brokers to try

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=DEFAULT_BASE_DELAY, min=0, max=DEFAULT_MAX_DELAY),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state, self=None: self.logger.warning(
            f"Retry {retry_state.attempt_number} due to {retry_state.outcome.exception()}"
        ) if self else None
    )
    def _run_with_retry(self, image: Img, question: str) -> str:
        """Run the VQA query with retry logic."""
        img_base64 = self._encode_image(image)
        
        # Create the message with image and text
        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": question}
            ]
        )
        
        response = self.mllm.invoke([message])
        self.logger.info(f"VQA response: {response.content}")
        return response.content
    
    def _run_multiimage(self, images: List[Img], question: str) -> str:
        """Perform VQA using Gemini with multiple images."""
        self.logger.info(f"Multi-image VQA request: {question}, images: {len(images)}")
        try:
            return self._run_multiimage_with_retry(images, question)
        except Exception as e:
            self.logger.error(f"Broker {self.current_broker_index} failed: {str(e)}")
            if self._try_next_broker():
                return self._run_multiimage(images, question)  # Retry with new broker
            raise  # Re-raise if no more brokers to try
            
    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=DEFAULT_BASE_DELAY, min=0, max=DEFAULT_MAX_DELAY),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state, self=None: self.logger.warning(
            f"Retry {retry_state.attempt_number} due to {retry_state.outcome.exception()}"
        ) if self else None
    )
    def _run_multiimage_with_retry(self, images: List[Img], question: str) -> str:
        """Run multi-image VQA with retry logic."""
        # Prepare content list for the message
        content = []
        
        # Process each image in the list
        for image in images:
            img_base64 = self._encode_image(image)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})
            
        # Add the question at the end
        content.append({"type": "text", "text": question})
        
        # Create the message and invoke
        message = HumanMessage(content=content)
        response = self.mllm.invoke([message])
        
        self.logger.info(f"Multi-image VQA response: {response.content}")
        return response.content
        
    def run_freestyle(self, prompt: List[Union[str, Img]]) -> str:
        """Run a freestyle prompt with a mix of text and images."""
        self.logger.info(f"Freestyle VQA request: {prompt}")
        try:
            return self._run_freestyle_with_retry(prompt)
        except Exception as e:
            self.logger.error(f"Broker {self.current_broker_index} failed: {str(e)}")
            if self._try_next_broker():
                return self.run_freestyle(prompt)  # Retry with new broker
            raise  # Re-raise if no more brokers to try
            
    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=DEFAULT_BASE_DELAY, min=0, max=DEFAULT_MAX_DELAY),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state, self=None: self.logger.warning(
            f"Retry {retry_state.attempt_number} due to {retry_state.outcome.exception()}"
        ) if self else None
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
        before_sleep=lambda retry_state, self=None: self.logger.warning(
            f"Retry {retry_state.attempt_number} due to {retry_state.outcome.exception()}"
        ) if self else None
    )
    def invoke(self, question: str) -> str:
        """Invoke the LLM with a text-only query."""
        self.logger.info(f"Text-only invoke: {question}")
        message = HumanMessage(content=[{'type': 'text', 'text': question}])
        response = self.mllm.invoke([message])
        self.logger.info(f"Response: {response.content}")
        return response.content