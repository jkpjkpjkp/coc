import cv2
import torch
from PIL import Image
from typing import Union
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import threading
from coc.config import depth_anything_path
import os.path

class DepthFactory:
    """Depth estimation using the DepthAnythingV2 model.

    Attributes:
        encoder (str): The model variant used for depth estimation.
        device (str): The computation device ('cuda' or 'cpu').
        model (DepthAnythingV2): The depth estimation model.
        semaphore (threading.Semaphore): Controls the number of concurrent depth estimations.
    """
    def __init__(self, encoder: str = 'vitl', device: str = None, max_parallel: int = 1):
        """Initialize.

        Args:
            encoder (str): Depth model encoder type ('vits', 'vitb', 'vitl', 'vitg').
            device (str, optional): Device to use. If None, auto-detects 'cuda' or 'cpu'.
            max_parallel (int): Maximum number of concurrent depth estimations (default: 1).
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.semaphore = threading.Semaphore(max_parallel)

        # Model configuration
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        if encoder not in model_configs:
            raise ValueError(f'Encoder must be one of {list(model_configs.keys())}')

        self.config = model_configs[encoder]

        # Load the model
        self.model = DepthAnythingV2(**self.config)
        self.model.load_state_dict(torch.load(os.path.join(depth_anything_path, f'checkpoints/depth_anything_v2_{self.config['encoder']}.pth') , map_location='cpu'))
        self.model = self.model.to(self.device).eval()

    def _run(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Compute the depth map for an image with concurrency control.

        Args:
            image (Union[str, Image.Image]): Image file path or PIL Image object.

        Returns:
            np.ndarray: HxW depth map as a NumPy array.

        Raises:
            ValueError: If the image input is invalid or cannot be read.
        """
        # Load and convert the image
        if isinstance(image, str):
            raw_img = cv2.imread(image)
            if raw_img is None:
                raise ValueError(f'Could not read image from {image}')
        elif isinstance(image, Image.Image):
            raw_img = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError('Image must be a file path (str) or a PIL Image')

        with self.semaphore:  # Limits concurrent executions to max_parallel
            # Compute depth map
            with torch.no_grad():
                depth_map = self.model.infer_image(raw_img)

        return depth_map

# Module-level variables for lazy initialization
_depth_factory = None
_depth_lock = threading.Lock()

def get_depth(encoder: str = 'vitl', device: str = None, max_parallel: int = 1):
    """Return a thread-safe depth estimation callable.

    Args:
        encoder (str): Depth model encoder type ('vits', 'vitb', 'vitl', 'vitg').
        device (str, optional): Device to use. If None, auto-detects 'cuda' or 'cpu'.
        max_parallel (int): Maximum number of concurrent depth estimations (default: 1).

    Returns:
        callable: A function that computes the depth map for an image.
    """
    global _depth_factory
    if _depth_factory is None:
        with _depth_lock:
            if _depth_factory is None:
                _depth_factory = DepthFactory(encoder=encoder, device=device, max_parallel=max_parallel)

    def process_depth(image: Union[str, Image.Image]) -> np.ndarray:
        """Compute the depth map for the given image with concurrency control. """
        try:
            depth_map = _depth_factory._run(image)
            return depth_map
        except Exception as e:
            raise RuntimeError(f'Depth estimation failed: {str(e)}')

    return process_depth

# Example usage
if __name__ == '__main__':
    # Create a depth estimator with a maximum of 2 concurrent estimations
    depth_estimator = get_depth(encoder='vitl', max_parallel=2)
    # Example calls:
    # depth_map = depth_estimator('path/to/image.jpg')
    # depth_map = depth_estimator(Image.open('path/to/image.jpg'))