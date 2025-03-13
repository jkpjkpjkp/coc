import os
import threading
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as Img
from typing import Literal, Dict, List
import matplotlib.pyplot as plt
from coc.config import sam_variant

# SAM2 imports (assuming these are available in the sam2 package)
from sam2.build_sam import build_sam2
# Hypothetical SAM2ImagePredictor import for prompted segmentation
from sam2.sam2_image_predictor import SAM2ImagePredictor

def show_predicted_masks(image: np.ndarray, annotations: List[Dict]) -> np.ndarray:
    """Visualize predicted masks on the image.

    Args:
        image: Original image as a NumPy array
        annotations: List of dictionaries with 'mask' (np.ndarray) and 'score' (float)

    Returns:
        Visualization image as a NumPy array
    """
    if len(annotations) == 0:
        return image.copy()

    # Sort by score and take the best prediction
    sorted_anns = sorted(annotations, key=lambda x: x['score'], reverse=True)
    vis_image = image.copy()

    # Overlay the highest-scoring mask in red with transparency
    best_mask = sorted_anns[0]['mask']
    color_mask = np.array([255, 0, 0], dtype=np.uint8)  # Red color
    vis_image[best_mask] = (vis_image[best_mask] * 0.5 + color_mask * 0.5).astype(np.uint8)

    return vis_image

class SAM2PredictorWrapper:
    """SAM2 model wrapper for image prediction with prompts, with parallel processing control"""

    _variant_map = {'t': 'tiny', 'l': 'large'}

    def __init__(
        self,
        variant: Literal['t', 'l'] = 't',
        device: torch.device = None,
        max_parallel: int = 1,
        checkpoint_dir: str = "/home/jkp/Pictures/sam2/checkpoints"
    ):
        """
        Initialize SAM2 predictor wrapper.

        Args:
            variant: Model variant ('t' for tiny, 'l' for large)
            device: Torch device to use (auto-detected if None)
            max_parallel: Maximum concurrent inference operations
            checkpoint_dir: Directory containing model checkpoints
        """
        self.device = device or self._detect_device()
        self.variant = variant
        self.semaphore = threading.BoundedSemaphore(max_parallel)
        self.checkpoint_dir = checkpoint_dir
        self.model = self._load_model(checkpoint_dir)

    def _detect_device(self) -> torch.device:
        """Auto-detect available compute device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, checkpoint_dir: str):
        """Load SAM2 model from checkpoint."""
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{self.variant}.yaml"
        checkpoint = f"sam2.1_hiera_{self._variant_map[self.variant]}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        model = build_sam2(model_cfg, checkpoint_path, device=self.device, apply_postprocessing=False)
        return model

    def _run(self, image: np.ndarray, prompts: Dict) -> List[Dict]:
        """
        Generate predicted masks for the given image and prompts with concurrency control.

        Args:
            image: Input image as a NumPy array (H, W, 3)
            prompts: Dictionary containing prompt data, e.g.,
                     {'point_coords': np.ndarray, 'point_labels': np.ndarray}

        Returns:
            List of dictionaries containing mask data, each with 'mask' (np.ndarray) and 'score' (float)
        """
        with self.semaphore:
            # Create a new predictor instance for each inference
            predictor = SAM2ImagePredictor(self.model)
            predictor.set_image(image)  # Preprocess the image

            # Perform prediction with prompts (assuming SAM2ImagePredictor.predict API)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                masks, scores, _ = predictor.predict(**prompts)  # Ignoring logits for simplicity

            # Convert torch tensors to numpy and package results
            annotations = [
                {
                    'mask': mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask,
                    'score': float(score.cpu() if isinstance(score, torch.Tensor) else score)
                }
                for mask, score in zip(masks, scores)
            ]
            return annotations

    def visualize_predictions(self, image: np.ndarray, annotations: List[Dict]) -> np.ndarray:
        """
        Create visualization of predicted masks on the image.

        Args:
            image: Original image as a NumPy array
            annotations: List of predicted mask dictionaries from _run

        Returns:
            Visualization image as a NumPy array
        """
        return show_predicted_masks(image, annotations)

# Global variables for lazy initialization of SAM2ImagePredictor
_sam_predictor_engine = None
_sam_predictor_lock = threading.Lock()

def get_sam_predictor(
    variant: Literal['t', 'l'] = sam_variant,
    max_parallel: int = 1,
    checkpoint_dir: str = "/home/jkp/Pictures/sam2/checkpoints"
):
    """
    Return a callable for SAM2 image prediction with prompts, with lazy loading and concurrency control.

    Args:
        variant: Model variant ('t' for tiny, 'l' for large)
        max_parallel: Maximum concurrent inference operations
        checkpoint_dir: Directory containing model checkpoints

    Returns:
        Callable that processes an image and prompts, returning predicted masks
    """
    global _sam_predictor_engine
    if _sam_predictor_engine is None:
        with _sam_predictor_lock:
            if _sam_predictor_engine is None:
                _sam_predictor_engine = SAM2PredictorWrapper(
                    variant=variant,
                    max_parallel=max_parallel,
                    checkpoint_dir=checkpoint_dir
                )

    def process_sam_predictor(image: Img, prompts: Dict) -> List[Dict]:
        """
        Process an image with prompts to generate predicted masks using SAM2.

        Args:
            image: Input image (PIL Image or NumPy array)
            prompts: Dictionary of prompts (e.g., {'point_coords': np.ndarray, 'point_labels': np.ndarray})

        Returns:
            List of dictionaries containing mask data
        """
        if isinstance(image, Img):
            image = np.array(image)
        return _sam_predictor_engine._run(image, prompts)

    return process_sam_predictor

# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    sam_predictor = get_sam_predictor(variant='t')

    # Load an example image
    image = np.array(Image.open("example.jpg"))

    # Define example point prompts: one point at (100, 100) labeled as foreground (1)
    prompts = {
        'point_coords': np.array([[100, 100]]),  # Shape: (N, 2) for N points
        'point_labels': np.array([1])            # Shape: (N,) with 1 for foreground, 0 for background
    }

    # Generate predicted masks
    annotations = sam_predictor(image, prompts)

    # Visualize the result
    vis_image = _sam_predictor_engine.visualize_predictions(image, annotations)
    plt.imshow(vis_image)
    plt.axis('off')
    plt.show()