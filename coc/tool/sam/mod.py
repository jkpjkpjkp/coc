import os
import threading
import numpy as np
import torch
from PIL import Image
from typing import Literal, Dict, List
import matplotlib.pyplot as plt

# SAM2 imports (assuming these are available in the sam2 package)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# For visualization, assuming utils has show_anns or similar functionality


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

class SAMWrapper:
    """SAM2 model wrapper with parallel processing control and parameter management"""

    _variant_map = {'t': 'tiny', 'l': 'large'}

    def __init__(
        self,
        variant: Literal['t', 'l'] = 't',
        device: torch.device = None,
        max_parallel: int = 1,
        checkpoint_dir: str = "/home/jkp/Pictures/sam2/checkpoints"
    ):
        """
        Initialize SAM2 model wrapper.

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

        # Load model eagerly in __init__ for consistency and thread-safety
        self.model = self._load_model(checkpoint_dir)

        # Default parameters for mask generation
        self.default_params = {
            'points_per_side': 32,
            'points_per_batch': 64,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.95,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'box_nms_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_nms_thresh': 0.7,
            'crop_overlap_ratio': 512/1500,
            'crop_n_points_downscale_factor': 1,
            'min_mask_region_area': 0,
            'use_m2m': False,
            'multimask_output': True,
            'output_mode': "binary_mask"
        }

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

    def generate_masks(self, image: np.ndarray, **kwargs) -> Dict:
        """Generate raw mask data from image."""
        params = {**self.default_params, **kwargs}
        generator = SAM2AutomaticMaskGenerator(self.model, **params)
        # Use autocast for mixed precision during inference
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            mask_data = generator._generate_masks(image)
        return {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in mask_data.items()
        }

    def post_process_masks(self, mask_data: Dict, **kwargs) -> List[Dict]:
        """Post-process mask data into final annotations."""
        params = {**self.default_params, **kwargs}
        generator = SAM2AutomaticMaskGenerator(self.model, **params)
        return generator._post_process_mask_data(mask_data)

    def _run(self, image: np.ndarray, **kwargs) -> List[Dict]:
        """
        Generate and post-process masks for the given image with concurrency control.

        Args:
            image: Input image as a NumPy array
            **kwargs: Additional parameters to override defaults

        Returns:
            List of dictionaries containing annotation data
        """
        with self.semaphore:
            mask_data = self.generate_masks(image, **kwargs)
            annotations = self.post_process_masks(mask_data, **kwargs)
            return annotations

    def visualize_masks(self, image: np.ndarray, mask_data: Dict) -> np.ndarray:
        """
        Create mask visualization from raw mask data.

        Args:
            image: Original image as a NumPy array
            mask_data: Raw mask data from generate_masks

        Returns:
            Visualization image as a NumPy array
        """
        # Convert mask_data to annotation-like format if needed
        # Assuming mask_data contains 'masks' or similar key; adjust based on actual output
        annotations = self.post_process_masks(mask_data)  # Convert to annotations for consistency
        return self.visualize_annotations(image, annotations)

    def visualize_annotations(self, image: np.ndarray, annotations: List[Dict]) -> np.ndarray:
        """
        Create annotation visualization from processed data.

        Args:
            image: Original image as a NumPy array
            annotations: List of annotation dictionaries from post_process_masks

        Returns:
            Visualization image as a NumPy array
        """
        return show_anns(annotations, image.copy())

# Global variables for lazy initialization of SAMWrapper
_sam_engine = None
_sam_lock = threading.Lock()

def get_sam(variant: Literal['t', 'l'] = 't', max_parallel: int = 1, checkpoint_dir: str = "/home/jkp/Pictures/sam2/checkpoints"):
    """
    Return a callable for SAM2 mask generation with lazy loading and concurrency control.

    Args:
        variant: Model variant ('t' for tiny, 'l' for large)
        max_parallel: Maximum concurrent inference operations
        checkpoint_dir: Directory containing model checkpoints

    Returns:
        Callable that processes an image and returns annotations
    """
    global _sam_engine
    if _sam_engine is None:
        with _sam_lock:
            if _sam_engine is None:
                _sam_engine = SAMWrapper(variant=variant, max_parallel=max_parallel, checkpoint_dir=checkpoint_dir)

    def process_sam(image: np.ndarray, **kwargs) -> List[Dict]:
        """
        Process an image to generate masks using SAM2.

        Args:
            image: Input image as a NumPy array
            **kwargs: Additional parameters for mask generation

        Returns:
            List of dictionaries containing annotation data
        """
        return _sam_engine._run(image, **kwargs)

    return process_sam

# Example usage
if __name__ == "__main__":
    sam_processor = get_sam()
    # Example call:
    # image = np.array(Image.open("example.jpg"))
    # annotations = sam_processor(image)
    # vis_image = _sam_engine.visualize_annotations(image, annotations)