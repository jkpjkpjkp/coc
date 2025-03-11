import os
import threading
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from typing import Literal, Dict, List
from coc.config import sam_port

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAMWrapper:
    """SAM2 model wrapper with parallel processing control and parameter management"""
    
    _variant_map = {'t': 'tiny', 'l': 'large'}
    
    def __init__(
        self,
        variant: Literal['t', 'l'] = 't',
        device: torch.device = None,
        max_parallelism: int = 4,
        checkpoint_dir: str = "/home/jkp/Pictures/sam2/checkpoints"
    ):
        """
        Initialize SAM2 model wrapper
        
        Args:
            variant: Model variant ('t' for tiny, 'l' for large)
            device: Torch device to use (auto-detected if None)
            max_parallelism: Maximum concurrent inference operations
            checkpoint_dir: Directory containing model checkpoints
        """
        self.device = device or self._detect_device()
        self.variant = variant
        self.max_parallelism = max_parallelism
        self.semaphore = threading.BoundedSemaphore(max_parallelism)
        
        # Model configuration
        self.model = self._load_model(checkpoint_dir)
        self._setup_precision()
        
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
        """Auto-detect available compute device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, checkpoint_dir: str):
        """Load SAM2 model from checkpoint"""
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{self.variant}.yaml"
        checkpoint = f"sam2.1_hiera_{self._variant_map[self.variant]}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        return build_sam2(model_cfg, checkpoint_path, device=self.device, apply_postprocessing=False)

    def _setup_precision(self):
        """Configure mixed precision and tensor core optimizations"""
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def generate_masks(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Dict:
        """Generate raw mask data from image with parallel processing control"""
        with self.semaphore:
            params = {**self.default_params, **kwargs}
            generator = SAM2AutomaticMaskGenerator(self.model, **params)
            mask_data = generator._generate_masks(image)
            return {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in mask_data.items()
            }

    def post_process_masks(
        self,
        mask_data: Dict,
        **kwargs
    ) -> List[Dict]:
        """Post-process mask data into final annotations"""
        with self.semaphore:
            params = {**self.default_params, **kwargs}
            generator = SAM2AutomaticMaskGenerator(self.model, **params)
            return generator._post_process_mask_data(mask_data)

    def create_gradio_interface(self) -> gr.TabbedInterface:
        """Create Gradio interface for interactive use"""
        return gr.TabbedInterface(
            [self._create_generation_interface(), self._create_postprocess_interface()],
            tab_names=["Mask Generation", "Post Processing"],
            title="SAM2 Segmentation Suite"
        )

    def _create_generation_interface(self) -> gr.Interface:
        """Create mask generation interface component"""
        return gr.Interface(
            fn=self._gradio_generate,
            inputs=[
                gr.Image(type="numpy", label="Input Image"),
                *self._create_parameter_controls()
            ],
            outputs=[
                gr.JSON(label="Mask Data"),
                gr.Image(label="Visualization")
            ],
            title="Image Segmentation Generator",
            allow_flagging="never"
        )

    def _create_postprocess_interface(self) -> gr.Interface:
        """Create post-processing interface component"""
        return gr.Interface(
            fn=self._gradio_postprocess,
            inputs=[
                gr.JSON(label="Mask Data"),
                gr.Radio(["binary_mask", "uncompressed_rle", "coco_rle"], 
                        value="binary_mask", label="Output Format")
            ],
            outputs=[
                gr.JSON(label="Processed Annotations"),
                gr.Image(label="Visualization")
            ],
            title="Mask Post-Processor",
            allow_flagging="never"
        )

    def _create_parameter_controls(self) -> List[gr.components.Component]:
        """Create standardized parameter controls for Gradio"""
        return [
            gr.Slider(1, 100, step=1, value=32, label="Points per Side"),
            gr.Slider(1, 256, step=1, value=64, label="Points per Batch"),
            gr.Slider(0.0, 1.0, step=0.01, value=0.8, label="IoU Threshold"),
            gr.Slider(0.0, 1.0, step=0.01, value=0.95, label="Stability Threshold"),
            gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="Stability Offset"),
            gr.Slider(-1.0, 1.0, step=0.01, value=0.0, label="Mask Threshold"),
        ]

    def _gradio_generate(self, image: np.ndarray, *args) -> tuple:
        """Wrapper for Gradio generation interface"""
        params = dict(zip([
            'points_per_side',
            'points_per_batch',
            'pred_iou_thresh',
            'stability_score_thresh',
            'stability_score_offset',
            'mask_threshold'
        ], args))
        
        mask_data = self.generate_masks(image, **params)
        visualization = self._visualize_masks(mask_data)
        return mask_data, visualization

    def _gradio_postprocess(self, mask_data: Dict, output_mode: str) -> tuple:
        """Wrapper for Gradio post-processing interface"""
        annotations = self.post_process_masks(mask_data, output_mode=output_mode)
        visualization = self._visualize_annotations(annotations)
        return annotations, visualization

    def _visualize_masks(self, mask_data: Dict) -> np.ndarray:
        """Create mask visualization from raw mask data"""
        # Implementation from original show_anns function
        # ... (omitted for brevity, use original visualization code)
        return visualization_image

    def _visualize_annotations(self, annotations: List[Dict]) -> np.ndarray:
        """Create annotation visualization from processed data"""
        # Implementation from original show_anns function
        # ... (omitted for brevity, use original visualization code)
        return visualization_image


if __name__ == '__main__':
    # Initialize wrapper with default settings
    sam_engine = SAMWrapper(
        variant='t',
        max_parallelism=2  # Adjust based on available GPU memory
    )
    
    # Create and launch Gradio interface
    interface = sam_engine.create_gradio_interface()
    interface.launch(server_port=sam_port)