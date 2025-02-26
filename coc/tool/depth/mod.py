import sys
sys.path.append('/home/jkp/Pictures/Depth-Anything-V2')
from langchain.tools import BaseTool
import torch
import numpy as np
from PIL import Image
import io
import base64
import cv2
from depth_anything_v2.dpt import DepthAnythingV2
class DepthEstimator(BaseTool):
    """depth estimator.

    using the Depth Anything V2 model.
    """

    name: str = 'depth_estimator'
    description: str = (
        'Estimates a normalized depth map from an input image. '
        'Input should be a base64-encoded string representing a PIL Image. '
        'Returns a base64-encoded string containing the depth map where '
        'brighter pixels indicate objects closer to the camera.'
    )
    device: str
    model: DepthAnythingV2

    def __init__(
        self,
        checkpoint_path: str = '/home/jkp/Pictures/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth',
    ):
        model = DepthAnythingV2(
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024]
        )

        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to('cuda').eval()
        except Exception as e:
            raise ValueError(f'Failed to load model: {e}')

        super().__init__(device='cuda', model=model)

    def _run(self, image: np.ndarray) -> np.ndarray:
        try:
            # Ensure we have a numpy array
            img_np = np.asarray(image)
            if img_np.size == 0:
                raise ValueError('Invalid or empty input image.')

            # Convert from RGB (PIL) to BGR (OpenCV)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with torch.inference_mode():
                depth_map = self.model.infer_image(img_bgr)  # H x W float32

            if not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
                raise ValueError('No valid depth output was produced.')

            # Normalize the depth to [0,255]
            d_min, d_max = depth_map.min(), depth_map.max()
            depth_range = max(d_max - d_min, 1e-8)
            depth_vis = (depth_map - d_min) / depth_range * 255.0
            depth_vis = depth_vis.astype(np.uint8)

            return depth_vis

        except Exception as e:
            raise e

if __name__ == '__main__':
    image_path = '/home/jkp/hack/diane/data/zerobench_images/zerobench/example_21_image_0.png'
    image = Image.open(image_path)
    depth_estimator = DepthEstimator()
    depth_map = depth_estimator._run(image)
    depth_map = Image.fromarray(depth_map)
    depth_map.save('depth_of_example_21.png')
    depth_map = np.array(depth_map)
    depth_map2 = np.where(depth_map > 63, 255, depth_map * 4)
    depth_map = np.where(depth_map > 127, 255, depth_map * 2)
    depth_map = Image.fromarray(depth_map)
    depth_map2 = Image.fromarray(depth_map2)
    depth_map.save('depth_of_example_21_enhanced.png')
    depth_map2.save('depth_of_example_21_enhanced_4x.png')
