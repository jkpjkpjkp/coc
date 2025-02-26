from typing import Dict, List, Any, Optional, Union
import numpy as np
from PIL import Image
import logging
from torch.cuda import empty_cache

# typed alias
Img = Union[np.ndarray, Image.Image]

import sys
sys.path.append("/home/jkp/hack/diane/aoa")

# The only place you import your "singleton factory"
from tool_factory import (
    get_sam2,
    get_sam2_ultralytics,
    get_depth_estimator,
    get_grounding,
    get_vqa
)

def sam(image: Img) -> List[Dict[str, Any]]:
    ret = get_sam2_ultralytics()(image)
    empty_cache()
    return ret

def show_anns(mask: List[Dict[str, Any]]) -> np.ndarray:
    return get_sam2()._show_anns(mask)

def depth_estimator(image: Img) -> np.ndarray:
    return get_depth_estimator()(image)

def grounding(image: Img, objects_of_interest: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if isinstance(image, Image.Image):
        image = image_to_numpy(image)
    return get_grounding()(image, objects_of_interest)

def vqa(image: Img, question: str) -> str:
    return get_vqa()._run(image, question)

def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))

