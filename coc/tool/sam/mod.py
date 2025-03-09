"""segment small objects.

"""

from sam2 import SamAutomaticMaskGenerator, build_sam

import torch
from typing import List, TypedDict
import PIL.Image
from PIL.Image import Image as Img

from langchain.tools import BaseTool


def Sam(BaseTool):
    name: str = 'SAM'
    description: str = (
        'segment small objects. '
        'Args: image: Img, mode: Literal["auto" ,"prompt"], *args **kwargs: additional sam args '
    )
    mask_generator: SamAutomaticMaskGenerator
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam = build_sam(checkpoint='./sam2/sam_vit_h_4b8939.pth')
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=96,
            crop_n_layers=2,
            points_per_batch=256, # for a100
        )
        super().__init__(mask_generator=mask_generator)

    