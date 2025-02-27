from PIL.Image import Image as Img
import torch
import numpy as np
import random

def image_to_numpy(image: Img) -> np.ndarray:
    return np.array(image.convert("RGB"))

def set_seed():
    from coc.config import SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)