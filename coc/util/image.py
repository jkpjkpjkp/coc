from PIL.Image import Image as Img
import numpy as np

def image_to_numpy(image: Img) -> np.ndarray:
    return np.array(image.convert("RGB"))
