import os
import threading
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as Img
from typing import Literal, Dict, List
import matplotlib.pyplot as plt
from coc.config import sam_path, sam_variant

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SamPredictor:
    def __init__(self):
        self._model = None
        self._model_lock = threading.Lock()
        self._sam_variant = sam_variant
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    sam2_checkpoint = f"{sam_path}/checkpoints/sam2.1_hiera_{'large' if self._sam_variant == 'l' else 'tiny'}.pt"
                    model_cfg = f"configs/sam2.1/sam2.1_hiera_{self._sam_variant}.yaml"

                    self._model = build_sam2(model_cfg, sam2_checkpoint, device=self._device)

    def _run(self, image, **kwargs):
        self._load_model()

        init_params = {}
        for param in ['mask_threshold', 'max_hole_area', 'max_sprinkle_area']:
            if param in kwargs:
                init_params[param] = kwargs.pop(param)

        predictor = SAM2ImagePredictor(sam_model=self._model, **init_params)
        predictor.set_image(np.array(image))
        ret = predictor.predict(**kwargs)
        return ret


    def _run_auto(self, image, **kwargs):
        self._load_model()
        generator = SAM2AutomaticMaskGenerator(model=self._model, **kwargs)
        ret = generator.generate(np.array(image))
        return ret

_prdct = SamPredictor()
_auto_sema = threading.Semaphore(1)
_pred_sema = threading.Semaphore(4)

def get_sam_auto():
    def process_sam_auto(image, **kwargs):
        with _auto_sema:
            ret = _prdct._run_auto(image, **kwargs)
        return ret
    return process_sam_auto

def get_sam_predict():
    def process_sam(image, **kwargs):
        with _pred_sema:
            ret = _prdct._run(image, **kwargs)
        return ret
    return process_sam