import pytest
import torch
from coc.tool.grounding.dino import get_dino
from coc.tool.context import create_dummy_image, check_bbox_list

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestDinoObjectDetection:
    def setup_method(self):
        self.test_image = create_dummy_image()
        self.test_texts = ["object"]

    def test_basic_detection(self):
        detections = get_dino()(
            image=self.test_image,
            texts=self.test_texts
        )
        check_bbox_list(detections)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            get_dino()(None, ["object"])
