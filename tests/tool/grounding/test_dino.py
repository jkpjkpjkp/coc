import pytest
import torch
from coc.tool.grounding.dino import DinoObjectDetectionFactory
from coc.tool.context import create_dummy_image, check_bbox_list

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestDinoObjectDetection:
    def setup_method(self):
        self.factory = DinoObjectDetectionFactory()
        self.test_image = create_dummy_image()
        self.test_texts = ["object"]

    def test_factory_init(self):
        assert self.factory.gd_processor is not None
        assert self.factory.gd_model is not None

    def test_basic_detection(self):
        detections = self.factory.grounding_dino(
            image=self.test_image,
            texts=self.test_texts
        )
        check_bbox_list(detections)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            self.factory.grounding_dino(None, ["object"])
