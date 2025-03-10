import pytest
import torch
from coc.tool.grounding.owl import OwlObjectDetectionFactory
from coc.tool.context import create_dummy_image, check_bbox_list

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestOwlObjectDetection:
    def setup_method(self):
        self.factory = OwlObjectDetectionFactory()
        self.test_image = create_dummy_image()
        self.test_texts = ["object"]

    def test_factory_init(self):
        assert self.factory.owlv2_processor is not None
        assert self.factory.owlv2_model is not None

    def test_basic_detection(self):
        detections = self.factory.owl2(
            image=self.test_image,
            texts=self.test_texts,
            threshold=0.1
        )
        check_bbox_list(detections)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            self.factory.owl2(None, ["object"])
