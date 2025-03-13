import pytest
import torch
from coc.tool.grounding.dino import get_dino
from tests.tool.test_context import create_dummy_image, check_bbox_list

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestDinoObjectDetection:
    def setup_method(self):
        self.test_image = create_dummy_image()
        self.test_texts = ["object"]

    def test_basic_detection(self):
        detections = get_dino()(
            image=self.test_image,
            objects=self.test_texts
        )
        check_bbox_list(detections[2])

    def test_invalid_input(self):
        ret = get_dino()(None, ["object"])
        assert ret[0] is None
        assert ret[1] == 'Please upload an image.'
        assert ret[2] == []
