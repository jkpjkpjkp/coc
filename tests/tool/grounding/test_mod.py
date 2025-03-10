import pytest
from coc.tool.grounding.mod import ObjectDetectionFactory, Bbox, box_trim
from coc.tool.context import create_dummy_image

class TestObjectDetectionFactory:
    def setup_method(self):
        self.factory = ObjectDetectionFactory()
        self.test_image = create_dummy_image()
        self.test_texts = ["object"]

    def test_trim_result(self):
        mock_detections = [
            Bbox(box=[0, 0, 10, 10], score=0.9, label="object"),
            Bbox(box=[1, 1, 9, 9], score=0.8, label="object"),
            Bbox(box=[50, 50, 60, 60], score=0.7, label="object")
        ]
        
        trimmed = self.factory.trim_result(mock_detections)
        assert len(trimmed) == 2  # Should remove overlapping box

    def test_run_method(self):
        results = self.factory._run(
            image=self.test_image,
            texts=self.test_texts
        )
        assert isinstance(results, list)
        for bbox in results:
            assert 'box' in bbox
            assert 'score' in bbox
            assert 'label' in bbox

class TestBoxTrim:
    def test_occlusion_trimming(self):
        boxes = [
            Bbox(box=[0.0, 0.0, 1.0, 1.0], score=0.9, label='cat'),
            Bbox(box=[0.1, 0.1, 0.9, 0.9], score=0.8, label='cat'),
            Bbox(box=[1.1, 1.1, 2.0, 2.0], score=0.7, label='dog')
        ]
        
        trimmed = box_trim(boxes)
        assert len(trimmed) == 2
        assert trimmed[0]['score'] == 0.9
        assert trimmed[1]['score'] == 0.7
