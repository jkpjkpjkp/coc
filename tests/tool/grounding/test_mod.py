import unittest
from unittest.mock import MagicMock, patch, Mock
from PIL import Image
import torch

# Assuming the code is in a module named 'object_detection'
from coc.tool.grounding.mod import Bbox, box_trim, ObjectDetectionFactory


class TestBbox(unittest.TestCase):
    def test_bbox_creation(self):
        box = [0.1, 0.2, 0.5, 0.6]
        score = 0.8
        label = "cat"
        bbox = Bbox(box=box, score=score, label=label)
        self.assertEqual(bbox['box'], box)
        self.assertEqual(bbox['score'], score)
        self.assertEqual(bbox['label'], label)


class TestBoxTrim(unittest.TestCase):
    def test_no_overlap(self):
        boxes = [
            Bbox(box=[0.0, 0.0, 1.0, 1.0], score=0.9, label='cat'),
            Bbox(box=[1.1, 1.1, 2.0, 2.0], score=0.8, label='dog'),
        ]
        result = box_trim(boxes)
        self.assertEqual(len(result), 2)

    def test_high_overlap_reject(self):
        boxes = [
            Bbox(box=[0.0, 0.0, 10.0, 10.0], score=0.9, label='cat'),
            Bbox(box=[0.0, 0.0, 9.0, 9.0], score=0.8, label='cat'),
        ]
        result = box_trim(boxes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['score'], 0.9)

    def test_low_overlap_accept(self):
        boxes = [
            Bbox(box=[0.0, 0.0, 10.0, 10.0], score=0.9, label='cat'),
            Bbox(box=[8.0, 8.0, 12.0, 12.0], score=0.8, label='cat'),
        ]
        result = box_trim(boxes)
        self.assertEqual(len(result), 2)

    def test_zero_area_accepted_box(self):
        # Accepted box has zero area (invalid), should be skipped
        boxes = [
            Bbox(box=[0.0, 0.0, 0.0, 0.0], score=0.9, label='cat'),
            Bbox(box=[0.0, 0.0, 1.0, 1.0], score=0.8, label='cat'),
        ]
        result = box_trim(boxes)
        self.assertEqual(len(result), 2)


class TestObjectDetectionFactory(unittest.TestCase):
    def test_run_with_empty_detections(self):
        factory = ObjectDetectionFactory()
        image = Image.new('RGB', (100, 100))
        result = factory._run(image, ['cat'])
        self.assertEqual(len(result), 0)

class TestGroundingNoMock(unittest.TestCase):
    def setUp(self):
        self.obj = ObjectDetectionFactory()


    def test_simple_counting(self):
        image_path = 'data/sample/4girls.jpg'
        image = Image.open(image_path)

        texts = ['girl']

        result = self.obj._run(image, texts)
        self.assertEqual(len(result), 4)

    def test_medium_counting(self):
        image_path = 'data/sample/4girls.jpg'
        image = Image.open(image_path)

        texts = ['hand']

        result = self.obj._run(image, texts)
        self.assertEqual(len(result), 6)  # one is at far top-left corner

    def test_multiple_object(self):

        image_path = 'data/sample/4girls.jpg'
        image = Image.open(image_path)

        texts = ['boy', 'girl', 'hand']

        result = self.obj._run(image, texts)

        boys = [x for x in result if 'boy' in x['label']]
        girls = [x for x in result if 'girl' in x['label']]
        hands = [x for x in result if 'hand' in x['label']]

        self.assertEqual(len(boys), 0)
        self.assertEqual(len(girls), 4)
        self.assertEqual(len(hands), 6)


if __name__ == '__main__':
    unittest.main()
