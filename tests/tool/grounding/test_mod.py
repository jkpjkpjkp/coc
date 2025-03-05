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
            Bbox([1.1, 1.1, 2.0, 2.0], 0.8, 'dog'),
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
            Bbox([0.0, 0.0, 10.0, 10.0], 0.9, 'cat'),
            Bbox(box=[8.0, 8.0, 12.0, 12.0], score=0.8, label='cat'),
        ]
        result = box_trim(boxes)
        self.assertEqual(len(result), 2)

    def test_zero_area_accepted_box(self):
        # Accepted box has zero area (invalid), should be skipped
        boxes = [
            Bbox(box=[0.0, 0.0, 0.0, 0.0], score=0.9, label='cat'),
            Bbox([0.0, 0.0, 1.0, 1.0], 0.8, 'cat'),
        ]
        result = box_trim(boxes)
        self.assertEqual(len(result), 2)


class TestObjectDetectionFactory(unittest.TestCase):
    @patch.object(ObjectDetectionFactory, 'grounding_dino')
    @patch.object(ObjectDetectionFactory, 'owl2')
    def test_run_combines_results(self, mock_owl, mock_gdino):
        mock_owl.return_value = [Bbox([], 0.8, 'cat')]
        mock_gdino.return_value = [
            Bbox([], 0.7, 'cat'),
            Bbox([], 0.6, 'dog'),
        ]
        factory = ObjectDetectionFactory()
        image = Image.new('RGB', (100, 100))
        result = factory._run(image, ['cat', 'dog'])
        # Only 'cat' is present in both results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'cat')

    @patch('transformers.AutoProcessor.from_pretrained')
    @patch('transformers.AutoModelForZeroShotObjectDetection.from_pretrained')
    def test_grounding_dino_process(self, mock_model, mock_processor):
        mock_processor.return_value = MagicMock(
            return_value={
                'input_ids': torch.tensor([0]),
                'pixel_values': torch.tensor([0])
            },
            post_process_grounded_object_detection=MagicMock(
                return_value=[
                    {
                        'boxes': torch.tensor([[0, 0, 10, 10]]),
                        'scores': torch.tensor([0.9]),
                        'text_labels': ['cat']
                    }
                ]
            )
        )
        mock_model.return_value = MagicMock(return_value=MagicMock())
        image = Image.new('RGB', (100, 100))
        texts = ['cat']
        factory = ObjectDetectionFactory()
        factory.gd_model = mock_model.return_value
        result = factory.grounding_dino(image, texts)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'cat')

    @patch('transformers.Owlv2Processor.from_pretrained')
    @patch('transformers.Owlv2ForObjectDetection.from_pretrained')
    def test_owlv2_process(self, mock_model, mock_processor):
        mock_processor.return_value = MagicMock(
            post_process_grounded_object_detection=MagicMock(
                return_value=[
                    {
                        'scores': torch.tensor([0.9]),
                        'labels': torch.tensor([0]),
                        'boxes': torch.tensor([[0, 0, 10, 10]])
                    }
                ]
            )
        )
        mock_model.return_value = MagicMock(return_value=MagicMock())
        image = Image.new('RGB', (100, 100))
        texts = ['cat']
        factory = ObjectDetectionFactory()
        factory.owlv2_model = mock_model.return_value
        result = factory.owl2(image, texts)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'cat')

    def test_run_with_empty_detections(self):
        factory = ObjectDetectionFactory()
        factory.owl2 = MagicMock(return_value=[])
        factory.grounding_dino = MagicMock(return_value=[Bbox([], 0.5, 'cat')])
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
