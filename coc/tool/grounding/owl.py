import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, TypedDict
from PIL import Image, ImageDraw, ImageFont
from .dino import Bbox, draw_boxes, format_detections
import threading

class OwlObjectDetectionFactory:
    """OWLv2 object detection service.

    Attributes:
        device: Computation device (cuda or cpu)
        owlv2_processor: OWLv2 processor
        owlv2_model: OWLv2 model
    """
    def __init__(self, max_parallel=1):
        """Initialize model and move to appropriate device."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._owlv2_processor = None
        self._owlv2_model = None
        self.semaphore = threading.Semaphore(max_parallel)

    @property
    def owlv2_processor(self):
        if self._owlv2_processor is None:
            self._owlv2_processor = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
        return self._owlv2_processor

    @property
    def owlv2_model(self):
        if self._owlv2_model is None:
            self._owlv2_model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')
        return self._owlv2_model

    def _run(self, image: Image.Image, texts: List[str], threshold=0.1) -> List[Bbox]:
        """Detect objects in image using OWLv2.

        Args:
            image: Input PIL image
            texts: List of text descriptions to detect
            threshold: Detection confidence threshold

        Returns:
            List of detected bounding boxes with scores and labels

        Raises:
            ValueError: If invalid input is provided
        """
        if not texts:
            raise ValueError('At least one text description required')
        if not image:
            raise ValueError('Valid image required')

        image = image.convert('RGB')
        inputs = self.owlv2_processor(text=texts, images=image, return_tensors='pt').to(self.device)
        self.owlv2_model.to(self.device)
        with torch.no_grad():
            outputs = self.owlv2_model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        processed_results = self.owlv2_processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )
        result = processed_results[0]
        detections = [
            Bbox(box=box.tolist(), score=score.item(), label=texts[label_idx.item()])
            for score, label_idx, box in zip(result['scores'], result['labels'], result['boxes'])
        ]
        return detections

# Initialize the factory
_owl = OwlObjectDetectionFactory()

def process_owl(image, object_list_text, threshold):
    if image is None:
        return None, "Please upload an image.", []
    objects = [obj.strip() for obj in object_list_text.split(",") if obj.strip()]
    if not objects:
        return image, "Please specify at least one object.", []
    try:
        detections = _owl._run(image, objects, threshold=threshold)
        drawn_image = draw_boxes(image.copy(), detections)
        details = format_detections(detections)
        return drawn_image, details, detections
    except Exception as e:
        return image, f"Error: {str(e)}", []

def get_owl():
    return process_owl