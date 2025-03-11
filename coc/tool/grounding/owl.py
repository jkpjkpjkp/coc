import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List
from PIL import Image
import threading
from .dino import draw_boxes, format_detections
from coc.tool.task import Bbox

class OwlObjectDetectionFactory:
    """OWLv2 object detection service."""
    def __init__(self):
        """Initialize model and processor eagerly."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.owlv2_processor = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')

    def _run(self, image: Image.Image, texts: List[str], threshold=0.1) -> List['Bbox']:
        """Detect objects in an image using OWLv2."""
        if not texts or not image:
            raise ValueError('Valid image and at least one text description required')

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

# Module-level variables for singleton and semaphore
_owl = None
_semaphore = None
_lock = threading.Lock()

def get_owl(max_parallel=1):
    """Return a concurrency-controlled object detection function with shared resources."""
    global _owl, _semaphore

    # Lazy initialization of _owl and _semaphore
    if _owl is None or _semaphore is None:
        with _lock:
            if _owl is None:
                _owl = OwlObjectDetectionFactory()  # Eagerly loads models
            if _semaphore is None:
                _semaphore = threading.Semaphore(max_parallel)

    def process_owl(image, object_list_text, threshold):
        """Process an image for object detection with concurrency control."""
        if image is None:
            return None, 'Please upload an image.', []
        objects = [obj.strip() for obj in object_list_text.split(',') if obj.strip()]
        if not objects:
            return image, 'Please specify at least one object.', []
        try:
            with _semaphore:  # Use the shared semaphore
                detections = _owl._run(image, objects, threshold=threshold)
            # Assuming draw_boxes and format_detections are defined elsewhere
            drawn_image = draw_boxes(image.copy(), detections)
            details = format_detections(detections)
            return drawn_image, details, detections
        except Exception as e:
            return image, f'Error: {str(e)}', []

    return process_owl