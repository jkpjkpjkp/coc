import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, TypedDict
from PIL import Image, ImageDraw, ImageFont
import threading
from coc.tool.task import Bbox

# Factory class for Grounding DINO object detection
class DinoObjectDetectionFactory:
    def __init__(self, max_parallel=1):
        """Initialize the factory with eager loading of models and concurrency control."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load processor and model eagerly during initialization
        self.gd_processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
        self.semaphore = threading.Semaphore(max_parallel)

    def _run(self, image: Image.Image, texts: List[str], box_threshold=0.2, text_threshold=0.1) -> List[Bbox]:
        """Detect objects in an image using Grounding DINO with concurrency control."""
        if not texts or not image:
            raise ValueError('Valid image and at least one text description required')

        with self.semaphore:  # Controls parallelism
            image = image.convert('RGB')
            text = '. '.join(text.strip().lower() for text in texts) + '.'
            inputs = self.gd_processor(images=image, text=text, return_tensors='pt').to(self.device)
            self.gd_model.to(self.device)
            with torch.no_grad():
                outputs = self.gd_model(**inputs)
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs, inputs['input_ids'], box_threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=[image.size[::-1]]
            )[0]
            return [Bbox(box=box.tolist(), score=score.item(), label=label)
                    for box, score, label in zip(results['boxes'], results['scores'], results['labels'])]

# Module-level variables for lazy initialization of the factory
_dino_factory = None
_dino_lock = threading.Lock()

def draw_boxes(image: Image.Image, detections: List[Bbox]) -> Image.Image:
    """Draw bounding boxes on the image based on detections."""
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {label: (int(255 * (hash(label) % 10) / 10),
                      int(255 * ((hash(label) // 10) % 10) / 10),
                      int(255 * ((hash(label) // 100) % 10) / 10))
              for label in set(det['label'] for det in detections)}
    for det in detections:
        box, label, score = det['box'], det['label'], det['score']
        draw.rectangle(box, outline=colors[label], width=3)
        text = f'{label}: {score:.2f}'
        text_size = draw.textbbox((0, 0), text, font=font)[2:4]
        draw.rectangle([box[0], box[1] - text_size[1], box[0] + text_size[0], box[1]], fill=colors[label])
        draw.text((box[0], box[1] - text_size[1]), text, fill='white', font=font)
    return image

def format_detections(detections: List[Bbox]) -> str:
    """Format the detection results into a readable string."""
    if not detections:
        return 'No objects detected.'
    return f'Found {len(detections)} objects:\n' + '\n'.join(
        f'- {det['label']}: score {det['score']:.2f}, box {[int(b) for b in det['box']]}'
        for det in detections
    )

def get_dino(max_parallel=1):
    """
    Return a Grounding DINO callable with thread-safe lazy initialization and concurrency control.

    Args:
        max_parallel (int): Maximum number of concurrent detections (default: 1).

    Returns:
        callable: A function that processes images for object detection.
    """
    global _dino_factory
    if _dino_factory is None:
        with _dino_lock:
            if _dino_factory is None:
                _dino_factory = DinoObjectDetectionFactory(max_parallel=max_parallel)

    def process_dino(image, object_list_text, box_threshold, text_threshold):
        """Process an image for object detection using Grounding DINO."""
        if not image:
            return None, 'Please upload an image.', []
        objects = [obj.strip() for obj in object_list_text.split(',') if obj.strip()]
        if not objects:
            return image, 'Please specify at least one object.', []
        try:
            detections = _dino_factory._run(image, objects, box_threshold, text_threshold)
            drawn_image = draw_boxes(image.copy(), detections)
            details = format_detections(detections)
            return drawn_image, details, detections
        except Exception as e:
            return image, f'Error: {str(e)}', []

    return process_dino

# Example usage
if __name__ == '__main__':
    detector = get_dino(max_parallel=1)
    # Use detector(image, 'cat, dog', 0.2, 0.1) with an actual image