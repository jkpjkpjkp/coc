import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, TypedDict
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from coc.config import owl_port
from .dino import Bbox, draw_boxes, format_detections

class OwlObjectDetectionFactory:
    """OWLv2 object detection service.

    Attributes:
        device: Computation device (cuda or cpu)
        owlv2_processor: OWLv2 processor
        owlv2_model: OWLv2 model
    """
    def __init__(self):
        """Initialize model and move to appropriate device."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._owlv2_processor = None
        self._owlv2_model = None

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

    def owl2(self, image: Image.Image, texts: List[str], threshold=0.1) -> List[Bbox]:
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
obj = OwlObjectDetectionFactory()

def process_owl(image, object_list_text, threshold):
    if image is None:
        return None, "Please upload an image.", []
    objects = [obj.strip() for obj in object_list_text.split(",") if obj.strip()]
    if not objects:
        return image, "Please specify at least one object.", []
    try:
        detections = obj.owl2(image, objects, threshold=threshold)
        drawn_image = draw_boxes(image.copy(), detections)
        details = format_detections(detections)
        return drawn_image, details, detections
    except Exception as e:
        return image, f"Error: {str(e)}", []

# Gradio interface (updated)
with gr.Blocks(title="OWLv2 Object Detection") as demo:
    gr.Markdown("# OWLv2 Object Detection")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            objects_input = gr.Textbox(label="Objects to Detect (comma-separated)")
            threshold = gr.Slider(0.1, 1.0, value=0.1, step=0.05, label="Detection Threshold")
            detect_button = gr.Button("Detect Objects")
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_text = gr.Textbox(label="Detection Details", lines=10)
            output_detections = gr.JSON(label="Detections (for API)", visible=False)  # Hidden JSON component
    detect_button.click(
        fn=process_owl,
        inputs=[image_input, objects_input, threshold],
        outputs=[output_image, output_text, output_detections],  # Updated outputs
        api_name="predict"  # Expose as API endpoint
    )

def launch():
    demo.launch(server_port=owl_port)  # Adjust port if needed

if __name__ == "__main__":
    launch()