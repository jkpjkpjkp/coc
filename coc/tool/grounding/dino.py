import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, TypedDict
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from coc.config import dino_port

class Bbox(TypedDict):
    box: List[float]
    score: float
    label: str

class ObjectDetectionFactory:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gd_processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')

    def grounding_dino(self, image: Image.Image, texts: List[str], box_threshold=0.2, text_threshold=0.1) -> List[Bbox]:
        if not texts or not image:
            raise ValueError('Valid image and at least one text description required')
        image = image.convert('RGB')
        text = '. '.join(text.strip().lower() for text in texts) + '.'
        inputs = self.gd_processor(images=image, text=text, return_tensors='pt').to(self.device)
        self.gd_model.to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs, inputs['input_ids'], box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]]
        )[0]
        return [Bbox(box=box.tolist(), score=score.item(), label=label) for box, score, label in zip(results['boxes'], results['scores'], results['labels'])]

obj = ObjectDetectionFactory()

def draw_boxes(image: Image.Image, detections: List[Bbox]) -> Image.Image:
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {label: (int(255 * (hash(label) % 10) / 10), int(255 * ((hash(label) // 10) % 10) / 10), int(255 * ((hash(label) // 100) % 10) / 10)) for label in set(det['label'] for det in detections)}
    for det in detections:
        box, label, score = det['box'], det['label'], det['score']
        draw.rectangle(box, outline=colors[label], width=3)
        text = f"{label}: {score:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)[2:4]
        draw.rectangle([box[0], box[1] - text_size[1], box[0] + text_size[0], box[1]], fill=colors[label])
        draw.text((box[0], box[1] - text_size[1]), text, fill="white", font=font)
    return image

def format_detections(detections: List[Bbox]) -> str:
    return "No objects detected." if not detections else f"Found {len(detections)} objects:\n" + "\n".join(f"- {det['label']}: score {det['score']:.2f}, box {[int(b) for b in det['box']]}" for det in detections)

def process_dino(image, object_list_text, box_threshold, text_threshold):
    if not image:
        return None, "Please upload an image.", []
    objects = [obj.strip() for obj in object_list_text.split(",") if obj.strip()]
    if not objects:
        return image, "Please specify at least one object.", []
    try:
        detections = obj.grounding_dino(image, objects, box_threshold, text_threshold)
        return draw_boxes(image.copy(), detections), format_detections(detections), detections
    except Exception as e:
        return image, f"Error: {str(e)}", []

def launch():
    with gr.Blocks(title="Grounding DINO Object Detection") as demo:
        gr.Markdown("# Grounding DINO Object Detection")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                objects_input = gr.Textbox(label="Objects to Detect (comma-separated)")
                box_threshold = gr.Slider(0.1, 1.0, value=0.2, step=0.05, label="Box Threshold")
                text_threshold = gr.Slider(0.1, 1.0, value=0.1, step=0.05, label="Text Threshold")
                detect_button = gr.Button("Detect Objects")
            with gr.Column():
                output_image = gr.Image(label="Detection Results")
                output_text = gr.Textbox(label="Detection Details", lines=10)
                output_detections = gr.JSON(label="Detections (for API)", visible=False)  # Added hidden JSON component
        detect_button.click(
            process_dino,
            [image_input, objects_input, box_threshold, text_threshold],
            [output_image, output_text, output_detections],  # Updated outputs
            api_name="predict"
        )
    demo.launch(server_port=dino_port)

if __name__ == "__main__":
    launch()