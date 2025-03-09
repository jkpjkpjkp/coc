import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, TypedDict
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

class Bbox(TypedDict):
    box: List[float]  # [x1, y1, x2, y2]
    score: float
    label: str

class ObjectDetectionFactory:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gd_processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')

    def grounding_dino(self, image: Image.Image, texts: List[str], threshold=0.2, text_threshold=0.1) -> List[Bbox]:
        image = image.convert('RGB')
        if len(texts) > 1:
            detections = []
            for text in texts:
                detections.extend(self.grounding_dino(image, [text], threshold, text_threshold))
            return detections
        text = '. '.join(texts).strip().lower() + '.'
        inputs = self.gd_processor(images=image, text=text, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self.gd_model.to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs, inputs['input_ids'], threshold=threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]]
        )
        result = results[0]
        detections = [
            Bbox(box=box.tolist(), score=score.item(), label=label)
            for box, score, label in zip(result['boxes'], result['scores'], result['labels'])
        ]
        return detections

# Initialize the factory
obj = ObjectDetectionFactory()

def draw_boxes(image: Image.Image, detections: List[Bbox]) -> Image.Image:
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    colors = {label: (int(255 * (hash(label) % 10) / 10), int(255 * ((hash(label) // 10) % 10) / 10), int(255 * ((hash(label) // 100) % 10) / 10)) for label in set(det['label'] for det in detections)}
    for det in detections:
        box = det['box']
        label = det['label']
        score = det['score']
        draw.rectangle(box, outline=colors[label], width=3)
        text = f"{label}: {score:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)[2:4]
        draw.rectangle([box[0], box[1] - text_size[1], box[0] + text_size[0], box[1]], fill=colors[label])
        draw.text((box[0], box[1] - text_size[1]), text, fill="white", font=font)
    return image

def format_detections(detections: List[Bbox]) -> str:
    if not detections:
        return "No objects detected."
    text = f"Found {len(detections)} objects:\n"
    for det in detections:
        box = [int(b) for b in det['box']]
        text += f"- {det['label']}: score {det['score']:.2f}, box {box}\n"
    return text

def process_dino(image, object_list_text, confidence, box_threshold, text_threshold):
    if image is None:
        return None, "Please upload an image."
    objects = [obj.strip() for obj in object_list_text.split(",") if obj.strip()]
    if not objects:
        return image, "Please specify at least one object."
    try:
        detections = obj.grounding_dino(image, objects, threshold=box_threshold, text_threshold=text_threshold)
        filtered_detections = [det for det in detections if det['score'] >= confidence]
        drawn_image = draw_boxes(image.copy(), filtered_detections)
        details = format_detections(filtered_detections)
        return drawn_image, details
    except Exception as e:
        return image, f"Error: {str(e)}"

# Gradio interface
with gr.Blocks(title="Grounding DINO Object Detection") as demo:
    gr.Markdown("# Grounding DINO Object Detection")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            objects_input = gr.Textbox(label="Objects to Detect (comma-separated)")
            confidence = gr.Slider(0.1, 1.0, value=0.2, step=0.05, label="Confidence Threshold")
            box_threshold = gr.Slider(0.1, 1.0, value=0.2, step=0.05, label="Box Threshold")
            text_threshold = gr.Slider(0.1, 1.0, value=0.1, step=0.05, label="Text Threshold")
            detect_button = gr.Button("Detect Objects")
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_text = gr.Textbox(label="Detection Details", lines=10)
    detect_button.click(
        fn=process_dino,
        inputs=[image_input, objects_input, confidence, box_threshold, text_threshold],
        outputs=[output_image, output_text]
    )


def launch():
    import os
    demo.launch(server_port=int(os.environ['dino_port']))  # Adjust port if needed

if __name__ == "__main__":
    launch()