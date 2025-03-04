import gradio as gr
import numpy as np
from typing import List, Optional, TypedDict
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as Img
from coc.tool.factory import *

class Bbox(TypedDict):
    """'bbox' stands for 'bounding box'"""
    box: List[float] # [x1, y1, x2, y2]
    score: float
    label: str

### grounding tools (add bbox to objects)
def grounding(image: Img, objects_of_interest: List[str]) -> List[Bbox]:
    """a combination of grounding dino and owl v2.
    grounding dino pre-mixes visual and text tokens, yielding better box accuracy.
        the downside of grounding dino, also because of pre-mixing, is it often hallucinates.
    so we use owl v2 to filter out hallucinated boxes.
    this implementation is generally duplication- and hallucination- free,
       but is limited by the capabilitie of pre-trained grounding dino 1.0.
    """
    return get_grounding()(image, objects_of_interest)

def process_image(input_image, object_list_text, confidence_threshold=0.5):
    """Process image with the grounding function and draw bounding boxes"""
    if input_image is None:
        return None, "Please upload an image."

    # Convert string of objects to list
    objects_of_interest = [obj.strip() for obj in object_list_text.split(",") if obj.strip()]

    if not objects_of_interest:
        return input_image, "Please specify at least one object of interest."

    # Convert to PIL Image if needed
    if isinstance(input_image, np.ndarray):
        pil_image = Image.fromarray(input_image)
    else:
        pil_image = input_image

    # Get detections
    try:
        detections = grounding(pil_image, objects_of_interest)
    except Exception as e:
        return input_image, f"Error in detection: {str(e)}"

    # Filter by confidence threshold
    filtered_detections = [det for det in detections if det.score >= confidence_threshold]

    # Create a copy of the image to draw on
    result_image = pil_image.copy()
    draw = ImageDraw.Draw(result_image)

    # Try to get a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw bounding boxes and labels
    colors = {
        label: (
            int(255 * (hash(label) % 10) / 10),
            int(255 * ((hash(label) // 10) % 10) / 10),
            int(255 * ((hash(label) // 100) % 10) / 10)
        )
        for label in set(det.label for det in filtered_detections)
    }

    results_text = f"Found {len(filtered_detections)} objects:\n"

    for det in filtered_detections:
        box = det.box
        label = det.label
        score = det.score

        # Convert box coordinates if they are normalized
        if max(box) <= 1.0:
            width, height = pil_image.size
            box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]

        # Draw rectangle
        draw.rectangle(box, outline=colors[label], width=3)

        # Draw label and score
        text = f"{label}: {score:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)[2:4]
        draw.rectangle(
            [box[0], box[1] - text_size[1], box[0] + text_size[0], box[1]],
            fill=colors[label]
        )
        draw.text((box[0], box[1] - text_size[1]), text, fill="white", font=font)

        results_text += f"- {label}: score {score:.2f}, box {[int(b) for b in box]}\n"

    if not filtered_detections:
        results_text = f"No objects detected with confidence threshold {confidence_threshold}."

    return result_image, results_text

# Create Gradio interface
with gr.Blocks(title="Object Detection with Grounding") as demo:
    gr.Markdown("# Object Detection with Grounding DINO and OWL v2")
    gr.Markdown("Upload an image and specify objects to detect, separated by commas.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            objects_input = gr.Textbox(
                label="Objects to Detect (comma-separated)",
                placeholder="person, car, dog, cat, ..."
            )
            confidence = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Confidence Threshold"
            )
            detect_button = gr.Button("Detect Objects")

        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_text = gr.Textbox(label="Detection Details", lines=10)

    detect_button.click(
        fn=process_image,
        inputs=[input_image, objects_input, confidence],
        outputs=[output_image, output_text]
    )

    gr.Examples(
        [
            ["example_image.jpg", "person, chair, table", 0.5],
            ["example_image2.jpg", "car, bicycle, dog", 0.4]
        ],
        inputs=[input_image, objects_input, confidence]
    )

    gr.Markdown("""
    ## How to use
    1. Upload an image using the input panel
    2. Enter objects you want to detect, separated by commas (e.g., "person, dog, car")
    3. Adjust the confidence threshold if needed
    4. Click "Detect Objects" to run the detection

    ## About
    This interface uses a combination of Grounding DINO and OWL v2 models to detect objects in images.
    Grounding DINO provides better box accuracy while OWL v2 helps filter out hallucinated detections.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
