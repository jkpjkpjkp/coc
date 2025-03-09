import torch
from typing import List, TypedDict
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import requests
from io import BytesIO
import os

class Bbox(TypedDict):
    box: List[float]  # [x1, y1, x2, y2]
    score: float
    label: str

def box_trim(detections: List[Bbox]) -> List[Bbox]:
    occlusion_threshold = 0.3
    sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    accepted = []

    def area(box: List[float]) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x1 >= x2 or y1 >= y2:
            return 0.0
        return (x2 - x1) * (y2 - y1)

    for candidate in sorted_detections:
        keep = True
        for accepted_bbox in accepted:
            inter_area = intersection_area(candidate['box'], accepted_bbox['box'])
            accepted_area = area(accepted_bbox['box'])
            if accepted_area == 0:
                continue
            ioa = inter_area / accepted_area
            if ioa >= occlusion_threshold:
                keep = False
                break
        if keep:
            accepted.append(candidate)
    return accepted

def trim_result(detections: List[Bbox]) -> List[Bbox]:
    unique_labels = {bbox['label'] for bbox in detections}
    trimmed_results = []
    for label in unique_labels:
        label_detections = [d for d in detections if d['label'] == label]
        trimmed = box_trim(label_detections)
        trimmed_results.extend(trimmed)
    return trimmed_results

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

# API endpoints for the small servers (adjust URLs based on where they are running)
DINO_API_URL = f"http://localhost:{os.environ['dino_port']}/api/predict"
OWL_API_URL = f"http://localhost:{os.environ['owl_port']}/api/predict"

def process_combined(image, object_list_text, confidence, dino_box_threshold, dino_text_threshold, owl_threshold):
    if image is None:
        return None, "Please upload an image."
    objects = [obj.strip() for obj in object_list_text.split(",") if obj.strip()]
    if not objects:
        return image, "Please specify at least one object."

    # Convert PIL image to bytes for API call
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    try:
        # Call Grounding DINO API
        dino_payload = {
            "data": [
                {"image": img_byte_arr.hex(), "mime_type": "image/png"},
                object_list_text,
                confidence,
                dino_box_threshold,
                dino_text_threshold
            ]
        }
        dino_response = requests.post(DINO_API_URL, json=dino_payload)
        dino_response.raise_for_status()
        dino_data = dino_response.json()['data']
        dino_detections = eval(dino_data[1].replace("Error:", "")) if "Error:" in dino_data[1] else dino_data[1]['value'].split('\n')[1:-1]
        if isinstance(dino_detections, list) and not dino_detections:
            dino_detections = []
        else:
            dino_detections = [Bbox(box=[float(x) for x in det.split('box ')[1][1:-1].split(', ')], score=float(det.split('score ')[1].split(',')[0]), label=det.split('- ')[1].split(':')[0].strip()) for det in dino_detections if 'score' in det]

        # Call OWLv2 API
        owl_payload = {
            "data": [
                {"image": img_byte_arr.hex(), "mime_type": "image/png"},
                object_list_text,
                confidence,
                owl_threshold
            ]
        }
        owl_response = requests.post(OWL_API_URL, json=owl_payload)
        owl_response.raise_for_status()
        owl_data = owl_response.json()['data']
        owl_detections = eval(owl_data[1].replace("Error:", "")) if "Error:" in owl_data[1] else owl_data[1]['value'].split('\n')[1:-1]
        if isinstance(owl_detections, list) and not owl_detections:
            owl_detections = []
        else:
            owl_detections = [Bbox(box=[float(x) for x in det.split('box ')[1][1:-1].split(', ')], score=float(det.split('score ')[1].split(',')[0]), label=det.split('- ')[1].split(':')[0].strip()) for det in owl_detections if 'score' in det]

        # Apply _run logic
        g_dino_result = trim_result(dino_detections)
        nonempty = {x['label'] for x in owl_detections}
        final_detections = [x for x in g_dino_result if x['label'] in nonempty]
        filtered_detections = [det for det in final_detections if det['score'] >= confidence]

        drawn_image = draw_boxes(image.copy(), filtered_detections)
        details = format_detections(filtered_detections)
        return drawn_image, details
    except Exception as e:
        return image, f"Error in combined detection: {str(e)}"

# Gradio interface
with gr.Blocks(title="Combined Object Detection") as demo:
    gr.Markdown("# Combined Object Detection (Grounding DINO + OWLv2)")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            objects_input = gr.Textbox(label="Objects to Detect (comma-separated)")
            confidence = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence Threshold")
            dino_box_threshold = gr.Slider(0.1, 1.0, value=0.2, step=0.05, label="DINO Box Threshold")
            dino_text_threshold = gr.Slider(0.1, 1.0, value=0.1, step=0.05, label="DINO Text Threshold")
            owl_threshold = gr.Slider(0.1, 1.0, value=0.1, step=0.05, label="OWL Detection Threshold")
            detect_button = gr.Button("Detect Objects")
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_text = gr.Textbox(label="Detection Details", lines=10)
    detect_button.click(
        fn=process_combined,
        inputs=[image_input, objects_input, confidence, dino_box_threshold, dino_text_threshold, owl_threshold],
        outputs=[output_image, output_text]
    )
import multiprocessing
import time
import socket

def is_port_open(port):
    """Check if a port is open on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', int(port))) == 0

def launch_all():
    from coc.tool.grounding import dino_server, owl_server
    dino_process = multiprocessing.Process(target=dino_server.launch)
    owl_process = multiprocessing.Process(target=owl_server.launch)
    dino_process.start()
    owl_process.start()

    # Wait for servers to be ready
    max_wait = 60  # Maximum wait time in seconds
    start_time = time.time()
    dino_port = int(os.environ['dino_port'])
    owl_port = int(os.environ['owl_port'])

    while time.time() - start_time < max_wait:
        if is_port_open(dino_port) and is_port_open(owl_port):
            print(f"Servers are up - DINO on {dino_port}, OWL on {owl_port}")
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Servers did not start within {max_wait} seconds. Check DINO on port {dino_port} and OWL on port {owl_port}.")

    # Launch main interface
    demo.launch(server_port=int(os.environ['grounding_port']))

    # Cleanup
    dino_process.join()
    owl_process.join()

if __name__ == "__main__":
    launch_all()
