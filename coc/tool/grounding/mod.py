"""visual grounding g_dino and owl2.

using two Gradio servers.
"""
import torch
from typing import List, TypedDict
import PIL.Image
from PIL.Image import Image as Img
import requests
import base64
from io import BytesIO
from .dino import draw_boxes
from .dino import get_dino
from .owl import get_owl

class Bbox(TypedDict):
    """'bbox' stands for 'bounding box'"""
    box: List[float]  # [x1, y1, x2, y2]
    score: float
    label: str

def box_trim(detections: List[Bbox]) -> List[Bbox]:
    """Trim overlapping detections based on occlusion threshold."""
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

def pil_to_base64(image: Img) -> str:
    """Convert a PIL image to a base64-encoded string for API requests."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

class ObjectDetectionFactory:
    """Grounding tool interfacing for object detection."""
    @classmethod
    def trim_result(cls, detections: List[Bbox]) -> List[Bbox]:
        """Group detections by label and trim each group."""
        unique_labels = {bbox['label'] for bbox in detections}
        trimmed_results = []
        for label in unique_labels:
            label_detections = [d for d in detections if d['label'] == label]
            trimmed = box_trim(label_detections)
            trimmed_results.extend(trimmed)
        return trimmed_results

    def _run(self, image: Img, texts: List[str]) -> List[Bbox]:
        """Run detection using both servers and combine results."""
        if not isinstance(image, Img):
            image = PIL.Image.fromarray(image)
        image = image.convert('RGB')

        owl_result = get_owl()(image, texts, threshold=0.1)
        g_dino_result = get_dino()(image, texts, box_threshold=0.2, text_threshold=0.1)
        g_dino_result = self.trim_result(g_dino_result)

        nonempty = {x['label'] for x in owl_result}
        ret = [x for x in g_dino_result if x['label'] in nonempty]

        if isinstance(ret, tuple) and len(ret) == 1:
            ret = ret[0]
        return ret

_object_detection = ObjectDetectionFactory()

def get_grounding():
    return _object_detection._run

def demo1():
    obj = ObjectDetectionFactory()
    image_path = '/home/jkp/hack/coc/data/sample/onions.jpg'
    image = PIL.Image.open(image_path)
    ret = obj._run(texts=['boy', 'girl', 'an onion'], image=image)
    if isinstance(ret, tuple) and len(ret) == 1:
        ret = ret[0]
    print(ret)
    res = draw_boxes(image, ret)
    res.save('raw_onions.jpg')

def demo2():
    import gradio as gr
    demo = gr.Interface(
        fn=get_grounding(),
        inputs=[
            gr.Image(type="pil"),
            gr.Textbox(label="Objects of Interest (comma-separated)")
        ],
        outputs=[
            gr.JSON(label="Raw Detection Data"),
        ],
        title="Object Grounding Demo",
        description="Upload an image and specify objects to detect, separated by commas."
    )
    app = demo.launch(share=False)

if __name__ == '__main__':
    demo2()