"""visual grounding.
"""
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection,\
                         Owlv2Processor, Owlv2ForObjectDetection
from dataclasses import dataclass
from typing import List
import PIL.Image
from PIL.Image import Image as Img

@dataclass
class Bbox:
    box: List[float]
    score: float
    label: str

def box_trim(detections: List[Bbox]) -> List[Bbox]:
    occlusion_threshold = 0.3

    sorted_detections = sorted(detections, key=lambda x: x.score, reverse=True)
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
            inter_area = intersection_area(candidate.box, accepted_bbox.box)
            accepted_area = area(accepted_bbox.box)
            if accepted_area == 0:
                continue
            ioa = inter_area / accepted_area
            if ioa >= occlusion_threshold:
                keep = False
                break
        if keep:
            accepted.append(candidate)
    return accepted

class ObjectDetectionFactory:
    """grounding tool.

    in compliance with LangChain Tool format.
    interface:
        def _run(self, image: Img, texts: List[str]) -> List[Bbox]
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.gd_processor = AutoProcessor.from_pretrained(
            'IDEA-Research/grounding-dino-base'
        )
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            'IDEA-Research/grounding-dino-base'
        )#.to(self.device)

        self.owlv2_processor = Owlv2Processor.from_pretrained(
            'google/owlv2-base-patch16-ensemble'
        )
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
            'google/owlv2-base-patch16-ensemble'
        )#.to(self.device)

    def grounding_dino(self, image: Img, texts: List[str]) -> List[Bbox]:
        image = image.convert('RGB')

        # PATCH
        # we decide to patch multi-object detection like this.
        if len(texts) > 1:
            detections = []
            for text in texts:
                detections.extend(self.grounding_dino(image, [text]))
            return detections

        text = '. '.join(texts).strip().lower() + '.'

        inputs = self.gd_processor(
            images=image, text=text, return_tensors='pt'
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self.gd_model.to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)
        # self.gd_model.to('cpu')
            # torch.cuda.empty_cache()

        results = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs['input_ids'],
            threshold=0.2,
            text_threshold=0.1,
            target_sizes=[image.size[::-1]]
        )

        result = results[0]

        # Convert results to list of Bbox
        detections = [
            Bbox(box=box.tolist(), score=score.item(), label=label)
            for box, score, label in \
                zip(result['boxes'], result['scores'], result['text_labels'])
        ]
        return detections

    @classmethod
    def trim_result(cls, detections: List[Bbox]) -> List[Bbox]:
        # Group by label and trim each group
        unique_labels = {bbox.label for bbox in detections}
        trimmed_results = []
        for label in unique_labels:
            label_detections = [d for d in detections if d.label == label]
            trimmed = box_trim(label_detections)
            trimmed_results.extend(trimmed)
        return trimmed_results

    def owl2(self, image: Img, texts: List[str]) -> List[Bbox]:
        image = image.convert('RGB')

        inputs = self.owlv2_processor(
            text=texts, images=image, return_tensors='pt'
        ).to(self.device)
        self.owlv2_model.to(self.device)
        with torch.no_grad():
            outputs = self.owlv2_model(**inputs)
        # self.owlv2_model.to('cpu')
            # torch.cuda.empty_cache()

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        processed_results = self.owlv2_processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1
        )
        result = processed_results[0]

        detections = []
        for score, label_idx, box in \
            zip(result['scores'], result['labels'], result['boxes']):
            label = texts[label_idx.item()]
            detections.append(Bbox(
                box=box.tolist(),
                score=score.item(),
                label=label
            ))
        return detections

    def _run(self, image: Img, texts: List[str]) -> List[Bbox]:
        if not isinstance(image, Img):
            image = PIL.Image.fromarray(image)
        image = image.convert('RGB')

        owl_result = self.owl2(image, texts)
        g_dino_result = self.grounding_dino(image, texts)
        g_dino_result = type(self).trim_result(g_dino_result)
        nonempty = {x.label for x in owl_result}
        ret = [x for x in g_dino_result if x.label in nonempty]
        if isinstance(ret, tuple) and len(ret) == 1:
            ret = ret[0]
        torch.cuda.empty_cache()
        return ret

_object_detection = ObjectDetectionFactory()

def get_grounding():
    return _object_detection._run

def get_grounding_dino():
    return _object_detection.grounding_dino

def get_owl():
    return _object_detection.owl2

def demo1():
    obj = ObjectDetectionFactory()
    image_path = '/home/jkp/hack/coc/data/sample/onions.jpg'
    image = PIL.Image.open(image_path)
    from coc.tool.grounding.draw import draw
    from coc.tool.grounding._51 import envision
    ret = obj._run(
            texts=['boy', 'girl', 'an onion'],
            image=image
        ),
    if isinstance(ret, tuple) and len(ret) == 1:
        ret = ret[0]
    print(ret)

    draw(image, ret, 'raw_onions.jpg')
    envision(
        image_path,
        ret,
        # output_path = 'raw_person.jpg'
    )

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