from coc.tool.grounding.dino import launch as dino_launch
from coc.tool.grounding.owl import launch as owl_launch
from coc.tool.sam.ultralytic import get_sam
from coc.tool.vqa import get_glm, get_qwen, get_gemini
# from coc.tool.grounding.mod import get_grounding, get_grounding_dino, get_owl
from coc.config import dino_port
from gradio_client import Client, handle_file
from PIL.Image import Image as Img
from typing import *
import multiprocessing

class Bbox(TypedDict):
    """'bbox' stands for 'bounding box'"""
    box: List[float] # [x1, y1, x2, y2]
    score: float
    label: str

class ObjectDetectionFactory:
    """Factory for managing grounding detection services."""
    def __init__(self):
        # Start both detection servers
        self.dino_proc = multiprocessing.Process(target=dino_launch)
        self.owl_proc = multiprocessing.Process(target=owl_launch)
        self.dino_proc.start()
        self.owl_proc.start()
        
        # Create API clients
        self.dino_client = Client(f"http://localhost:{dino_port}")
        self.owl_client = Client(f"http://localhost:{owl_port}")

    def __del__(self):
        """Cleanup processes on destruction"""
        for proc in [self.dino_proc, self.owl_proc]:
            if proc.is_alive():
                proc.join()

    def grounding_dino(self, image_path: str, texts: List[str], box_thresh=0.2, text_thresh=0.1) -> List[Bbox]:
        """Run Grounding DINO detection through Gradio API"""
        result = self.dino_client.predict(
            image=handle_file(image_path),
            object_list_text=", ".join(texts),
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            api_name="/predict"
        )
        return result[2]  # Return detections list

    def owl(self, image_path: str, texts: List[str], threshold=0.1) -> List[Bbox]:
        """Run OWLv2 detection through Gradio API"""
        result = self.owl_client.predict(
            image=handle_file(image_path),
            object_list_text=", ".join(texts),
            threshold=threshold,
            api_name="/predict"
        )
        return result[2]  # Return detections list

# Singleton instance for the factory
_grounding_factory = ObjectDetectionFactory()

def get_grounding_dino():
    return _grounding_factory.grounding_dino

def get_owl():
    return _grounding_factory.owl


from coc.tool.google_search import google_search
