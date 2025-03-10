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

class DinoFactory():
    def __init__(self):
        self.dino_proc = multiprocessing.Process(target=dino_launch)
        self.dino_proc.start()
        self.client = Client(f"http://localhost:{dino_port}")

    def __del__(self):
        if self.dino_process.is_alive():
            self.dino_process.join()

    def _run(self, image: Img, objects_of_interest: List[str]) -> List[Bbox]:
        # Test with a valid image and input
        image_path = "data/sample/4girls.jpg"  # Replace with a real test image path
        result = self.client.predict(
            image=handle_file(image_path),
            object_list_text="cat, dog, person",
            box_threshold=0.2,
            text_threshold=0.1,
            api_name="/predict"
        )
        self.assertIsInstance(result[0], str)  # Output image path
        self.assertIsInstance(result[1], str)  # Detection details text
        self.assertIsInstance(result[2], list)  # Detections list
        self.assertTrue("Found" in result[1] or "No objects detected" in result[1])


from coc.tool.google_search import google_search