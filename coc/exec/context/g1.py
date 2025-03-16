from typing import *
from PIL import Image
from PIL.Image import Image as Img
import numpy as np
from coc.tool.mod import *

class Bbox(TypedDict):
    """'bbox' stands for 'bounding box'"""
    box: List[float]  # [x1, y1, x2, y2]
    score: float
    label: str

# Brief overview of available tools
print("""
Available tool types:
- grounding: Tools for adding bounding boxes to objects in images.
- vqa: Visual language models for answering questions about images.
- segment_anything: Tools for segmenting images.
- depth_estimation: Tool for estimating depth in images.
- web_search: Tool for searching the web.

For detailed information about a specific tool type, use info(tool_type).
For general experience and tips on using these tools, use info('general_experience').
For encouragement and ideas on implementing other helper tools, use info('other_tools').
""")

# Define the info function
tool_info = {
    'grounding': '''Grounding Tools:
These tools are used to add bounding boxes to objects in images.
Available functions:
- grounding(image: Img, objects_of_interest: List[str], owl_threshold=0.1, dino_box_threshold=0.2, dino_text_threshold=0.1) -> Tuple[Img, str, List[Bbox]]
    A combination of grounding dino and owl v2. Returns rendered image with bounding boxes, string form, and list form of bounding boxes.
- grounding_dino(image: Img, objects_of_interest: List[str], box_threshold=0.2, text_threshold=0.1) -> Tuple[Img, str, List[Bbox]]
    Grounding dino 1.0. May have duplicated or hallucinated boxes.
- owl(image: Img, objects_of_interest: List[str], threshold=0.15) -> Tuple[Img, str, List[Bbox]]
    Owl v2. Better text-image alignment but worse box IoU. Generally no duplicated boxes.
''',
    'vqa': '''VQA Tools:
These are visual language models that can answer questions about images.
Available functions:
- glm(image: Img, question: str) -> str
    GLM 4V Plus.
- qwen(image: Img, question: str) -> str
    Qwen VL2.5 72B.
- gemini(image: Img, question: str) -> str
    Gemini 2.0 Pro.
''',
    'segment_anything': '''Segment Anything Tools:
These tools are for segmenting images.
Available functions:
- sam_predict(image: Img, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    Predicts masks based on various input prompts like points, boxes, or masks.
- sam_auto(image: Img, **kwargs) -> List[Dict[str, Any]]
    Automatically generates masks for the entire image.
''',
    'depth_estimation': '''Depth Estimation Tool:
Available functions:
- depth(image: Img) -> np.ndarray
    Returns a depth map of the image.
''',
    'web_search': '''Web Search Tool:
Available functions:
- google_search(query: str) -> List[str]
    Performs a Google search and returns a list of results.
''',
    'general_experience': '''General Experience with Existing Tools:
VLMs have the best understanding of images and greatly align with text. However, they may ignore details and cannot see very clearly. When the information of interest occupies a large portion of the image, they are reliable.
A good strategy for counting things is to first use grounding to get bounding boxes, then zoom in on each one and query a VLM for each bbox. This uses VLM's strength to compensate for potential hallucinations in grounding tools.
If some objects are completely unnoticed by the grounding tool, consider using a sliding window focus that covers the whole image and use VLM to count objects in each window.
''',
    'other_tools': '''Other Tools:
The above tools are those with tricky implementations. You are encouraged to implement other helper functions or code as needed.
For example, if objects are too small to detect, apply a sliding window and pass each window to the tool.
You can also draw helper lines, bounding boxes, or circles to focus VLM's attention, crop or zoom in on the image, or superpose mask returns with the original image to better understand the result.
Implement these tools using various Python packages (remember to import them before use). Resource usage is not a concern; a strong and correct conclusion is paramount.
'''
}

def info(tool_type: str) -> str:
    return tool_info.get(tool_type, 'Tool type not found')

# Tool function definitions
def grounding(image: Img, objects_of_interest: List[str], owl_threshold=0.1, dino_box_threshold=0.2, dino_text_threshold=0.1) -> Tuple[Img, str, List[Bbox]]:
    return get_grounding()(image, objects_of_interest, owl_threshold, dino_box_threshold, dino_text_threshold)

def grounding_dino(image: Img, objects_of_interest: List[str], box_threshold=0.2, text_threshold=0.1) -> Tuple[Img, str, List[Bbox]]:
    return get_dino()(image, objects_of_interest, box_threshold, text_threshold)

def owl(image: Img, objects_of_interest: List[str], threshold=0.15) -> Tuple[Img, str, List[Bbox]]:
    return get_owl()(image, objects_of_interest, threshold)

def glm(image: Img, question: str) -> str:
    return get_glm()(image, question)

def qwen(image: Img, question: str) -> str:
    return get_qwen()(image, question)

def gemini(image: Img, question: str) -> str:
    return get_gemini()(image, question)

def sam_predict(image: Img, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return get_sam_predict()(image, **kwargs)

def sam_auto(image: Img, **kwargs) -> List[Dict[str, Any]]:
    return get_sam_auto()(image, **kwargs)

def depth(image: Img) -> np.ndarray:
    return get_depth()(image)

def google_search(query: str) -> List[str]:
    return google_search(query)

class Task(TypedDict):
    images: List[Img]
    question: str

task: Task  # this variable is already initialized to the current task you are solving.