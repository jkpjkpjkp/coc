from typing import *
from PIL.Image import Image as Img
import numpy as np
from .mod import *


class Bbox(TypedDict):
    """'bbox' stands for 'bounding box'"""
    box: List[float] # [x1, y1, x2, y2]
    score: float
    label: str

### grounding tools (add bbox to objects)
#   Returns:
#      image: Img, a rendering of boxes on the input image
#      details: str, a string representation of the boxes
#      boxes: List[Bbox], a list of boxes, each box is a dict with keys 'box', 'score', 'label'

def grounding(image: Img, objects_of_interest: List[str]) -> Tuple[Img, str, List[Bbox]]:
    """a combination of grounding dino and owl v2.

    grounding dino pre-mixes visual and text tokens, yielding better box accuracy.
        the downside of grounding dino, also because of pre-mixing, is it often hallucinates.
    so we use owl v2 to filter out hallucinated boxes.

    this implementation is generally duplication- and hallucination- free,
       but is limited by the capabilitie of pre-trained grounding dino 1.0.
    """
    return get_grounding()(image, objects_of_interest)

def grounding_dino(image: Img, objects_of_interest: List[str]) -> Tuple[Img, str, List[Bbox]]:
    """grounding dino 1.0.

    boxes may duplicated (same object, multiple boxes) or hallucinate
    """
    return get_dino()(image, objects_of_interest)

def owl(image: Img, objects_of_interest: List[str]) -> Tuple[Img, str, List[Bbox]]:
    """owl v2.

    better text-image align, worse box IoU.
    boxes generally do not duplicate.
    """
    return get_owl()(image, objects_of_interest)



### vqa tools (visual language models)
"""comparison of options.

Qwen2.5 72B (Alibaba's Qwen): Excels in document understanding, video processing, and agentic capabilities.
GLM-4V Plus (Zhipu AI): Strong visual processing but struggles with language integration.
Gemini 2.0 Pro (Google DeepMind): Advanced multimodal and agentic features but lacks detailed performance data.

Gemini 2.0 Pro leads overall, while GLM-4V Plus and Qwen2.5 72B shine in specific visual and innovative areas, respectively.
a good strategy is to default to Qwen2.5, and use GLM-4V and Gemini to verify tricky ones.
"""

def glm(image: Img, question: str) -> str:
    """glm 4v plus"""
    return get_glm()(image, question)

def qwen(image: Img, question: str) -> str:
    """qwen vl2.5 72b"""
    return get_qwen()(image, question)

def gemini(*args: Tuple[Union[Img, str]]) -> str:
    """gemeni 2.0 pro.

    it supports multiple images and text, in arbitrary order.
    """
    return get_gemini()(*args)


# segment anything
def segment_anything(image: Img) -> List[Dict]:
    """generate masks of objects. """
    return get_sam()(image)


# depth anything
def depth_anything(image: Img) -> np.ndarray:
    """monocular depth estimation.

    Returns:
        np.ndarray: HxW depth map as a NumPy array.
    """
    return get_depth()(image)


### search the web (text only)
def google_search(query: str) -> List[str]:
    """google search"""
    return google_search(query)


### general advice on using tools
"""experience.

VLMs understand images best, and aligns with text.
but they cannot see very clearly and ignore details.
but when the information of interest occupies a large portion of the image, they are reliable.

so a good strategy, for example, of counting things, would be to first use grouding to get some bboxes,
then zoom in on each one, querying a vlm for each bbox.
the benefit of this strategy is that it uses vlm's strength to compensate for that grounding tool may sometimes hallucinate, or think multiple objects are one.

this stragegy, of course, will fail if some object of interest is completely unnoticed by the grounding tool.

in which case and other potential pitfalls, it is up to your ingenuity to think of compensating strategies or further validations.
(for this particular pitfall, one may, e.g., use a sliding window focus that covers the whole image, and use vlm to count objects in each.)
"""



### other tools
"""other tools.

the above are tools whose implementation is somehow tricky.
feel free to implement other helper function/code.

for example, all tools cannot detect objects that is too small; in which case it will be helpful to apply sliding window, passing each window to the tool.

also, you may draw helper lines/bbox/circles to focus vlm's attention.

also, you may wish to crop/zoomin the image.

also, you may wish to superpose masks/depth-maps with the orignal image to better understand the result.


all these other tools can be simply implemented with a python package (remember to import them before use), and you are highly encouraged to do so.

remember, we do not care how much resource you use, how many iterations you take, but a strong and correct conclusion is of paramount importance.
"""



class Task(TypedDict):
    images: List[Img]
    question: str

task: Task # this variable will be initialized to the current task you are solving.