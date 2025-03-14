from typing import *
from PIL import Image
from PIL.Image import Image as Img
import numpy as np
from coc.tool.mod import *


class Bbox(TypedDict):
    """'bbox' stands for 'bounding box'"""
    box: List[float] # [x1, y1, x2, y2]
    score: float
    label: str

### grounding tools (add bbox to objects)
#   Returns:
#       rendered image (with bounding boxes drawn). Note: you cannot directly see this, but may use vlm (vqa tool below) to help you.
#       string form of bounding boxes.
#       list form of bounding boxes.

def grounding(image: Img, objects_of_interest: List[str], text: Optional[str]=None, owl_threshold=0.1, dino_box_threshold=0.2, dino_text_threshold=0.1) -> Tuple[Img, str, List[Bbox]]:
    """a combination of grounding dino and owl v2.

    grounding dino pre-mixes visual and text tokens, yielding better box accuracy.
        the downside of grounding dino, also because of pre-mixing, is it often hallucinates.
    so we use owl v2 to filter out hallucinated boxes.

    this implementation is generally duplication- and hallucination- free,
       but is limited by the capabilitie of pre-trained grounding dino 1.0.
    """
    if text:
        objects_of_interest.append(text)
    return get_grounding()(image, objects_of_interest, owl_threshold, dino_box_threshold, dino_text_threshold)

def grounding_dino(image: Img, objects_of_interest: List[str], box_threshold=0.2, text_threshold=0.1) -> Tuple[Img, str, List[Bbox]]:
    """grounding dino 1.0.

    boxes may duplicated (same object, multiple boxes) or hallucinate.
    """
    return get_dino()(image, objects_of_interest, box_threshold, text_threshold)

def owl(image: Img, objects_of_interest: List[str], threashold=0.15) -> Tuple[Img, str, List[Bbox]]:
    """owl v2.

    better text-image align, worse box IoU.
    boxes generally do not duplicate.
    """
    return get_owl()(image, objects_of_interest, threashold)



### vqa tools (visual language models)
"""comparison of options.

Qwen2.5 72B (Alibaba's Qwen): Excels in document understanding, video processing, and agentic capabilities.
GLM-4V Plus (Zhipu AI): Strong visual processing but struggles with language integration.
Gemini 2.0 Pro (Google DeepMind): Advanced multimodal and agentic features but lacks detailed performance data.

Qwen2.5 72B leads overall, while GLM-4V Plus and Gemini 2.0 Pro shine in specific visual and innovative areas, respectively.
"""

def glm(image: Img, question: str) -> str:
    """glm 4v plus"""
    return get_glm()(image, question)

def qwen(image: Img, question: str) -> str:
    """qwen vl2.5 72b"""
    return get_qwen()(image, question)

def gemini(image: Img, question: str) -> str:
    """gemeni 2.0 pro"""
    return get_gemini()(image, question)


### segment anything

def sam_predict(image: Img, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args (*all args are optional*):
        mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
        max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
        max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.

        point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
        point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
        box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
        mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
        multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
        return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
        normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] and point_coords is expected to be wrt. image dimensions.

    Returns:
        (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
        (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
        (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
    """
    return get_sam_predict()(image, **kwargs)

def sam_auto(image: Img, **kwargs) -> List[Dict[str, Any]]:
    """
    Args (*all args are optional*):
        points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
        pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
        stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
        stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
        mask_threshold (float): Threshold for binarizing the mask logits
        box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
        crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
        crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
        crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
        crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
        point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
        min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
        use_m2m (bool): Whether to add a one step refinement using previous mask predictions.
        multimask_output (bool): Whether to output multimask at each point of the grid.

    Returns:
        list(dict(str, any)): A list over records for masks. Each record is
            a dict containing the following keys:

            segmentation (np.ndarray): The mask. an array of shape HW.
            bbox (list(float)): The box around the mask, in XYWH format.
            area (int): The area in pixels of the mask.
            predicted_iou (float): The model's own prediction of the mask's
                quality. This is filtered by the pred_iou_thresh parameter.
            point_coords (list(list(float))): The point coordinates input
                to the model to generate this mask.
            stability_score (float): A measure of the mask's quality. This
                is filtered on using the stability_score_thresh parameter.
            crop_box (list(float)): The crop of the image used to generate
                the mask, given in XYWH format.
    """
    return get_sam_auto()(image, **kwargs)



# depth anything
def depth(image: Img) -> np.ndarray:
    """depth anything.

    Returns:
        np.ndarray: H*W depth map.
    """
    return get_depth()(image)


### search the web (text only)
def google_search(query: str) -> List[str]:
    """google search"""
    return google_search(query)


### general experience with existing tools
"""experience.

VLMs has the best understanding of images, and greatly aligns with text.
however, they will ignore details, and cannot see very clearly.
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

also, you may wish to superpose mask returns with the orignal image to better understand the result.


all these tools can be simply implemented with various python packages (remember to import them before use), and you are encouraged to do so.

remember, we do not care how much resource you use, but a strong and correct conclusion is paramountly important.
"""

from .info import info

def display(image: Img) -> str:
    """a pseudo-display function, that instead returns a dense caption. """
    return qwen(image, 'describe the image in detail.')

class Task(TypedDict):
    images: List[Img]
    question: str

task: Task # this variable is already initialized to the current task you are solving.
