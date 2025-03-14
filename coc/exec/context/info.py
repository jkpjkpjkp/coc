_info = {
    'grounding': """
### Grounding Tools

These tools detect objects in an image based on text descriptions, returning bounding boxes around matching objects. A bounding box (`Bbox`) is a dictionary with:
- `box`: List of floats `[x1, y1, x2, y2]` (top-left and bottom-right coordinates).
- `score`: Float indicating confidence.
- `label`: String naming the detected object.

All functions return:
- Rendered image with bounding boxes drawn.
- String representation of the boxes.
- List of `Bbox` dictionaries.

#### `grounding(image: Img, objects_of_interest: List[str], owl_threshold=0.1, dino_box_threshold=0.2, dino_text_threshold=0.1)`

- **Purpose**: Detects objects accurately by combining Grounding DINO and OWL v2.
- **How it Works**: Grounding DINO detects objects, and OWL v2 filters out false positives (hallucinations).
- **Parameters**:
  - `image`: Input image.
  - `objects_of_interest`: List of strings (e.g., ["cat", "dog"]) to detect.
  - `owl_threshold`: OWL v2 filtering threshold (default: 0.1).
  - `dino_box_threshold`: Grounding DINO box confidence (default: 0.2).
  - `dino_text_threshold`: Grounding DINO text confidence (default: 0.1).
- **Notes**: Reliable, with minimal duplicates or hallucinations, but limited by Grounding DINO 1.0’s capabilities.

#### `grounding_dino(image: Img, objects_of_interest: List[str], box_threshold=0.2, text_threshold=0.1)`

- **Purpose**: Detects objects using Grounding DINO 1.0 alone.
- **Parameters**: Similar to `grounding`, without OWL filtering.
- **Notes**: May detect duplicates (multiple boxes for one object) or hallucinate (detect non-existent objects).

#### `owl(image: Img, objects_of_interest: List[str], threshold=0.15)`

- **Purpose**: Detects objects using OWL v2 alone.
- **Parameters**:
  - `threshold`: Confidence threshold (default: 0.15).
- **Notes**: Better at matching text to objects, less accurate box placement, avoids duplicates.

**Choosing a Tool**:
- Use `grounding` for balanced accuracy and reliability.
- Use `grounding_dino` for higher detection rates, accepting possible duplicates or errors.
- Use `owl` for better text alignment and fewer duplicates.
""",

    'sam': """
### SAM (Segment Anything Model) Tools

These tools segment images into regions or objects, returning masks that highlight those areas.

#### `sam_predict(image: Img, **kwargs)`

- **Purpose**: Segments specific parts of an image based on prompts like points or boxes.
- **Key Parameters** (all optional):
  - `point_coords`: Array of points (Nx2, in pixels) to mark areas (e.g., [[100, 150]]).
  - `point_labels`: Array of labels (1 for foreground, 0 for background).
  - `box`: Array `[x1, y1, x2, y2]` to focus segmentation.
  - `mask_input`: Previous mask (1x256x256) for refinement.
  - `multimask_output`: If `True`, returns multiple masks; if `False`, returns one (default: `True`).
- **Returns**:
  - Masks: Array (CxHxW) of binary or logit masks (C = number of masks).
  - Quality scores: Array of mask quality predictions.
  - Low-res logits: Array (Cx256x256) for further refinement.
- **Example**: Click a point on an object to segment it or provide a box to isolate an area.

#### `sam_auto(image: Img, **kwargs)`

- **Purpose**: Automatically segments the entire image without prompts.
- **Key Parameters** (all optional):
  - `points_per_side`: Number of grid points per side (e.g., 32).
  - `pred_iou_thresh`: Mask quality threshold (default: 0.88).
  - `stability_score_thresh`: Mask stability threshold (default: 0.95).
  - `min_mask_region_area`: Minimum mask area to keep (default: 0).
- **Returns**: List of dictionaries, each with:
  - `segmentation`: Mask (HxW array).
  - `bbox`: Box `[x, y, width, height]`.
  - `area`: Pixel count of the mask.
  - `predicted_iou`: Quality score.
  - `stability_score`: Stability measure.
  - `crop_box`: Crop area used.
- **Example**: Segments all objects in an image automatically.

**Choosing a Tool**:
- Use `sam_predict` for targeted segmentation with prompts.
- Use `sam_auto` for automatic, full-image segmentation.
""",

    'other': """
### Other Tools and Strategies

This section offers tips and helper strategies for using the tools effectively.

#### Tips for Using Tools
- **VLMs**: Great at understanding images with text but may miss small details. Best when the target is prominent.
- **Counting Objects**:
  1. Use `grounding` to find bounding boxes.
  2. Zoom into each box and use a VLM (e.g., `qwen`) to count or verify.
  - This reduces errors from grounding hallucinations.
- **If Grounding Misses Objects**: Use a sliding window with VLMs to scan the whole image.

#### Helper Strategies
- **Sliding Windows**: Split the image into smaller parts to detect small objects.
- **Draw Attention**: Add lines, boxes, or circles to guide VLM focus.
- **Cropping/Zooming**: Focus on specific areas for better results.
- **Mask Overlay**: Combine masks with the original image to visualize segments.
- **Resource Use**: Prioritize accuracy over efficiency—use as many resources as needed.

**Implementation**: Use Python packages (e.g., NumPy, OpenCV) to create these helpers. Import them as needed.
""",

    'vlm': """
### VLM (Visual Language Model) Tools

These models answer questions about images or generate text from images. Each takes an `image` and a `question`, returning a string answer.

#### `glm(image: Img, question: str)`

- **Model**: GLM-4V Plus.
- **Strength**: Excellent visual processing.
- **Weakness**: Weaker language integration.

#### `qwen(image: Img, question: str)`

- **Model**: Qwen2.5 72B.
- **Strength**: Best for documents, videos, and general tasks.

#### `gemini(image: Img, question: str)`

- **Model**: Gemini 2.0 Pro.
- **Strength**: Strong language skills, innovative in specific areas.

**Choosing a Model**:
- Use `qwen` for most tasks, especially documents or videos.
- Use `glm` for strong visual analysis.
- Use `gemini` for advanced language needs.
"""
}

class Info:
    def __init__(self):
        self._data = _info

    def __getitem__(self, key):
        return self._data[key]

    def __call__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(f"info for key '{key}' not found")

info = Info()