import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import gradio as gr
import numpy as np
import torch
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
if __name__ == '__main__':
    print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "/home/jkp/Pictures/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
model = sam2

mask_generator = SAM2AutomaticMaskGenerator(sam2)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

# Function for _generate_masks server
def generate_masks_fn(
    image,
    points_per_side,
    points_per_batch,
    pred_iou_thresh,
    stability_score_thresh,
    stability_score_offset,
    mask_threshold,
    box_nms_thresh,
    crop_n_layers,
    crop_nms_thresh,
    crop_overlap_ratio,
    crop_n_points_downscale_factor,
    min_mask_region_area,
    use_m2m,
    multimask_output
):
    """
    Generates mask data from an image using specified parameters.

    Args:
        image (np.ndarray): Input image in HWC uint8 format.
        points_per_side (int): Number of points per side of the grid.
        points_per_batch (int): Number of points processed per batch.
        ... (other parameters from __init__)

    Returns:
        dict: MaskData as a dictionary with numpy arrays.
    """
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        mask_threshold=mask_threshold,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        use_m2m=use_m2m,
        multimask_output=multimask_output,
        output_mode="binary_mask"  # Default, not used in _generate_masks directly
    )
    mask_data = generator._generate_masks(image)
    # Convert MaskData to a dictionary with numpy arrays
    mask_data_dict = {
        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in mask_data.items()
    }
    return mask_data_dict

# Function for _post_process_mask_data server
def post_process_mask_data_fn(mask_data_dict, output_mode):
    """
    Post-processes mask data into annotations.

    Args:
        mask_data_dict (dict): Mask data as a dictionary.
        output_mode (str): Format of the output masks ('binary_mask', 'uncompressed_rle', 'coco_rle').

    Returns:
        list: List of mask annotation dictionaries.
    """
    # Create an instance with output_mode and default parameters
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        output_mode=output_mode,
        # Default values for unused parameters
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=False,
        multimask_output=True
    )
    # Pass the dictionary directly, assuming _post_process_mask_data can handle it
    annotations = generator._post_process_mask_data(mask_data_dict)
    return annotations

# Function for generate server
def generate_fn(
    image,
    points_per_side,
    points_per_batch,
    pred_iou_thresh,
    stability_score_thresh,
    stability_score_offset,
    mask_threshold,
    box_nms_thresh,
    crop_n_layers,
    crop_nms_thresh,
    crop_overlap_ratio,
    crop_n_points_downscale_factor,
    min_mask_region_area,
    use_m2m,
    multimask_output,
    output_mode
):
    """
    Generates masks for an image by calling generate_masks_fn and post_process_mask_data_fn.

    Args:
        image (np.ndarray): Input image in HWC uint8 format.
        ... (all __init__ parameters)

    Returns:
        list: List of mask annotation dictionaries.
    """
    # Generate mask data
    mask_data_dict = generate_masks_fn(
        image,
        points_per_side,
        points_per_batch,
        pred_iou_thresh,
        stability_score_thresh,
        stability_score_offset,
        mask_threshold,
        box_nms_thresh,
        crop_n_layers,
        crop_nms_thresh,
        crop_overlap_ratio,
        crop_n_points_downscale_factor,
        min_mask_region_area,
        use_m2m,
        multimask_output
    )
    # Post-process mask data
    annotations = post_process_mask_data_fn(mask_data_dict, output_mode)
    return annotations

# Define Gradio interfaces
generate_masks_interface = gr.Interface(
    fn=generate_masks_fn,
    inputs=[
        gr.Image(type="numpy", label="Image"),
        gr.Slider(1, 100, step=1, value=32, label="Points per Side"),
        gr.Slider(1, 256, step=1, value=64, label="Points per Batch"),
        gr.Slider(0.0, 1.0, step=0.01, value=0.8, label="Predicted IoU Threshold"),
        gr.Slider(0.0, 1.0, step=0.01, value=0.95, label="Stability Score Threshold"),
        gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="Stability Score Offset"),
        gr.Slider(-1.0, 1.0, step=0.01, value=0.0, label="Mask Threshold"),
        gr.Slider(0.0, 1.0, step=0.01, value=0.7, label="Box NMS Threshold"),
        gr.Slider(0, 5, step=1, value=0, label="Crop Layers"),
        gr.Slider(0.0, 1.0, step=0.01, value=0.7, label="Crop NMS Threshold"),
        gr.Slider(0.0, 1.0, step=0.01, value=512/1500, label="Crop Overlap Ratio"),
        gr.Slider(1, 5, step=1, value=1, label="Crop Points Downscale Factor"),
        gr.Slider(0, 1000, step=1, value=0, label="Min Mask Region Area"),
        gr.Checkbox(value=False, label="Use M2M"),
        gr.Checkbox(value=True, label="Multimask Output")
    ],
    outputs=gr.JSON(label="Mask Data"),
    title="Generate Masks",
    description="Generate raw mask data from an image."
)

post_process_mask_data_interface = gr.Interface(
    fn=post_process_mask_data_fn,
    inputs=[
        gr.JSON(label="Mask Data"),
        gr.Radio(["binary_mask", "uncompressed_rle", "coco_rle"], value="binary_mask", label="Output Mode")
    ],
    outputs=gr.JSON(label="Annotations"),
    title="Post Process Mask Data",
    description="Convert mask data into final annotations."
)

generate_interface = gr.Interface(
    fn=generate_fn,
    inputs=[
        gr.Image(type="numpy", label="Image"),
        gr.Slider(1, 100, step=1, value=32, label="Points per Side"),
        gr.Slider(1, 256, step=1, value=64, label="Points per Batch"),
        gr.Slider(0.0, 1.0, step=0.01, value=0.8, label="Predicted IoU Threshold"),
        gr.Slider(0.0, 1.0, step=0.01, value=0.95, label="Stability Score Threshold"),
        gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="Stability Score Offset"),
        gr.Slider(-1.0, 1.0, step=0.01, value=0.0, label="Mask Threshold"),
        gr.Slider(0.0, 1.0, step=0.01, value=0.7, label="Box NMS Threshold"),
        gr.Slider(0, 5, step=1, value=0, label="Crop Layers"),
        gr.Slider(0.0, 1.0, step=0.01, value=0.7, label="Crop NMS Threshold"),
        gr.Slider(0.0, 1.0, step=0.01, value=512/1500, label="Crop Overlap Ratio"),
        gr.Slider(1, 5, step=1, value=1, label="Crop Points Downscale Factor"),
        gr.Slider(0, 1000, step=1, value=0, label="Min Mask Region Area"),
        gr.Checkbox(value=False, label="Use M2M"),
        gr.Checkbox(value=True, label="Multimask Output"),
        gr.Radio(["binary_mask", "uncompressed_rle", "coco_rle"], value="binary_mask", label="Output Mode")
    ],
    outputs=gr.JSON(label="Annotations"),
    title="Generate Annotations",
    description="Generate mask annotations using both mask generation and post-processing."
)

# Combine interfaces into a tabbed application
app = gr.TabbedInterface(
    [generate_masks_interface, post_process_mask_data_interface, generate_interface],
    tab_names=["Generate Masks", "Post Process Mask Data", "Generate Annotations"],
    title="SAM2 Automatic Mask Generator"
)

if __name__ == '__main__':
    app.launch()