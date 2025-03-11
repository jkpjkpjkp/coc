import fiftyone as fo
import numpy as np
from ultralytics import SAM
import os
from PIL import Image

def sam_masks_to_fo_detections(results, label="object"):
    """Convert SAM masks to FiftyOne detections with bounding boxes"""
    detections = []

    for result in results:
        if result.masks is None:
            continue

        # Get image dimensions from results
        height, width = result.orig_shape

        for mask in result.masks:
            # Extract mask array and find bounding box coordinates
            mask_array = mask.data[0].cpu().numpy()
            y, x = np.where(mask_array)

            if len(x) == 0 or len(y) == 0:
                continue  # Skip empty masks

            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            # Convert to relative coordinates [x_center, y_center, width, height]
            x_rel = x_min / width
            y_rel = y_min / height
            w_rel = (x_max - x_min) / width
            h_rel = (y_max - y_min) / height

            # Get confidence if available
            conf = mask.conf if hasattr(mask, 'conf') else None

            detections.append(fo.Detection(
                label=label,
                bounding_box=[x_rel, y_rel, w_rel, h_rel],
                confidence=conf,
                fill=True,
                opacity=0.7
            ))

    return fo.Detections(detections=detections)

def visualize_sam_results(image_path, model_path):
    # Load model and run inference
    model = SAM(model_path)
    image_path = os.path.expanduser(image_path)
    results = model(image_path)

    # Create FiftyOne dataset
    dataset = fo.Dataset(name="SAM_Visualization")
    sample = fo.Sample(filepath=image_path)

    # Convert SAM results to FiftyOne format
    sample["predictions"] = sam_masks_to_fo_detections(results)
    dataset.add_sample(sample)

    # Launch interactive visualization
    session = fo.launch_app(dataset)
    session.show()
    session.wait()

if __name__ == "__main__":
    visualize_sam_results(
        image_path="~/Images/ioi.png",
        model_path="/home/jkp/Pictures/sam2/checkpoints/sam2.1_t.pt"
    )