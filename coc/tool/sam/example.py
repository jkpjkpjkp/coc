import fiftyone as fo
import numpy as np
from ultralytics import SAM
import os

# Load model and run inference
model = SAM("/home/jkp/Pictures/sam2/checkpoints/sam2.1_b.pt")
image_path = os.path.expanduser("~/Images/ioi.png")  # Expand ~ to full path
results = model(image_path)

# Create FiftyOne dataset
dataset = fo.Dataset(name="SAM_Results")
sample = fo.Sample(filepath=image_path)

# Process results
detections = []
for result in results:
    if result.masks is not None:
        # Get original image dimensions
        height, width = result.orig_shape

        # Convert each mask to FiftyOne Detection
        for mask in result.masks:
            # Get binary mask array
            mask_array = mask.data[0].cpu().numpy()

            # Create FiftyOne Detection with mask
            detections.append(
                fo.Detection(
                    label="object",
                    mask=mask_array,
                    confidence=mask.conf if hasattr(mask, "conf") else None
                )
            )

# Add detections to sample
sample["predictions"] = fo.Detections(detections=detections)
dataset.add_sample(sample)

# Launch interactive app
session = fo.launch_app(dataset)
session.wait()