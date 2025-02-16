

from ultralytics import SAM

# Load a model
model = SAM("/home/jkp/Pictures/sam2/checkpoints/sam2.1_l.pt")

# Display model information (optional)
model.info()

# Run inference with bboxes prompt
results = model("data/sample/onions.jpg")
