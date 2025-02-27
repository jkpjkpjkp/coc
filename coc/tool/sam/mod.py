

from ultralytics import SAM

def get_sam():
    return SAM("/home/jkp/Pictures/sam2/checkpoints/sam2.1_l.pt")

if __name__ == '__main__':

    # Load a model
    model = get_sam()

    # Display model information (optional)
    model.info()

    # Run inference with bboxes prompt
    results = model('~/Images/ioi.png')