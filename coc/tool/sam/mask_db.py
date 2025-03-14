import os
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import datetime
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Assuming MaskDatabase is provided as in the query
class MaskDatabase:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)

    def save_masks(self, image_id, masks):
        """Save multiple masks for a specific image"""
        image_dir = self.root_dir / str(image_id)
        image_dir.mkdir(exist_ok=True)
        existing_masks = list(image_dir.glob("*.pkl"))
        start_idx = len(existing_masks)
        for i, mask in enumerate(masks):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            filename = image_dir / f"mask_{start_idx + i}_{timestamp}"
            masks.save(str(filename), mask)

    def get_all_masks(self, image_id):
        """Retrieve all masks for a specific image"""
        image_dir = self.root_dir / str(image_id)
        if not image_dir.exists():
            return []
        masks = []
        for mask_file in sorted(image_dir.glob("*.pkl")):
            masks.append(np.load(str(mask_file), allow_pickle=True))
        print(len(masks))
        return masks

    def get_image_ids(self):
        """Get list of all image IDs in database"""
        return [d.name for d in self.root_dir.glob("*") if d.is_dir()]

    def get_list_of_masks(self, image_id):
        """Get list of all mask files for a specific image"""
        masks = self.get_all_masks(image_id)
        print(len(masks))
        return SAM2AutomaticMaskGenerator._my_postprocess_masks(masks[0])

if __name__ == '__main__':
    db = MaskDatabase('/data/mask_db')
    print(db.get_image_ids())
    print(db.get_all_masks(db.get_image_ids()[0]))