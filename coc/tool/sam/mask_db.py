import os
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import datetime

# Assuming MaskDatabase is provided as in the query
class MaskDatabase:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)

    def save_masks(self, image_id, masks):
        """Save multiple masks for a specific image"""
        image_dir = self.root_dir / str(image_id)
        image_dir.mkdir(exist_ok=True)
        existing_masks = list(image_dir.glob("*.npy"))
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
        for mask_file in sorted(image_dir.glob("*.npy")):
            masks.append(np.load(str(mask_file)))
        return masks

    def get_image_ids(self):
        """Get list of all image IDs in database"""
        return [d.name for d in self.root_dir.glob("*") if d.is_dir()]

if __name__ == '__main__':
    db = MaskDatabase('/data/mask_db')
    print(db.get_image_ids())
    print(db.get_all_masks(db.get_image_ids()[0]))