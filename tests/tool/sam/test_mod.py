from gradio_client import Client, handle_file
import unittest

class testServer(unittest.TestCase):
    def test_server(self):
        client = Client("http://127.0.0.1:7045/")
        result = client.predict(
                image=handle_file('data/sample/onions.jpg'),
                points_per_side=8,
                points_per_batch=8,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.95,
                stability_score_offset=1,
                mask_threshold=0,
                box_nms_thresh=0.7,
                crop_n_layers=0,
                crop_nms_thresh=0.7,
                crop_overlap_ratio=0.3413333333333333,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=0,
                use_m2m=False,
                multimask_output=True,
                api_name="/predict"
        )
        self.assertEqual(set(result.keys()), {'iou_preds', 'points', 'low_res_masks', 'stability_score', 'boxes', 'rles', 'crop_boxes'})