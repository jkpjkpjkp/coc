"""visualize bboxes on the image - interactively.

"""
import fiftyone as fo
from PIL import Image
from coc.tool.grounding import Bbox

def xyxy_to_rel_midpoint(box, img_width, img_height):
    x1, y1, x2, y2 = box
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return [x1 / img_width, y1 / img_height, w, h]

def envision(img_path: str, bbox_list):
    print(bbox_list)

    img = Image.open(img_path)
    img_width, img_height = img.size

    # Convert your Bbox objects (assumes box=[x1,y1,x2,y2] in pixel coordinates)
    def convert_bboxes(bbox_list):
        detections = []
        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox['box']
            normalized_box = xyxy_to_rel_midpoint(bbox['box'], img_width, img_height)
            detections.append(fo.Detection(
                label=str(bbox['label']),  # Labels must be strings
                bounding_box=normalized_box,
                confidence=float(bbox['score']),
                # Force visual properties
                fill=True,          # <<< Critical for visibility
                opacity=0.7         # <<< Avoid 0% transparency
            ))
        return fo.Detections(detections=detections)

    # Create dataset
    dataset = fo.Dataset()
    sample = fo.Sample(filepath=img_path)
    # bbox_list = [Bbox(box=[730.1387939453125, 190.27630615234375, 765.3465576171875, 214.32354736328125], score=0.18632036447525024, label='person'), Bbox(box=[788.237548828125, 0.47208720445632935, 1498.1820068359375, 469.3883972167969], score=0.15282350778579712, label='person'), Bbox(box=[2.399057149887085, 291.7645568847656, 819.9247436523438, 997.913330078125], score=0.12830998003482819, label='person'), Bbox(box=[654.9287719726562, 624.33984375, 1463.01220703125, 998.6312866210938], score=0.12822820246219635, label='person'), Bbox(box=[1389.5648193359375, 166.1981658935547, 1493.7200927734375, 228.39537048339844], score=0.10807681828737259, label='person')]

    sample["my_detections"] = convert_bboxes(bbox_list)  # Use a clearer field name
    dataset.add_sample(sample)

    # Launch and verify
    session = fo.launch_app(dataset)
    session.show()  # Open browser if not auto-launched
    session.wait()

if __name__ == '__main__':
    envision('/home/jkp/hack/coc/data/sample/4girls.jpg')