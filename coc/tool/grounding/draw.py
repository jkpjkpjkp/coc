"""

"""
from PIL.Image import Image as Img
from typing import List
from coc.tool.grounding.mod import Bbox
import PIL.Image
import PIL.ImageDraw

def draw(image: Img, result: List[Bbox], output_path: str):
    draw = PIL.ImageDraw.Draw(image)
    for x in result:
        draw.rectangle(x['box'], outline="green", width=2)
    image.save(output_path)
    print(f"Image with bounding boxes saved to {output_path}")

if __name__ == '__main__':
    draw(
        image=PIL.Image.open("data/sample/4girls.jpg"),
        result=[Bbox(box=[730.1387939453125, 190.27630615234375, 765.3465576171875, 214.32354736328125], score=0.18632036447525024, label='person'), Bbox(box=[788.237548828125, 0.47208720445632935, 1498.1820068359375, 469.3883972167969], score=0.15282350778579712, label='person'), Bbox(box=[2.399057149887085, 291.7645568847656, 819.9247436523438, 997.913330078125], score=0.12830998003482819, label='person'), Bbox(box=[654.9287719726562, 624.33984375, 1463.01220703125, 998.6312866210938], score=0.12822820246219635, label='person'), Bbox(box=[1389.5648193359375, 166.1981658935547, 1493.7200927734375, 228.39537048339844], score=0.10807681828737259, label='person')],
        output_path="bbox.jpg"
    )