"""prompt used by paper 'Visual Sketchpad'

which is a 24.06 nips paper highly related. 
"""

initial_prompt = """Here are some tools that can help you. All are python codes. They are in tools.py and will be imported for you.
The images has their own coordinate system. The upper left corner of the image is the origin (0, 0). All coordinates are normalized, i.e., the range is [0, 1].
All bounding boxes are in the format of [x, y, w, h], which is a python list. x is the horizontal coordinate of the upper-left corner of the box, y is the vertical coordinate of that corner, w is the box width, and h is the box height.
Notice that you, as an AI assistant, is not good at locating things and describe them with coordinate. You can use tools to generate bounding boxes.
You can zoom in on the image to see the details. Below are the tools in tools.py:
```python

def detection(image, objects):
    \"\"\"Object detection using Grounding DINO model. It returns the annotated image and the bounding boxes of the detected objects.
    The text can be simple noun, or simple phrase (e.g., 'bus', 'red car'). Cannot be too hard or the model will break.
    The detector is not perfect, it may wrongly detect objects or miss some objects.
    Also, notice that the bounding box label might be out of the image boundary.
    You should use the output as a reference, not as a ground truth.
    When answering questions about the image, you should double-check the detected objects.
```python


```
output:


    Args:
        image (PIL.Image.Image): the input image
        objects (List[str]): a list of objects to detect. Each object should be a simple noun or a simple phrase. Should not be hard or abstract concepts like "text" or "number".

    Returns:
        output_image (AnnotatedImage): the original image, annotated with bounding boxes. Each box is labeled with the detected object, and an index.
        processed boxes (List): listthe bounding boxes of the detected objects

    Example:
        image = Image.open("sample_img.jpg")
        output_image, boxes = detection(image, ["bus"])
        display(output_image.annotated_image)
        print(boxes) # [[0.24, 0.21, 0.3, 0.4], [0.6, 0.3, 0.2, 0.3]]
        # you need to double-check the detected objects. Some objects may be missed or wrongly detected.
    \"\"\"

def segment_and_mark(image, anno_mode:list = ['Mask', 'Mark']):
    \"\"\"Use a segmentation model to segment the image, and add colorful masks on the segmented objects. Each segment is also labeled with a number.
    The annotated image is returned along with the bounding boxes of the segmented objects.
    This tool may help you to better reason about the relationship between objects, which can be useful for spatial reasoning etc.
    DO NOT use this tool to search or detect an object. It is likely the object is small and segmentaiton does not help.
    Segmentation and marking can also be helpful for 3D and video reasoning. For example, helping you to see more clearly and analyzes the relationship between different frames of a video.

    Args:
        image (PIL.Image.Image): the input image
        anno_mode (list, optional): What annotation is added on the input image. Mask is the colorful masks. And mark is the number labels. Defaults to ['Mask', 'Mark'].

    Returns:
        output_image (AnnotatedImage): the original image annotated with colorful masks and number labels. Each mask is labeled with a number. The number label starts at 1.
        bboxes (List): listthe bounding boxes of the masks.The order of the boxes is the same as the order of the number labels.

    Example:
        User request: I want to find a seat close to windows, where should I sit?
        Code:
        ```python
        image = Image.open("sample_img.jpg")
        output_image, bboxes = segment_and_mark(image)
        display(output_image.annotated_image)
        ```
        Model reply: You can sit on the chair numbered as 5, which is close to the window.
        User: Give me the bounding box of that chair.
        Code:
        ```python
        print(bboxes[4]) # [0.24, 0.21, 0.3, 0.4]
        ```
        Model reply: The bounding box of the chair numbered as 5 is [0.24, 0.21, 0.3, 0.4].
    \"\"\"

def depth(image):
    \"\"\"Depth estimation using DepthAnything model. It returns the depth map of the input image.
    A colormap is used to represent the depth. It uses Inferno colormap. The closer the object, the warmer the color.
    This tool may help you to better reason about the spatial relationship, like which object is closer to the camera.

    Args:
        image (PIL.Image.Image): the input image

    Returns:
        output_image (PIL.Image.Image): the depth map of the input image

    Example:
        image = Image.open("sample_img.jpg")
        output_image = depth(image)
        display(output_image)
    \"\"\"
```
# GOAL #: Based on the above tools, I want you to reason about how to solve the # USER REQUEST # and generate the actions step by step (each action is a python jupyter notebook code block) to solve the request.
You may need to use the tools above to process the images and make decisions based on the visual outputs of the previous code blocks.
Your visual ability is not perfect, so you should use these tools to assist you in reasoning about the images.

The jupyter notebook has already executed the following code to import the necessary packages:
```python
from PIL import Image
from IPython.display import display
from tools import *
```

# REQUIREMENTS #:
1. The generated actions can resolve the given user request # USER REQUEST # perfectly. The user request is reasonable and can be solved. Try your best to solve the request.
2. The arguments of a tool must be the same number, modality, and format specified in # TOOL LIST #;
3. If you think you got the answer, use ANSWER: <your answer> to provide the answer, and ends with TERMINATE.
4. All images in the initial user request are stored in PIL Image objects named image_1, image_2, ..., image_n. You can use these images in your code blocks. Use display() function to show the image in the notebook for you too see.
7. When detection tool failed to detect an object, you can use the sliding_window_detection tool to search for the object.
8. Bounding boxes may be wrong and misled you. When answering questions about small objects in bounding boxes, you should zoom in on the object to see the details.
9. Segmentation and marking tool can help you better reason about the relationship between objects. Example use cases are spatial reasoning (e.g., left/right/above/below/on) and counting.


Below are some examples of how to use the tools to solve the user requests. You can refer to them for help. You can also refer to the tool descriptions for more information.
# EXAMPLE: Answering a simple question about the image
# USER REQUEST #: <A image here> which city is this?
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: The tools are not helpful for this task. I can directly answer the question.
ACTION 0: No action needed.
ANSWER: This looks like New York City. I can see the Empire State Building in the image. TERMINATE


# EXAMPLE: Detecting objects in the image
# USER REQUEST #: <A image here> is there a dog in the image? Options: (a) Yes (b) No
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: Firstly, I need to detect dogs in the image.
ACTION 0:
```python
image_1_dog_detection, image_1_dog_boxes = detection(image_1, ["dog"])
display(image_1_dog_detection.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: A dog is detected by the tool. After double-checking, I can confirm that the tool is correct and there is a dog in the image.
ACTION 1: No action needed.
ANSWER: Yes, there is a dog in the image. So the answer is (a). TERMINATE


# EXAMPLE: Zooming in to answer the question about a small object
# USER REQUEST #: <A image here> What is the color of the busket? Options: (a) white (b) black (c) red
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: To identify the color of the bucket in the image, I need to locate the bucket first. Since the bucket might be small and not easily detectable, I will use the detection tool to find it.
ACTION 0:
```python
image_1_bucket_detection, image_1_bucket_boxes = detection(image_1, ["bucket"])
display(image_1_bucket_detection.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
If you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: The bucket has been detected and is labeled as "bucket 1" in the image. I will zoom in on the bucket to identify its color.
ACTION 1:
```python
bucket_box = image_1_bucket_boxes[0]
# zoom-in to bucket_box (with padding for context), code omitted.
display(zoomed_bucket)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 2: From the zoomed-in image, I can see that the color of the bucket is white.
ACTION 2: No action needed.
ANSWER: The color of the bucket is white. So the answer is (a). TERMINATE


# EXAMPLE: Searching for small objects in the image
# USER REQUEST #: <A image here> What animal is drawn on that red signicade?
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: I do not see a sign that is both red and has an animal drawn on it. I need to use the sliding_window_detection tool to find that sign and zoom in on it.
ACTION 0:
```python
possible_patches = sliding_window_detection(image_1, ["animal", "sign"])
for i, patch in enumerate(possible_patches):
    print(f"Patch {i+1}:")
    display(patch.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
Patch 1:
<image output 1>
Patch 2:
<image output 2>
<More patches below...>
THOUGHT 1: Among the above patches, Patch 4 contains the animal drawn on the red signicade. An animal was detected. I will zoom in to double check if the detection is correct.
ACTION 1:
```python
relevant_patch = possible_patches[3]
display(relevant_patch.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 2: I can confirm that there is an animal drawn on the red signicade. I will zoom in to see what animal it is.
ACTION 2:
```python
animal_box = relevant_patch['box']es[2]
zoomed_patch = zoom_in_image_by_bbox(relevant_patch.original_image, animal_box)
display(zoomed_patch)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 3: It can be seen that the animal drawn on the red signicade is a tiger.
ACTION 3: No action needed.
ANSWER: The animal drawn on the red signicade is a tiger. TERMINATE


# EXAMPLE: Reasoning about the depth of the objects
# USER REQUEST #: <A image here> Which point is closer to the camera? Options: (A) Point A (B) Point B
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: I can use the depth estimation tool. This tool will provide a depth map where different colors represent different distances from the camera.
I can also use the overlay_images tool to overlay the original image on the heat map to get a better sense of where point A and point B are on the depth map.
ACTION 0:
```python
image_1_depth = depth(image_1)
overlayed_image_1 = overlay_images(image_1_depth, image_1, alpha=0.3)
display(overlayed_image_1)

# depth can be used together with segmentation and marking tool
image_1_segmented, image_1_boxes = segment_and_mark(image_1)
overlayed_image_2 = overlay_images(image_1_depth, image_1_segmented.annotated_image, alpha=0.5)
display(overlayed_image_2)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 1: The tool is helpful. From the overlayed image, I can see that the color in the circle under point B is warmer than that of point A, which means point B is closer to the camera.
ACTION 1: No action needed.
ANSWER: Based on the depth map, the circle under point B, which shows warmer colors, is closer to the camera than point A. So the answer is (B). TERMINATE


# EXAMPLE: Reasoning about the spatial relationship between objects
# USER REQUEST #: <A image here> What is on the left of the right laptop? Options:
(A) Lamp
(B) Chair
(C) Plant
(D) None of the above
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: To help me better reason about the relationship between objects, I can use the segmentation and marking tool.
ACTION 0:
```python
image_1_segmented, image_1_boxes = segment_and_mark(image_1)
display(image_1_segmented.annotated_image)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 1: The tool is helpful. I can see the right laptop is labeled as 9. There is a lamp on its left, which is labeled as 12.
ACTION 1: No action needed.
ANSWER: On the left side of the right laptop, there is a lamp. So the answer is (A). TERMINATE


# EXAMPLE: Doing Jigsaw Puzzle
# USER REQUEST #: <An image here> <An image here> <An image here> The lower right part of the first image is black. The second and the third images might fit into the black part. Can you check which one fits?
# USER IMAGE stored in image_1, image_2, image_3 as PIL image.
# RESULT #:
THOUGHT 0: To check if the image fits, I can use the overlay_images tool to overlay the second and third image on the first image. I will overlay them on the black part to see if they fit.
ACTION 0:
```python
image_1_overlayed_with_image_2 = overlay_images(image_1, image_2, alpha=1.0, bounding_box=[0.5, 0.5, 0.5, 0.5])
image_1_overlayed_with_image_3 = overlay_images(image_1, image_3, alpha=1.0, bounding_box=[0.5, 0.5, 0.5, 0.5])
display(image_1_overlayed_with_image_2)
display(image_1_overlayed_with_image_3)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
THOUGHT 1: Comparing two images, I can see that the third image fits into the black part. The second image does not fit. You can see a building in the second image that is not in the first image.
ACTION 1: No action needed.
ANSWER: The third image fits into the black part. TERMINATE


"""