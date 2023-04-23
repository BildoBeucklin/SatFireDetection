# Object Detection using YOLOv8

This code implements an object detection system using the YOLOv8 architecture. It uses torch for running the model and cv2 for reading and visualizing frames. It also uses argparse for parsing command-line arguments, ultralytics for loading the pretrained YOLOv8 model, and supervision for drawing and annotating the detected objects.
## Requirements

- torch >= 1.7.0
- cv2 >= 4.4.0
- argparse >= 1.4.0
- ultralytics >= 0.3.3
- supervision >= 0.2.2


## Class ObjectDetection

The ObjectDetection class contains the following attributes and methods:
### Attributes:

- capture_index (int, optional): Index of camera to use for capturing frames. Defaults to None.
- image_path (str, optional): Path to image to use for object detection. Defaults to None.
- device (str): Device to use for object detection. Either "cuda" or "cpu".
- model (YOLO): Pretrained YOLOv8 model.
- CLASS_NAMES_DICT (dict): Dictionary of class names and their index in the model.
- box_annotator (BoxAnnotator): Annotator to draw boxes around detected objects.
- labels (list of str): Labels for each detected object.

### Methods:

- __init__(capture_index=None, image_path=None): Initializes ObjectDetection instance.
- load_model(): Loads pretrained YOLO model.
- predict(frame): Makes predictions on the given frame.
- plot_bboxes(results, frame): Plots bounding boxes around detected objects in the frame.
