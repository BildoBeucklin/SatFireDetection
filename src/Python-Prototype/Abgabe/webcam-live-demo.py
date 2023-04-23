import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator


class ObjectDetection:
    """
    ObjectDetection class for detecting objects in a given frame.

    Attributes:
        capture_index (int, optional): Index of camera to use for capturing frames.
        device (str): Device to use for object detection. Either "cuda" or "cpu".
        model (YOLO): Pretrained YOLOv8 model.
        CLASS_NAMES_DICT (dict): Dictionary of class names and their index in the model.
        box_annotator (BoxAnnotator): Annotator to draw boxes around detected objects.
        labels (list of str): Labels for each detected object.

    """


    def __init__(self, capture_index):
        """
        Initializes ObjectDetection instance.

        Args:
            capture_index (int): Index of camera to use for capturing frames.

        """
        self.capture_index = capture_index

        # Set device to use for object detection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        # Load YOLO model
        self.model = self.load_model()

        # Set class names dictionary
        self.CLASS_NAMES_DICT = self.model.model.names

        # Initialize box annotator
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)


    def load_model(self):
        """
        Loads pretrained YOLO model.

        Returns:
            YOLO: Loaded YOLO model.

        """
        model = YOLO("best.pt")  # load a pretrained YOLOv8n model
        model.fuse()

        return model


    def predict(self, frame):
        """
        Makes predictions on the given frame.

        Args:
            frame (ndarray): Frame to make predictions on.

        Returns:
            results (list): List of predictions for the frame.

        """
        results = self.model(frame)

        return results


    def plot_bboxes(self, results, frame):
        """
        Plots bounding boxes around detected objects in the frame.

        Args:
            results (list): List of predictions for the frame.
            frame (ndarray): Frame to draw bounding boxes on.

        Returns:
            ndarray: Frame with bounding boxes drawn around detected objects.

        """
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for fire class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id == 0:

                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))


        # Setup detections for visualization
        detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )

        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id in detections]

        # Annotate and display frame
        frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=self.labels)

        return frame



    def __call__(self):
        """
        Callable method to perform object detection on the live feed from the webcam of the device, it will display the
        detection results in a seperate window.

        """

        # Create a VideoCapture object for the camera specified by capture_index
        cap = cv2.VideoCapture(self.capture_index)

        # Check if the video capture is successfully opened
        assert cap.isOpened()

        # Set the desired width of video frames to 1280 pixels
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

        # Set the desired height of video frames to 720 pixels
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


        while True:
            # Measure time to calculate the framerate
            start_time = time()

            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)

            # Show framerate on screen
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            # Show the detection result with a live feed from the webcam on the screen
            cv2.imshow('YOLOv8 Detection', frame)

            # Wait for a key press event
            if cv2.waitKey(5) & 0xFF == 27:

                break

        cap.release()
        # Close all windows
        cv2.destroyAllWindows()



detector = ObjectDetection(capture_index=0)
detector()
