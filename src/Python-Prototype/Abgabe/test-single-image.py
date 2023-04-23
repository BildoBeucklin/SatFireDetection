import torch
import cv2
import argparse
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator


class ObjectDetection:
    """
    ObjectDetection class for detecting objects in a given frame or image.

    Attributes:
        capture_index (int, optional): Index of camera to use for capturing frames. Defaults to None.
        image_path (str, optional): Path to image to use for object detection. Defaults to None.
        device (str): Device to use for object detection. Either "cuda" or "cpu".
        model (YOLO): Pretrained YOLOv8 model.
        CLASS_NAMES_DICT (dict): Dictionary of class names and their index in the model.
        box_annotator (BoxAnnotator): Annotator to draw boxes around detected objects.
        labels (list of str): Labels for each detected object.

    """

    def __init__(self, capture_index=None, image_path=None):
        """
        Initializes ObjectDetection instance.

        Args:
            capture_index (int, optional): Index of camera to use for capturing frames. Defaults to None.
            image_path (str, optional): Path to image to use for object detection. Defaults to None.

        """
        self.capture_index = capture_index
        self.image_path = image_path

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
        model = YOLO("best.pt")
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
            con = result.boxes.conf.cpu().numpy()
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

        if len(xyxys) > 0:
            print("Detected Fires:", len(xyxys), "\n")
            for i in range(0, len(xyxys)):
                print("Fire", i + 1)
                print("Confidence: ", "{:.2f}".format(confidences[i][0]))
                print(
                    f"Coordinates: TL:(x={xyxys[i][0][0]}, y={str(xyxys[i][0][1])})\n             BR:(x={xyxys[i][0][2]}, y={str(xyxys[i][0][3])})\n")

        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, confidence, class_id, tracker_id
                       in detections]

        # Annotate and display frame
        frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=self.labels)

        return frame

    def __call__(self):
        """
        Callable method to perform object detection on a given frame. If the `image_path` attribute is set, it will display the
        detection result.

        Returns:
            0
        """
        # Load the frame from the given image path
        frame = cv2.imread(self.image_path)
        # Perform object detection on the frame
        results = self.predict(frame)
        # Plot the bounding boxes on the frame
        frame = self.plot_bboxes(results, frame)
        if args.r:
            # Resize the frame
            frame = cv2.resize(frame, (600, 600))
        # Show the detection result on the screen
        cv2.imshow('YOLOv8 Detection', frame)
        if args.cp:
            cv2.imwrite('output.bmp', frame)
        if args.o:
            cv2.imwrite(args.filename, frame)
        # Wait for a key press event
        cv2.waitKey(0)
        # Close all windows
        cv2.destroyAllWindows()
        # Return 1
        return 1


# Create an argument parser
parser = argparse.ArgumentParser(description='Test image for wildfires')
parser.add_argument('filename', metavar='F', type=str, help="Filename of the image with ending e.g. jpg")
parser.add_argument('-r', action='store_true', help="Resize shown image to 600x600")
parser.add_argument('-cp', action='store_true', help="Save output image as a copy")
parser.add_argument('-o', action='store_true', help="Override input image with output image")
args = parser.parse_args()
# Create an instance of the ObjectDetection class
detector = ObjectDetection(image_path=args.filename)
# Call the instance
detector()
