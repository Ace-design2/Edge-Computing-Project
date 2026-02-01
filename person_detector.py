from ultralytics import YOLO
import cv2

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("Model loaded.")

    def detect_people(self, frame, conf_threshold=0.5):
        """
        Runs inference on the frame and returns bounding boxes for detected people.
        Returns: List of (x1, y1, x2, y2) tuples.
        """
        results = self.model(frame, classes=[0], verbose=False) # class 0 is person in COCO
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] > conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append((int(x1), int(y1), int(x2), int(y2)))
        
        return detections
