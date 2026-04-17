import numpy as np
import cv2
from ultralytics import YOLO
from app.services.vision.basePredictor import BasePredictor
from app.config import settings


class YOLOPredictor(BasePredictor):

    def __init__(self):
        self.model = None

    def load(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image
        #resizedImage = cv2.resize(image,(640,640))
        #return resizedImage
    
    def predict(self, image: np.ndarray) -> dict:
        processed = self.preprocess(image)
        results = self.model(processed, conf = settings.MODEL_CONFIDENCE_THRESHOLD)

        detections = []
        for box in results[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            detections.append({
                "label": results[0].names[int(box.cls)],
                "confidence" : float(box.conf),
                "bbox": {"x1":x1, "y1":y1, "x2":x2, "y2":y2}
            })

        return {"detections": detections}