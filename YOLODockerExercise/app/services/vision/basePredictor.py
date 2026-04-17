from abc import ABC, abstractmethod
import numpy as np

class BasePredictor(ABC):
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Loads the model weights into memory"""
        ...

    @abstractmethod
    def predict(self, image: np.ndarray) -> dict:
        """Inference on image, returns raw results"""

    def preprocess(self, image: np.ndarray) -> dict:
        """Preprocess the image before feeding it into the model. Override if needed"""