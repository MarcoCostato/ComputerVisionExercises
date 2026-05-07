from abc import ABC, abstractmethod
import numpy as np

class BaseClassifier(ABC):
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load the classifier model from the specified path."""
        ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels for the given features and return them as a numpy array."""
        ...