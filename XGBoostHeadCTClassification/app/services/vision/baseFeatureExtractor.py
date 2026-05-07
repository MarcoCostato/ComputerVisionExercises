from abc import ABC, abstractmethod
import numpy as np

class BaseFeatureExtractor(ABC):
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load the feature extractor model from the specified path."""
        ...

    @abstractmethod
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from the input image and return them as a numpy array."""
        ...

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the input image before feature extraction."""
        ...