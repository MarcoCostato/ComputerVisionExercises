import torchvision
from torchvision.models import resnet18
from torch import nn
import torch
import numpy as np

from app.services.vision.baseFeatureExtractor import BaseFeatureExtractor


def convert_to_3channels(x):
    """Convert image to 3-channel RGB"""
    if x.shape[0] == 1:
        # Grayscale to RGB
        return x.repeat(3, 1, 1)
    elif x.shape[0] == 4:
        # RGBA to RGB (drop alpha channel)
        return x[:3, :, :]
    else:
        return x


class ResNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        resnet = resnet18()
        feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.model = feature_extractor

    def load(self, model_path: str) -> None:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ConvertImageDtype(torch.float),
            torchvision.transforms.Lambda(convert_to_3channels),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            preprocessed = self.preprocess(image).unsqueeze(0)  # Add batch dimension
            features = self.model(preprocessed)
            return features.squeeze().numpy()

        