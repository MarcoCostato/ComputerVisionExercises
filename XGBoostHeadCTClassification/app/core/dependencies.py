from functools import lru_cache
from app.config import settings
from app.services.vision.ResNetFeatureExtractor import ResNetFeatureExtractor
from app.services.vision.XGBoostClassifier import XGBoostClassifier

@lru_cache
def get_feature_extractor() -> ResNetFeatureExtractor:
    feature_extractor = ResNetFeatureExtractor()
    feature_extractor.load(str(settings.FEATURE_EXTRACTOR_PATH))
    return feature_extractor

def get_xgb_model() -> XGBoostClassifier:
    xgb_model = XGBoostClassifier()
    xgb_model.load(str(settings.XGBOOST_MODEL_PATH))
    return xgb_model