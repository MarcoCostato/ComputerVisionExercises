from functools import lru_cache
from app.config import settings
from app.services.vision.yoloPredictor import YOLOPredictor


@lru_cache
def get_predictor() -> YOLOPredictor:
    predictor = YOLOPredictor()
    predictor.load(str(settings.MODEL_PATH))
    return predictor