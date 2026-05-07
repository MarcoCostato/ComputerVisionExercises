from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.schemas.prediction import PredictionResponse
from app.core.dependencies import get_xgb_model, get_feature_extractor
from app.services.vision.baseFeatureExtractor import BaseFeatureExtractor
from app.services.vision.baseClassifier import BaseClassifier
import numpy as np
import cv2

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    feature_extractor: BaseFeatureExtractor = Depends(get_feature_extractor),
    xgb_model: BaseClassifier = Depends(get_xgb_model)
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")
    
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not read the image file.")
    features = feature_extractor.extract_features(image)
    prediction = xgb_model.predict(features.reshape(1, -1))[0]
    return PredictionResponse(prediction=prediction)