from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.schemas.prediction import PredictionResponse
from app.core.dependencies import get_predictor
from app.services.vision.basePredictor import BasePredictor
import numpy as np
import cv2

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    predictor: BasePredictor = Depends(get_predictor)
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported Format. Use JPEG or PNG")
    
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="The image was not decoded correctly.")
    
    result = predictor.predict(image)
    return PredictionResponse(**result)