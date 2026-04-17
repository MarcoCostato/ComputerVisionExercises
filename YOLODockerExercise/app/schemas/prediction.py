from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: float
    x2: float
    y1: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox

class PredictionResponse(BaseModel):
    detections: list[Detection]