from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.v1.router import router as v1_router
from app.core.dependencies import get_xgb_model, get_feature_extractor

async def lifespan(app: FastAPI):
    # Load the XGBoost model and feature extractor at startup
    xgb_model = get_xgb_model()
    feature_extractor = get_feature_extractor()
    
    # Store them in the app state for later use
    app.state.xgb_model = xgb_model
    app.state.feature_extractor = feature_extractor
    
    yield

app = FastAPI(  title="Head CT Classification API",
                version="1.0",
                lifespan=lifespan) 

app.include_router(v1_router, prefix="/api/v1")