from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.v1.router import router as v1_router
from app.core.dependencies import get_predictor

async def lifespan(app: FastAPI):
    get_predictor()
    yield


app = FastAPI(
    title="VisionAPI",
    version="1.0.0",
    lifespan=lifespan
    )   

app.include_router(v1_router, prefix="/api/v1")