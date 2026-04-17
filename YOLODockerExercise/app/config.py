from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "Vision API"
    DEBUG: bool = False
    
    MODEL_PATH: Path = Path("models/yolov8n.pt")
    MODEL_CONFIDENCE_THRESHOLD: float = 0.5

    class Config:
        env_file = "env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
    