from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "Head CT Classification API"
    DEBUG: bool = False

    XGBOOST_MODEL_PATH: Path = Path("models/xgb_model.ubj")
    FEATURE_EXTRACTOR_PATH: Path = Path("models/feature_extractor.pth")

    class Config:
        env_file = "env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()