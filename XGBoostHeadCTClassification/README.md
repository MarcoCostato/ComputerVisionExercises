# XGBoost Head CT Classification

A FastAPI-based service for classifying head CT scans using a ResNet feature extractor and an XGBoost classifier.

## Project Overview

- `app/` contains the API implementation, dependency wiring, and model loading logic.
- `models/` stores the trained artifacts: `feature_extractor.pth` and `xgb_model.ubj`.
- `training.ipynb` contains the feature extraction and XGBoost training pipeline.
- `environment.yml` defines the Conda environment used by the project.
- `Dockerfile` and `docker-compose.yml` provide containerized deployment.

## API

The app exposes a single prediction endpoint:

- `POST /api/v1/`
  - accepts `image/jpeg` and `image/png`
  - returns JSON with the `prediction` field

Example response:

```json
{
  "prediction": 0
}
```

## Local Setup

1. Create the Conda environment:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate XGBoostHeadCTClassification
```

3. Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. Send a request to the endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/" -F "file=@path/to/image.png"
```

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## Notes

- The predictor uses a ResNet-based feature extractor to generate image features.
- The XGBoost model is loaded from `models/xgb_model.ubj`.
- The feature extractor weights are loaded from `models/feature_extractor.pth`.

## Files of Interest

- `app/main.py` — FastAPI application startup and router registration
- `app/api/v1/endpoints/predict.py` — image upload and prediction endpoint
- `app/core/dependencies.py` — model loading and dependency provider
- `training.ipynb` — dataset preparation, feature extraction, and XGBoost training
