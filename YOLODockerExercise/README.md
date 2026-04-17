# YOLO Docker Exercise

Simple FastAPI-based object detection service using Ultralytics YOLO and Docker.

## Overview

This project exposes a REST endpoint for image-based object detection. It loads a YOLOv8 model from `models/yolov8n.pt` and returns detected object labels, confidence scores, and bounding boxes.

## Features

- FastAPI backend
- YOLOv8 object detection via `ultralytics`
- Docker container support
- Single prediction endpoint with JPEG/PNG upload

## Project Structure

- `app/main.py` - FastAPI application entry point
- `app/api/v1/endpoints/predict.py` - prediction endpoint
- `app/services/vision/yoloPredictor.py` - YOLO model wrapper
- `app/config.py` - app settings and model path
- `Dockerfile` - container build definition
- `docker-compose.yml` - service configuration
- `models/yolov8n.pt` - default YOLO model weights

## Requirements

- Docker and Docker Compose, or
- Python environment with dependencies from `requirements.txt`

## Run with Docker

1. Build and start the service:

```bash
docker-compose up --build
```

2. The API will be available at:

```text
http://localhost:8000/api/v1/
```

3. Open interactive docs at:

```text
http://localhost:8000/docs
```

## API Usage

Send a `POST` request to `/api/v1/` with an image file field named `file`.

Example using `curl`:

```bash
curl -X POST "http://localhost:8000/api/v1/" \
  -F "file=@path/to/image.jpg" \
  -H "Content-Type: multipart/form-data"
```

Response format:

```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.87,
      "bbox": {"x1": 100.0, "y1": 50.0, "x2": 200.0, "y2": 300.0}
    }
  ]
}
```

## Configuration

Default model path and confidence threshold are set in `app/config.py`:

- `models/yolov8n.pt`
- `MODEL_CONFIDENCE_THRESHOLD = 0.5`

## Notes

- Supported upload content types: `image/jpeg`, `image/png`
- The model loads once at startup via the FastAPI lifespan hook

