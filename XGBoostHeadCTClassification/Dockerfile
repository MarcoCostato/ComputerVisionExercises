FROM continuumio/miniconda3:latest
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

COPY app/ ./app
COPY models/ ./models

EXPOSE 8000

CMD ["conda", "run", "-n", "XGBoostHeadCTClassification",\
    "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]   