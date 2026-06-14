FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    "clearml>=1.16,<3" \
    "clearml-serving>=1.3,<2" \
    fastapi \
    uvicorn \
    grpcio \
    numpy \
    scikit-learn \
    joblib

EXPOSE 8080

CMD ["uvicorn", "clearml_serving.serving.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]