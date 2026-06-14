FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir gradio requests

COPY ui/app.py /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]