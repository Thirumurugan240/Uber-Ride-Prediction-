# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY . .

# Ensure a fresh, version-compatible model exists in the image.
RUN python -m src.train

EXPOSE 5000

# Serve with gunicorn (the app factory exposes `app` in app.py).
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
