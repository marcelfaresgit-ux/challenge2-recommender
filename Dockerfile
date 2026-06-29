FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY . .
RUN python -m ecommerce_recommender.data.generate \
    && python -m ecommerce_recommender.features.prepare \
    && python -m ecommerce_recommender.models.train \
    && python -m ecommerce_recommender.models.evaluate
EXPOSE 8000
CMD ["sh", "-c", "uvicorn ecommerce_recommender.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
