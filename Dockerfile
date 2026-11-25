# Image that runs the forecasting pipeline CLI.
# Slim Debian base: Ray and XGBoost ship manylinux wheels, so no compiler is needed.
FROM python:3.12-slim

# No .pyc files in the image; unbuffered output so logs stream in `docker compose`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: OpenMP runtime that XGBoost's CPU build links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Data, MLflow DB/artifacts and Ray results are written under /app/state,
# which docker-compose mounts as a volume so they survive the container.
ENV PF_DATA_PATH=/app/state/data/consumption.csv \
    MLFLOW_TRACKING_URI=sqlite:////app/state/mlflow.db \
    PF_ARTIFACT_ROOT=/app/state/mlartifacts \
    PF_RAY_RESULTS=/app/state/ray_results
WORKDIR /app/state

ENTRYPOINT ["pharma-forecast"]
CMD ["--help"]
