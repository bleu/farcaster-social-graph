# Use Python 3.12.8 slim as base image
FROM python:3.12.8-slim

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME="/opt/poetry" \
    PATH="/opt/poetry/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    ENVIRONMENT="production"

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Create data directories with appropriate permissions
RUN mkdir -p /app/data/raw /app/data/interim /app/data/checkpoints /app/data/models \
    && chmod -R 777 /app/data

# Copy the entire monorepo
COPY . .

# Then install the API package
WORKDIR /app/farcaster-social-graph-api
RUN poetry install --no-root --no-dev
RUN poetry run prisma generate

# Set the final working directory
WORKDIR /app/farcaster-social-graph-api

# Set the default command
CMD ["poetry", "run", "python", "farcaster_social_graph_api/jobs.py"]