# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.1
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the application files
COPY . .

# Copy the .env file
COPY .env .env

# Expose the port
EXPOSE 8000

# Set environment variables from the .env file
# The following line reads the .env file and exports each variable
CMD ["/bin/bash", "-c", "set -o allexport && source .env && set +o allexport && uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"]

