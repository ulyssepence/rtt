FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/src/ src/
COPY frontend/static/ static/
RUN npm run build

FROM python:3.12-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/rtt/ src/rtt/
COPY frontend/index.html frontend/index.html
COPY --from=frontend /build/static/ frontend/static/
ENV RTT_OLLAMA_URL=http://host.docker.internal:11434
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
ENTRYPOINT ["uv", "run", "rtt", "serve"]
