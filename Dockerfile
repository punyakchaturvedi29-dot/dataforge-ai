# Single-process deployment: FastAPI serves the API and the built frontend.
# docker build -t dataforge . && docker run -p 8000:8000 -v dataforge-data:/data dataforge

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    DATAFORGE_DATA_DIR=/data FRONTEND_DIST=/app/frontend/dist
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY sample_data.csv /app/sample_data.csv
COPY --from=web /web/dist /app/frontend/dist
RUN useradd --create-home --uid 10001 dataforge && mkdir -p /data && chown dataforge /data
USER dataforge
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
