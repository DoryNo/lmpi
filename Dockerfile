FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config.yaml ./config.yaml

RUN useradd --system --create-home lmpi
USER lmpi

EXPOSE 8080

# Respect LMPI_HOST / LMPI_PORT env vars with sane defaults.
CMD ["sh", "-c", "exec uvicorn src.main:app --host ${LMPI_HOST:-0.0.0.0} --port ${LMPI_PORT:-8080}"]
