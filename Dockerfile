# 本机演示仍建议直接 uvicorn：镜像会带上 torch，体积很大。
# compose 只负责 Postgres / Redis。
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV ANONYMIZED_TELEMETRY=False

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY data/terms.draft.json ./data/terms.draft.json

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
