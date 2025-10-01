# Use official Python 3.10 slim
FROM python:3.10-slim as builder

# Set workdir
WORKDIR /app

# Copy files
COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt
# Upgrade pip
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]
