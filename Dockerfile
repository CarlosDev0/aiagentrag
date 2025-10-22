# Use official Python 3.10 slim
FROM python:3.12-slim as builder

# Set workdir
WORKDIR /app

# Copy files
COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt
# Upgrade pip
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PORT=8080
EXPOSE 8080

CMD exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT main:app