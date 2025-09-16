# -------------------------------
# 1. Base Image
# -------------------------------
FROM python:3.11-slim

# -------------------------------
# 2. Set environment variables
# -------------------------------
# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------------
# 3. Set work directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 4. Install system dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 5. Copy requirements and install
# -------------------------------
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 6. Copy the app source code
# -------------------------------
COPY . .

# -------------------------------
# 7. Expose port
# -------------------------------
EXPOSE 8080

# -------------------------------
# 8. Define entrypoint
# -------------------------------
# Run FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
