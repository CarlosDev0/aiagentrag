# Use official Python 3.10 slim
FROM python:3.12-slim as builder

ARG HUGGING_FACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGING_FACE_TOKEN}
# Set workdir
WORKDIR /app

# Copy files
COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

ENV MODEL_DIR=/app/models/gemma-2-2b-it

# Install git and use huggingface_hub to download the model.
#pip install huggingface_hub && \
RUN mkdir -p ${MODEL_DIR} && \
    /bin/bash -c " \
    # 4b. Download the model using the authenticated session.
    python -c \"from huggingface_hub import snapshot_download; \
               snapshot_download(repo_id='google/gemma-2-2b-it', \
                                 local_dir='${MODEL_DIR}', \
                                 local_dir_use_symlinks=False, \
                                 token='$HUGGINGFACE_TOKEN')\" \
    "

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
# Copy the pre-downloaded model from the builder stage
COPY --from=builder ${MODEL_DIR} ${MODEL_DIR}
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PORT=8080
ENV GEMMA_MODEL_PATH=${MODEL_DIR}
EXPOSE 8080

CMD exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT main:app