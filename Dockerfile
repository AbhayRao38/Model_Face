FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn
COPY . /app
ENV MODEL_PRETRAINED_PATH=/app/pretrained
EXPOSE 7860
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "face_api:app"]
