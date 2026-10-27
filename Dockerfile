FROM python:3.11-slim

# Install system dependencies required for OpenCV, Audio, and ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libasound2 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV PORT=5000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
