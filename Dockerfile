FROM python:3.11-slim

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY src/    ./src/
COPY main.py .

ENV NANO_IP=""
ENV AUTH_TOKEN=""

CMD ["python", "main.py"]
