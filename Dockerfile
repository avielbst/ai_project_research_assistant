FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

ENV CMAKE_ARGS="-DLLAMA_NATIVE=ON"
ENV FORCE_CMAKE=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
