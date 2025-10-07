
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fluidsynth && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY vocalchemy_tools.py ./vocalchemy_tools.py
COPY server.py ./server.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["bash","-lc","uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
