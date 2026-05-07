FROM python:3.10-slim

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY JPGProcessingModule ./JPGProcessingModule
COPY PDFProcessingModule ./PDFProcessingModule
COPY tessdata ./tessdata
COPY worker.py .
COPY settings.py .

CMD ["celery", "-A", "worker:celery", "worker", "--loglevel=info", "-c", "1"]