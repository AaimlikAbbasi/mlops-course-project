# pipeline/app.py

from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import random

# Metrics Definitions
DATA_INGESTED = Counter('data_ingested_total', 'Total data ingested')
PREDICTIONS = Counter('model_predictions_total', 'Total number of model predictions')
PREDICTION_ERRORS = Counter('model_prediction_errors_total', 'Total prediction errors')
API_REQUESTS_IN_PROGRESS = Gauge('api_requests_in_progress', 'Number of API requests in progress')
REQUEST_LATENCY = Summary('api_request_latency_seconds', 'Latency of API requests')

def ingest_data():
    """Simulate data ingestion."""
    DATA_INGESTED.inc()
    time.sleep(0.5)  # Simulate ingestion delay

def make_prediction():
    """Simulate making a prediction."""
    API_REQUESTS_IN_PROGRESS.inc()
    with REQUEST_LATENCY.time():
        time.sleep(1)  # Simulate prediction processing time
        if random.random() < 0.1:  # 10% chance of prediction error
            PREDICTION_ERRORS.inc()
        PREDICTIONS.inc()
    API_REQUESTS_IN_PROGRESS.dec()

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(8000)
    # Simulate pipeline operations
    while True:
        ingest_data()
        make_prediction()
