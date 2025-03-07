# pipeline/analyze_performance.py

import os
import logging
import time  # Added import
from datetime import datetime, timedelta
from prometheus_client import Gauge, generate_latest, Summary, start_http_server
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from fetch_live_data import LiveData, Base  # Import the LiveData model

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

# Prometheus Metrics Definitions for Analysis
AVG_VALIDATION_ACCURACY = Gauge('avg_validation_accuracy_percent', 'Average validation accuracy percentage over the last hour')
TOTAL_PREDICTIONS = Gauge('total_predictions_last_hour', 'Total number of predictions made in the last hour')
TOTAL_VALIDATION_FAILURES = Gauge('total_validation_failures_last_hour', 'Total number of validation failures in the last hour')

# Database Setup with SQLAlchemy
DATA_DIR = 'data'
DATABASE_PATH = os.path.join(DATA_DIR, 'pipeline_data.db')
engine = create_engine(f'sqlite:///{DATABASE_PATH}')
Session = sessionmaker(bind=engine)
session = Session()

def calculate_metrics():
    """Calculate performance metrics from the database."""
    try:
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        # Total Predictions in the Last Hour
        total_predictions = session.query(func.count(LiveData.id)).filter(LiveData.timestamp >= one_hour_ago).scalar()
        TOTAL_PREDICTIONS.set(total_predictions if total_predictions else 0)
        logging.info(f"Total Predictions in the Last Hour: {total_predictions}")
        
        # Average Validation Accuracy in the Last Hour
        accuracies = session.query(LiveData.prediction, LiveData.actual_temperature).filter(LiveData.timestamp >= one_hour_ago).all()
        
        if accuracies:
            # Filter out invalid data (None values)
            valid_accuracies = [(pred, actual) for pred, actual in accuracies if pred is not None and actual is not None]
            differences = [abs(pred - actual) for pred, actual in valid_accuracies]
            threshold = 2.0  # degrees Celsius
            accurate_predictions = [diff for diff in differences if diff <= threshold]
            
            if differences:
                avg_accuracy = (len(accurate_predictions) / len(differences)) * 100
                AVG_VALIDATION_ACCURACY.set(avg_accuracy)
                logging.info(f"Average Validation Accuracy in the Last Hour: {avg_accuracy:.2f}%")
            else:
                AVG_VALIDATION_ACCURACY.set(0)
                logging.info("No valid predictions to calculate average accuracy.")
        else:
            AVG_VALIDATION_ACCURACY.set(0)
            logging.info("No predictions to calculate average accuracy.")
        
        # Total Validation Failures in the Last Hour
        total_failures = session.query(func.count(LiveData.id)).filter(
            LiveData.timestamp >= one_hour_ago,
            LiveData.prediction.isnot(None),  # Ensure prediction is not None
            LiveData.actual_temperature.isnot(None),  # Ensure actual_temperature is not None
            func.abs(LiveData.prediction - LiveData.actual_temperature) > 2.0  # Use func.abs for SQL
        ).scalar()
        TOTAL_VALIDATION_FAILURES.set(total_failures if total_failures else 0)
        logging.info(f"Total Validation Failures in the Last Hour: {total_failures}")
        
    except Exception as e:
        logging.exception(f"Error calculating metrics: {e}")

def main():
    """Main function to calculate and expose metrics."""
    # Start Prometheus metrics server on port 8002
    start_http_server(8002)
    logging.info("Prometheus metrics server for analyzer started on port 8002.")
    
    while True:
        calculate_metrics()
        time.sleep(3600)  # Calculate metrics every hour

if __name__ == '__main__':
    # Run the main data fetching loop
    main()
