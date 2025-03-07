# pipeline/fetch_live_data.py

import os
import requests
import json
import logging
import time
from datetime import datetime
from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    start_http_server
)
from flask import Flask, Response
import joblib
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Initialize Flask App for Prometheus Metrics
app = Flask(__name__)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

# Securely load API keys from environment variables
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
AIRVISUAL_API_KEY = os.getenv('AIRVISUAL_API_KEY')

# Define API Endpoints
OPENWEATHERMAP_URL = 'http://api.openweathermap.org/data/2.5/weather'
AIRVISUAL_URL = 'http://api.airvisual.com/v2/nearest_city'

# Define Parameters for Islamabad, Pakistan
LOCATION = {
    'lat': '33.6844',
    'lon': '73.0479',
    'country': 'PK',
    'state': 'Islamabad Capital Territory',
    'city': 'Islamabad'
}

# Prometheus Metrics Definitions
LIVE_DATA_FETCHED = Counter('live_data_fetched_total', 'Total live data fetched from APIs')
LIVE_DATA_FETCH_ERRORS = Counter('live_data_fetch_errors_total', 'Total errors while fetching live data')
DATA_INGESTED_TOTAL = Counter('data_ingested_total', 'Total data ingested')
MODEL_PREDICTIONS_TOTAL = Counter('model_predictions_total', 'Total number of model predictions')
MODEL_PREDICTION_ERRORS_TOTAL = Counter('model_prediction_errors_total', 'Total prediction errors')
API_REQUEST_LATENCY_SECONDS = Gauge('api_request_latency_seconds', 'Latency of API requests in seconds')
API_REQUESTS_IN_PROGRESS = Gauge('api_requests_in_progress', 'Number of API requests in progress')
CURRENT_TEMPERATURE = Gauge('current_temperature_celsius', 'Current temperature in Celsius')
CURRENT_HUMIDITY = Gauge('current_humidity_percent', 'Current humidity percentage')
CURRENT_AQI_US = Gauge('current_aqi_us', 'Current US AQI value')
VALIDATION_ACCURACY = Gauge('validation_accuracy_percent', 'Validation accuracy percentage')
VALIDATION_FAILURES = Counter('validation_failures_total', 'Total validation failures')

# Directory to Save Data
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Database Setup with SQLAlchemy
Base = declarative_base()

class LiveData(Base):
    __tablename__ = 'live_data'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    temperature = Column(Float)
    humidity = Column(Float)
    aqi = Column(Integer)
    prediction = Column(String)
    actual_temperature = Column(Float)

# Initialize SQLite Database
DATABASE_PATH = os.path.join(DATA_DIR, 'pipeline_data.db')
engine = create_engine(f'sqlite:///{DATABASE_PATH}')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Load the trained ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    logging.info("ML model loaded successfully.")
except Exception as e:
    logging.exception(f"Failed to load ML model: {e}")
    model = None

def fetch_openweathermap_data():
    """Fetch weather data from OpenWeatherMap API."""
    params = {
        'lat': LOCATION['lat'],
        'lon': LOCATION['lon'],
        'appid': OPENWEATHERMAP_API_KEY,
        'units': 'metric'
    }
    API_REQUESTS_IN_PROGRESS.inc()
    start_time = time.time()
    try:
        response = requests.get(OPENWEATHERMAP_URL, params=params)
        latency = time.time() - start_time
        API_REQUEST_LATENCY_SECONDS.set(latency)
        if response.status_code == 200:
            LIVE_DATA_FETCHED.inc()
            logging.info("Successfully fetched OpenWeatherMap data.")
            return response.json()
        elif response.status_code == 429:
            LIVE_DATA_FETCH_ERRORS.inc()
            logging.warning("OpenWeatherMap API rate limit exceeded. Backing off for 60 seconds.")
            time.sleep(60)
            return fetch_openweathermap_data()
        else:
            LIVE_DATA_FETCH_ERRORS.inc()
            logging.error(f"OpenWeatherMap API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        LIVE_DATA_FETCH_ERRORS.inc()
        logging.exception(f"Exception occurred while fetching OpenWeatherMap data: {e}")
        return None
    finally:
        API_REQUESTS_IN_PROGRESS.dec()

def fetch_airvisual_data():
    """Fetch air quality data from AirVisual API."""
    params = {
        'lat': LOCATION['lat'],
        'lon': LOCATION['lon'],
        'key': AIRVISUAL_API_KEY
    }
    API_REQUESTS_IN_PROGRESS.inc()
    start_time = time.time()
    try:
        response = requests.get(AIRVISUAL_URL, params=params)
        latency = time.time() - start_time
        API_REQUEST_LATENCY_SECONDS.set(latency)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                LIVE_DATA_FETCHED.inc()
                logging.info("Successfully fetched AirVisual data.")
                return data
            else:
                LIVE_DATA_FETCH_ERRORS.inc()
                logging.error(f"AirVisual API Error: {data['data']['message']}")
                return None
        elif response.status_code == 429:
            LIVE_DATA_FETCH_ERRORS.inc()
            logging.warning("AirVisual API rate limit exceeded. Backing off for 60 seconds.")
            time.sleep(60)
            return fetch_airvisual_data()
        else:
            LIVE_DATA_FETCH_ERRORS.inc()
            logging.error(f"AirVisual API HTTP Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        LIVE_DATA_FETCH_ERRORS.inc()
        logging.exception(f"Exception occurred while fetching AirVisual data: {e}")
        return None
    finally:
        API_REQUESTS_IN_PROGRESS.dec()

def save_data(data, filename):
    """Save data to a JSON file."""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        DATA_INGESTED_TOTAL.inc()
        logging.info(f"Data saved to {filepath}")
    except Exception as e:
        LIVE_DATA_FETCH_ERRORS.inc()
        logging.exception(f"Failed to save data to {filepath}: {e}")

def make_prediction(temperature, humidity, aqi):
    """Make a prediction using the ML model."""
    if not model:
        logging.error("ML model is not loaded. Cannot make predictions.")
        return None
    try:
        input_features = [[temperature, humidity, aqi]]
        prediction = model.predict(input_features)[0]
        return prediction
    except Exception as e:
        MODEL_PREDICTION_ERRORS_TOTAL.inc()
        logging.exception(f"Prediction Error: {e}")
        return None

def validate_prediction(prediction, actual):
    """Compare prediction with actual outcome and log accuracy."""
    try:
        if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
            difference = abs(prediction - actual)
            logging.info(f"Prediction Difference: {difference}°C")
            
            # Define acceptable difference threshold
            THRESHOLD = 2.0  # degrees Celsius
            
            if difference <= THRESHOLD:
                logging.info("Prediction is within the acceptable threshold.")
                VALIDATION_ACCURACY.set(100 - (difference / THRESHOLD) * 100)
            else:
                logging.warning("Prediction exceeds the acceptable threshold.")
                VALIDATION_ACCURACY.set(0)
                VALIDATION_FAILURES.inc()
    except Exception as e:
        MODEL_PREDICTION_ERRORS_TOTAL.inc()
        logging.exception(f"Error during prediction validation: {e}")

def save_to_db(temperature, humidity, aqi, prediction, actual_temperature):
    """Save prediction data to the database."""
    try:
        live_data = LiveData(
            temperature=temperature,
            humidity=humidity,
            aqi=aqi,
            prediction=prediction,
            actual_temperature=actual_temperature
        )
        session.add(live_data)
        session.commit()
        logging.info("Data saved to database.")
    except Exception as e:
        LIVE_DATA_FETCH_ERRORS.inc()
        logging.exception(f"Failed to save data to database: {e}")
        session.rollback()

def process_data(owm_data, av_data, ground_truth_data):
    """Process fetched data and perform predictions."""
    try:
        # Extract relevant information
        temperature = owm_data['main']['temp']
        humidity = owm_data['main']['humidity']
        aqi = av_data['data']['current']['pollution']['aqius']
        
        # Update Prometheus Gauges
        CURRENT_TEMPERATURE.set(temperature)
        CURRENT_HUMIDITY.set(humidity)
        CURRENT_AQI_US.set(aqi)
        
        # Extract actual outcome for validation
        actual_temperature = ground_truth_data['main']['temp']  # Adjust based on your ground truth data structure
        
        # Make Prediction
        prediction = make_prediction(temperature, humidity, aqi)
        if prediction is not None:
            MODEL_PREDICTIONS_TOTAL.inc()
            logging.info(f"Prediction made: {prediction}")
            
            # Save prediction data to database
            save_to_db(temperature, humidity, aqi, prediction, actual_temperature)
            
            # Save prediction data as JSON
            prediction_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'temperature': temperature,
                'humidity': humidity,
                'aqi': aqi,
                'prediction': prediction,
                'actual_temperature': actual_temperature
            }
            save_data(prediction_data, f'prediction_{int(time.time())}.json')
            
            # Validate Prediction
            validate_prediction(prediction, actual_temperature)
    except Exception as e:
        MODEL_PREDICTION_ERRORS_TOTAL.inc()
        logging.exception(f"Error during data processing and prediction: {e}")

def fetch_ground_truth_data():
    """Fetch actual outcomes to validate predictions."""
    # Implement fetching ground truth data
    # This could be another API call or data source providing actual values
    # For demonstration, assume it's another endpoint similar to OpenWeatherMap
    # Replace with actual ground truth data fetching logic
    try:
        # Example: Fetch actual data from a hypothetical ground truth API
        # GROUND_TRUTH_URL = 'http://api.groundtruth.com/data'
        # response = requests.get(GROUND_TRUTH_URL, params={'lat': LOCATION['lat'], 'lon': LOCATION['lon']})
        # if response.status_code == 200:
        #     return response.json()
        # else:
        #     logging.error(f"Ground Truth API Error: {response.status_code} - {response.text}")
        #     return None
        
        # Placeholder: Use the same OpenWeatherMap data as ground truth for demonstration
        return fetch_openweathermap_data()
    except Exception as e:
        LIVE_DATA_FETCH_ERRORS.inc()
        logging.exception(f"Exception occurred while fetching ground truth data: {e}")
        return None

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), mimetype='text/plain')

def fetch_live_data():
    """Fetch data from APIs and process it."""
    owm_data = fetch_openweathermap_data()
    if owm_data:
        save_data(owm_data, f'openweathermap_{int(time.time())}.json')
    
    av_data = fetch_airvisual_data()
    if av_data:
        save_data(av_data, f'airvisual_{int(time.time())}.json')
    
    # Fetch ground truth data
    ground_truth_data = fetch_ground_truth_data()
    if ground_truth_data:
        save_data(ground_truth_data, f'ground_truth_{int(time.time())}.json')
    
    if owm_data and av_data and ground_truth_data:
        process_data(owm_data, av_data, ground_truth_data)

def main():
    """Main function to schedule data fetching."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(fetch_live_data, 'interval', seconds=10)
    scheduler.start()
    logging.info("Scheduler started. Fetching data every 10 seconds.")
    
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logging.info("Scheduler shut down gracefully.")

if __name__ == '__main__':
    # Start Prometheus metrics server on port 8001
    start_http_server(8001)  # Port for Prometheus to scrape
    logging.info("Prometheus metrics server started on port 8001.")
    
    # Run the main data fetching loop
    main()
