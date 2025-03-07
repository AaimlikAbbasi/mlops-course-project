
import requests
import json
import os
from datetime import datetime

# Configuration
OPENWEATHERMAP_API_KEY = '078e06f092d156e65754a31d674d49ec'
AIRVISUAL_API_KEY = '91b16d71-9ed4-45e8-ba20-0e28e99875b2'  # If using AirVisual
# Define API Endpoints
OPENWEATHERMAP_URL = 'http://api.openweathermap.org/data/2.5/weather'
AIRVISUAL_URL = 'http://api.airvisual.com/v2/nearest_city'

# Define Parameters (Updated for Islamabad, Pakistan)
LOCATION = {
    'lat': '33.6844',      # Latitude for Islamabad
    'lon': '73.0479',      # Longitude for Islamabad
    # 'zip': 'some_postal_code',  # Removed or updated if you have a specific postal code
    'country': 'PK',       # Country code for Pakistan
    'state': 'Islamabad Capital Territory',  # State/Province
    'city': 'Islamabad'    # City
}

# Directory to Save Data
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_openweathermap_data():
    params = {
        'lat': LOCATION['lat'],
        'lon': LOCATION['lon'],
        'appid': OPENWEATHERMAP_API_KEY,
        'units': 'metric'
    }
    response = requests.get(OPENWEATHERMAP_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"OpenWeatherMap API Error: {response.status_code}")
        return None

def fetch_airvisual_data():
    params = {
        'lat': LOCATION['lat'],
        'lon': LOCATION['lon'],
        'key': AIRVISUAL_API_KEY
    }
    response = requests.get(AIRVISUAL_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"AirVisual API Error: {response.status_code}")
        return None

# Removed EPA AirNow and NOAA functions as they are US-specific
# If you have equivalent APIs for Pakistan, you can add similar functions

# def fetch_epa_airnow_data():
#     params = {
#         'zipCode': LOCATION['zip'],
#         'format': 'application/json',
#         'API_KEY': EPA_AIRNOW_API_KEY
#     }
#     response = requests.get(EPA_AIRNOW_URL, params=params)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"EPA AirNow API Error: {response.status_code}")
#         return None

# def fetch_noaa_data():
#     # NOAA API usage can be complex; this is a placeholder
#     # You need to specify grid points and other parameters
#     response = requests.get(NOAA_URL)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"NOAA API Error: {response.status_code}")
#         return None

def save_data(data, filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filepath}")

def main():
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    
    # Fetch and save OpenWeatherMap data
    owm_data = fetch_openweathermap_data()
    if owm_data:
        save_data(owm_data, f'openweathermap_{timestamp}.json')
    
    # Fetch and save AirVisual data
    av_data = fetch_airvisual_data()
    if av_data:
        save_data(av_data, f'airvisual_{timestamp}.json')
    
    # Fetch and save EPA AirNow data (Commented out)
    # epa_data = fetch_epa_airnow_data()
    # if epa_data:
    #     save_data(epa_data, f'epa_airnow_{timestamp}.json')
    
    # Fetch and save NOAA data (Commented out)
    # noaa_data = fetch_noaa_data()
    # if noaa_data:
    #     save_data(noaa_data, f'noaa_{timestamp}.json')

if __name__ == '__main__':
    main()
