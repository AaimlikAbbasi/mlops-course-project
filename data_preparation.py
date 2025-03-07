import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Load JSON data
with open('data/openweathermap_20241212142802.json', 'r') as file:  # Corrected filename
    raw_data = json.load(file)

# Extract weather and pollution data
data_clean = pd.DataFrame([{
    'City': raw_data.get('name', None),
    'Country': raw_data['sys'].get('country', None),
    'Longitude': raw_data['coord'].get('lon', None),
    'Latitude': raw_data['coord'].get('lat', None),
    'Temperature': raw_data['main'].get('temp', None),
    'Pressure': raw_data['main'].get('pressure', None),
    'Humidity': raw_data['main'].get('humidity', None),
    'WindSpeed': raw_data['wind'].get('speed', None),
    'Visibility': raw_data.get('visibility', None)
}])

# Display initial data
print("Initial Data:")
print(data_clean.head())

# Check for missing values
print("\nMissing Values:")
print(data_clean.isnull().sum())

# Handle missing values (numeric columns only)
numeric_columns = ['Longitude', 'Latitude', 'Temperature', 'Pressure', 'Humidity', 'WindSpeed', 'Visibility']
data_clean[numeric_columns] = data_clean[numeric_columns].fillna(data_clean[numeric_columns].mean())

# Verify no missing values remain
print("\nData After Handling Missing Values:")
print(data_clean.isnull().sum())

# Visualize outliers using boxplots for numeric columns
sns.boxplot(data=data_clean[numeric_columns])
plt.title('Boxplot for Numeric Features')
plt.show()

# Remove outliers using the IQR method
Q1 = data_clean[numeric_columns].quantile(0.25)
Q3 = data_clean[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Define a threshold (e.g., 1.5 times the IQR)
threshold = 1.5
data_no_outliers = data_clean[~((data_clean[numeric_columns] < (Q1 - threshold * IQR)) |
                                (data_clean[numeric_columns] > (Q3 + threshold * IQR))).any(axis=1)]

# Verify outliers are removed
sns.boxplot(data=data_no_outliers[numeric_columns])
plt.title('Boxplot After Removing Outliers')
plt.show()

# Save cleaned data to JSON
data_no_outliers.to_json('data/cleaned_weather_data.json', orient='records', lines=True)
