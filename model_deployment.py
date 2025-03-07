import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load cleaned data from JSON
data = pd.read_json('data/cleaned_weather_data.json', lines=True)

# Ensure the target column exists
target = 'Temperature'  # Update the target to an existing column
if target not in data.columns:
    raise ValueError(f"The dataset must contain the target column '{target}'.")

# Use the DataFrame index as a sequential "time" variable
data['Index'] = range(len(data))  # Create a sequential index
data.set_index('Index', inplace=True)

# Display the first few rows of the data
print("Data Preview:")
print(data.head())

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

print(f"Training Data Shape: {train.shape}")
print(f"Testing Data Shape: {test.shape}")

# ARIMA Model
print("\n--- ARIMA Model ---")
try:
    # Fit the ARIMA model
    model_arima = ARIMA(train[target], order=(5, 1, 0))  # (p, d, q) parameters
    model_arima_fit = model_arima.fit()

    # Forecast
    forecast_arima = model_arima_fit.forecast(steps=len(test))

    # Calculate metrics
    rmse_arima = np.sqrt(mean_squared_error(test[target], forecast_arima))
    mae_arima = mean_absolute_error(test[target], forecast_arima)

    print(f"ARIMA RMSE: {rmse_arima}")
    print(f"ARIMA MAE: {mae_arima}")

    # Plot ARIMA Predictions
    plt.figure(figsize=(14, 5))
    plt.plot(test.index, test[target], label='Actual Temperature')
    plt.plot(test.index, forecast_arima, label='ARIMA Forecast')
    plt.title('ARIMA Forecast vs Actual Temperature')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
except Exception as e:
    print(f"ARIMA Model Error: {e}")

# LSTM Model
print("\n--- LSTM Model ---")
def create_dataset(series, time_step=1):
    X, Y = [], []
    for i in range(len(series) - time_step - 1):
        a = series[i:(i + time_step)]
        X.append(a)
        Y.append(series[i + time_step])
    return np.array(X), np.array(Y)

try:
    # Prepare data for LSTM
    series = data[target].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))

    # Create datasets
    time_step = 4
    X, Y = create_dataset(series_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # [samples, time_steps, features]

    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Build LSTM model
    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])

    # Compile the model
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model_lstm.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)

    # Predict
    predictions = model_lstm.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Inverse transform Y_test
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Calculate metrics
    rmse_lstm = np.sqrt(mean_squared_error(Y_test_inv, predictions))
    mae_lstm = mean_absolute_error(Y_test_inv, predictions)

    print(f"LSTM RMSE: {rmse_lstm}")
    print(f"LSTM MAE: {mae_lstm}")

    # Plot LSTM Predictions
    plt.figure(figsize=(14, 5))
    plt.plot(test.index[time_step+1:], Y_test_inv, label='Actual Temperature')
    plt.plot(test.index[time_step+1:], predictions, label='LSTM Forecast')
    plt.title('LSTM Forecast vs Actual Temperature')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
except Exception as e:
    print(f"LSTM Model Error: {e}")

# Save the LSTM model in .h5 format
model_lstm.save('lstm_model.h5')
print("LSTM model saved as 'lstm_model.h5'")
