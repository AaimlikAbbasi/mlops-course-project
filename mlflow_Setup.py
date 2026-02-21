import mlflow
import joblib
import random
# Set the tracking URI to a local directory
mlflow.set_tracking_uri("file:///C:/MLOPS_PROJECT/course-project-AaimlikAbbasi/mlruns")
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.statsmodels
import matplotlib.pyplot as plt

# ============================================
# REPRODUCIBILITY: Set random seeds
# ============================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Load cleaned data
data = pd.read_json('data/cleaned_weather_data.json', lines=True)
data['Index'] = range(len(data))  # Create a sequential index
data.set_index('Index', inplace=True)

# Set target
target = 'Temperature'  # Use Temperature as the target
if target not in data.columns:
    raise ValueError(f"The dataset must contain the target column '{target}'.")

# Split data
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Start MLflow run
with mlflow.start_run(run_name="ARIMA_Model"):
    # Define ARIMA parameters
    p, d, q = 5, 1, 0
    mlflow.log_param("p", p)
    mlflow.log_param("d", d)
    mlflow.log_param("q", q)
    
    # Fit ARIMA model
    model_arima = ARIMA(train[target], order=(p, d, q))
    model_arima_fit = model_arima.fit()
    
    # Forecast
    forecast_arima = model_arima_fit.forecast(steps=len(test))
    
    # Calculate metrics
    rmse_arima = np.sqrt(mean_squared_error(test[target], forecast_arima))
    mae_arima = mean_absolute_error(test[target], forecast_arima)
    
    mlflow.log_metric("RMSE", rmse_arima)
    mlflow.log_metric("MAE", mae_arima)
    
    # Log the model
    mlflow.statsmodels.log_model(model_arima_fit, "arima_model")
    
    print(f"ARIMA RMSE: {rmse_arima}")
    print(f"ARIMA MAE: {mae_arima}")
    
    # Save and plot forecasts
    plt.figure(figsize=(14, 5))
    plt.plot(test.index, test[target], label='Actual Temperature')
    plt.plot(test.index, forecast_arima, label='ARIMA Forecast')
    plt.title('ARIMA Forecast vs Actual Temperature')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.legend()
    plt.savefig('arima_forecast.png')
    plt.close()
    
    mlflow.log_artifact('arima_forecast.png')

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

# Load cleaned data
data = pd.read_json('data/cleaned_weather_data.json', lines=True)
data['Index'] = range(len(data))  # Create a sequential index
data.set_index('Index', inplace=True)

# Set target
target = 'Temperature'  # Use Temperature as the target
series = data[target].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series.reshape(-1, 1))

# Create dataset
def create_dataset(series, time_step=1):
    X, Y = [], []
    for i in range(len(series) - time_step - 1):
        a = series[i:(i + time_step)]
        X.append(a)
        Y.append(series[i + time_step])
    return np.array(X), np.array(Y)

time_step = 5  # Adjust time_step based on dataset size
X, Y = create_dataset(series_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Start MLflow run
with mlflow.start_run(run_name="LSTM_Model"):
    # Define hyperparameters
    epochs = 20
    batch_size = 32
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    
    # Build LSTM
    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    
    # Compile
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    model_lstm.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Predict
    predictions = model_lstm.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))
    
    # Calculate metrics
    rmse_lstm = np.sqrt(mean_squared_error(Y_test_inv, predictions))
    mae_lstm = mean_absolute_error(Y_test_inv, predictions)
    
    mlflow.log_metric("RMSE", rmse_lstm)
    mlflow.log_metric("MAE", mae_lstm)
    
    # Log the model
    mlflow.tensorflow.log_model(model_lstm, "lstm_model")
    
    print(f"LSTM RMSE: {rmse_lstm}")
    print(f"LSTM MAE: {mae_lstm}")
    
    # Save and plot predictions
    plt.figure(figsize=(14, 5))
    plt.plot(data.index[-len(predictions):], Y_test_inv, label='Actual Temperature')
    plt.plot(data.index[-len(predictions):], predictions, label='LSTM Forecast')
    plt.title('LSTM Forecast vs Actual Temperature')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.legend()
    plt.savefig('lstm_forecast.png')
    plt.close()
    
    mlflow.log_artifact('lstm_forecast.png')

    # Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series.reshape(-1, 1))

# Save the scaler for later use
joblib.dump(scaler, "data/scaler.pkl")
print("Scaler saved to data/scaler.pkl")
