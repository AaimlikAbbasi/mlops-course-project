import pandas as pd
import numpy as np
import random
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.statsmodels

# ============================================
# REPRODUCIBILITY: Set random seeds
# ============================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Note: TensorFlow seeds are set in LSTM section below
# Remaining nondeterminism: ARIMA optimization uses numerical solvers 
# that may have slight variations across different hardware/BLAS implementations

# Load cleaned data from JSON
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

# Define parameter ranges
p_values = range(0, 3)  # Reduced ranges to save time with limited data
d_values = range(0, 2)
q_values = range(0, 3)

best_score, best_cfg = float("inf"), None

# Start MLflow experiment
mlflow.set_experiment("ARIMA_Hyperparameter_Tuning")

# Iterate through all combinations
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                # Start MLflow run
                with mlflow.start_run(run_name=f"ARIMA_{order}"):
                    # Log parameters
                    mlflow.log_param("p", p)
                    mlflow.log_param("d", d)
                    mlflow.log_param("q", q)
                    
                    # Fit ARIMA model
                    model = ARIMA(train[target], order=order)
                    model_fit = model.fit()
                    
                    # Forecast
                    forecast = model_fit.forecast(steps=len(test))
                    
                    # Calculate metrics
                    rmse = np.sqrt(mean_squared_error(test[target], forecast))
                    mlflow.log_metric("RMSE", rmse)
                    
                    # Log the model
                    mlflow.statsmodels.log_model(model_fit, f"arima_model_{order}")
                    
                    print(f"ARIMA{order} RMSE={rmse}")
                    
                    # Update best score
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
            except Exception as e:
                print(f"ARIMA{order} failed with error: {e}")
                continue

print(f"Best ARIMA{best_cfg} RMSE={best_score}")


import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

# ============================================
# REPRODUCIBILITY: Set seeds for LSTM/TensorFlow
# ============================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Load cleaned data from JSON
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

time_step = 5  # Adjusted for smaller dataset
X, Y = create_dataset(series_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Define hyperparameter ranges
epochs_list = [10, 20]
batch_size_list = [16, 32]

best_rmse, best_params = float("inf"), None

# Start MLflow experiment
mlflow.set_experiment("LSTM_Hyperparameter_Tuning")

for epochs in epochs_list:
    for batch_size in batch_size_list:
        with mlflow.start_run(run_name=f"LSTM_epochs{epochs}_batch{batch_size}"):
            # Log hyperparameters
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            
            # Build LSTM
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                LSTM(50, return_sequences=False),
                Dense(1)
            ])
            
            # Compile
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train
            model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Predict
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(Y_test_inv, predictions))
            mae = mean_absolute_error(Y_test_inv, predictions)
            
            # Log metrics
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            
            # Log the model
            mlflow.tensorflow.log_model(model, f"lstm_model_epochs{epochs}_batch{batch_size}")
            
            print(f"LSTM Epochs={epochs}, Batch Size={batch_size} => RMSE={rmse}, MAE={mae}")
            
            # Update best score
            if rmse < best_rmse:
                best_rmse, best_params = rmse, (epochs, batch_size)

print(f"Best LSTM Params: Epochs={best_params[0]}, Batch Size={best_params[1]} with RMSE={best_rmse}")
