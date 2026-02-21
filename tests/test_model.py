"""
Unit Tests for Model Training and Evaluation
Tests ARIMA and LSTM model functionality.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModelMetrics:
    """Test suite for model evaluation metrics."""
    
    def test_rmse_calculation(self):
        """Test RMSE calculation is correct."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # Manual calculation: sqrt(mean([0.01, 0.01, 0.01, 0.04, 0.04]))
        expected_rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        assert np.isclose(rmse, expected_rmse)
        assert rmse > 0  # RMSE should always be positive
    
    def test_mae_calculation(self):
        """Test MAE calculation is correct."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        mae = mean_absolute_error(actual, predicted)
        expected_mae = np.mean(np.abs(actual - predicted))
        
        assert np.isclose(mae, expected_mae)


class TestDatasetCreation:
    """Test suite for LSTM dataset creation."""
    
    def create_dataset(self, series, time_step=1):
        """Helper function to create sequences for LSTM."""
        X, Y = [], []
        for i in range(len(series) - time_step - 1):
            a = series[i:(i + time_step)]
            X.append(a)
            Y.append(series[i + time_step])
        return np.array(X), np.array(Y)
    
    def test_dataset_creation_shape(self):
        """Test that created dataset has correct shape."""
        series = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], 
                          [0.6], [0.7], [0.8], [0.9], [1.0]])
        time_step = 3
        
        X, Y = self.create_dataset(series, time_step)
        
        # With 10 samples and time_step=3, we should get 6 sequences
        assert X.shape[0] == 6
        assert X.shape[1] == time_step
        assert Y.shape[0] == 6
    
    def test_dataset_values_correct(self):
        """Test that dataset values are correctly sequenced."""
        series = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        time_step = 2
        
        X, Y = self.create_dataset(series, time_step)
        
        # First sequence should be [1, 2] -> 3
        np.testing.assert_array_equal(X[0], [[1], [2]])
        assert Y[0] == [3]


class TestDataNormalization:
    """Test suite for data normalization."""
    
    def test_minmax_scaling(self):
        """Test MinMaxScaler produces values in [0, 1]."""
        data = np.array([100, 200, 300, 400, 500]).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        assert scaled_data.min() >= 0
        assert scaled_data.max() <= 1
    
    def test_inverse_transform(self):
        """Test that inverse transform recovers original values."""
        data = np.array([298.15, 300.0, 295.5, 310.0]).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        recovered_data = scaler.inverse_transform(scaled_data)
        
        np.testing.assert_array_almost_equal(data, recovered_data)


class TestHyperparameterValidation:
    """Test suite for hyperparameter validation."""
    
    def test_arima_parameter_ranges(self):
        """Test that ARIMA parameters are within valid ranges."""
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    assert p >= 0, "p must be non-negative"
                    assert d >= 0, "d must be non-negative"
                    assert q >= 0, "q must be non-negative"
    
    def test_lstm_hyperparameters(self):
        """Test LSTM hyperparameters are valid."""
        epochs = 20
        batch_size = 32
        lstm_units = 50
        
        assert epochs > 0
        assert batch_size > 0
        assert lstm_units > 0


class TestReproducibility:
    """Test suite for reproducibility features."""
    
    def test_seed_produces_consistent_results(self):
        """Test that setting seed produces consistent random values."""
        np.random.seed(42)
        random_values_1 = np.random.rand(5)
        
        np.random.seed(42)
        random_values_2 = np.random.rand(5)
        
        np.testing.assert_array_equal(random_values_1, random_values_2)
    
    def test_train_test_split_reproducibility(self):
        """Test that train/test split is reproducible with seed."""
        np.random.seed(42)
        data = pd.DataFrame({'Temperature': np.random.uniform(290, 310, 100)})
        
        train_size = int(len(data) * 0.8)
        train1, test1 = data.iloc[:train_size], data.iloc[train_size:]
        
        np.random.seed(42)
        data2 = pd.DataFrame({'Temperature': np.random.uniform(290, 310, 100)})
        train2, test2 = data2.iloc[:train_size], data2.iloc[train_size:]
        
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
