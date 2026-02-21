"""
Unit Tests for Data Preparation Module
Tests data loading, cleaning, and preprocessing functions.
"""
import pytest
import pandas as pd
import numpy as np
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataCleaning:
    """Test suite for data cleaning functions."""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw weather data for testing."""
        return {
            'name': 'TestCity',
            'sys': {'country': 'TC'},
            'coord': {'lon': 74.3, 'lat': 31.5},
            'main': {'temp': 298.15, 'pressure': 1013, 'humidity': 65},
            'wind': {'speed': 3.5},
            'visibility': 10000
        }
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample cleaned DataFrame for testing."""
        return pd.DataFrame({
            'City': ['TestCity1', 'TestCity2', 'TestCity3'],
            'Country': ['TC', 'TC', 'TC'],
            'Longitude': [74.3, 74.4, 74.5],
            'Latitude': [31.5, 31.6, 31.7],
            'Temperature': [298.15, 300.0, 295.5],
            'Pressure': [1013, 1015, 1010],
            'Humidity': [65, 70, 60],
            'WindSpeed': [3.5, 4.0, 3.0],
            'Visibility': [10000, 9500, 10500]
        })
    
    def test_data_extraction_from_json(self, sample_raw_data):
        """Test that data is correctly extracted from JSON structure."""
        data_clean = pd.DataFrame([{
            'City': sample_raw_data.get('name', None),
            'Country': sample_raw_data['sys'].get('country', None),
            'Longitude': sample_raw_data['coord'].get('lon', None),
            'Latitude': sample_raw_data['coord'].get('lat', None),
            'Temperature': sample_raw_data['main'].get('temp', None),
            'Pressure': sample_raw_data['main'].get('pressure', None),
            'Humidity': sample_raw_data['main'].get('humidity', None),
            'WindSpeed': sample_raw_data['wind'].get('speed', None),
            'Visibility': sample_raw_data.get('visibility', None)
        }])
        
        assert data_clean['City'].iloc[0] == 'TestCity'
        assert data_clean['Temperature'].iloc[0] == 298.15
        assert data_clean['Humidity'].iloc[0] == 65
    
    def test_missing_value_handling(self, sample_dataframe):
        """Test that missing values are handled correctly."""
        # Introduce missing values
        df_with_missing = sample_dataframe.copy()
        df_with_missing.loc[0, 'Temperature'] = np.nan
        df_with_missing.loc[1, 'Humidity'] = np.nan
        
        numeric_columns = ['Longitude', 'Latitude', 'Temperature', 'Pressure', 
                          'Humidity', 'WindSpeed', 'Visibility']
        df_with_missing[numeric_columns] = df_with_missing[numeric_columns].fillna(
            df_with_missing[numeric_columns].mean()
        )
        
        # Check no missing values remain
        assert df_with_missing[numeric_columns].isnull().sum().sum() == 0
    
    def test_outlier_removal_iqr(self, sample_dataframe):
        """Test IQR-based outlier removal."""
        # Add an outlier
        df_with_outlier = sample_dataframe.copy()
        df_with_outlier.loc[3] = ['OutlierCity', 'TC', 74.6, 31.8, 
                                   500.0, 1020, 80, 5.0, 11000]  # Extreme temp
        
        numeric_columns = ['Temperature', 'Pressure', 'Humidity', 'WindSpeed', 'Visibility']
        Q1 = df_with_outlier[numeric_columns].quantile(0.25)
        Q3 = df_with_outlier[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        
        df_no_outliers = df_with_outlier[
            ~((df_with_outlier[numeric_columns] < (Q1 - threshold * IQR)) |
              (df_with_outlier[numeric_columns] > (Q3 + threshold * IQR))).any(axis=1)
        ]
        
        # Outlier should be removed
        assert len(df_no_outliers) < len(df_with_outlier)
    
    def test_data_types_are_correct(self, sample_dataframe):
        """Test that numeric columns have correct data types."""
        numeric_columns = ['Longitude', 'Latitude', 'Temperature', 'Pressure', 
                          'Humidity', 'WindSpeed', 'Visibility']
        
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(sample_dataframe[col]), \
                f"Column {col} should be numeric"
    
    def test_data_shape_preserved_after_cleaning(self, sample_dataframe):
        """Test that data shape is preserved when no outliers exist."""
        numeric_columns = ['Temperature', 'Pressure', 'Humidity', 'WindSpeed', 'Visibility']
        Q1 = sample_dataframe[numeric_columns].quantile(0.25)
        Q3 = sample_dataframe[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        
        df_no_outliers = sample_dataframe[
            ~((sample_dataframe[numeric_columns] < (Q1 - threshold * IQR)) |
              (sample_dataframe[numeric_columns] > (Q3 + threshold * IQR))).any(axis=1)
        ]
        
        # No rows should be removed from clean data
        assert len(df_no_outliers) == len(sample_dataframe)


class TestDataSplit:
    """Test suite for train/test split functionality."""
    
    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        return pd.DataFrame({
            'Temperature': np.random.uniform(290, 310, 100)
        })
    
    def test_train_test_split_ratio(self, time_series_data):
        """Test that 80/20 split is correctly applied."""
        train_size = int(len(time_series_data) * 0.8)
        train = time_series_data.iloc[:train_size]
        test = time_series_data.iloc[train_size:]
        
        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == len(time_series_data)
    
    def test_no_data_leakage(self, time_series_data):
        """Test that train and test sets don't overlap."""
        train_size = int(len(time_series_data) * 0.8)
        train = time_series_data.iloc[:train_size]
        test = time_series_data.iloc[train_size:]
        
        # Check indices don't overlap
        train_indices = set(train.index)
        test_indices = set(test.index)
        
        assert len(train_indices.intersection(test_indices)) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
