# tests/test_model.py

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to Python path to enable imports from src/
try:
    # This works in .py files
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # This works in notebooks
    base_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

sys.path.append(base_path)

# Mock raw data for testing
@pytest.fixture
def sample_raw_df():
    """Provides a sample raw DataFrame for testing."""
    data = {
        'TransactionId': [f'T{i}' for i in range(10)],
        'BatchId': [f'B{i}' for i in range(10)],
        'AccountId': ['A1', 'A2', 'A1', 'A3', 'A1', 'A2', 'A4', 'A1', 'A3', 'A5'],
        'SubscriptionId': ['S1', 'S2', 'S1', 'S3', 'S1', 'S2', 'S4', 'S1', 'S3', 'S5'],
        'CustomerId': ['C1', 'C2', 'C1', 'C3', 'C1', 'C2', 'C4', 'C1', 'C3', 'C5'],
        'CurrencyCode': ['UGX'] * 10,
        'CountryCode': [256] * 10,
        'ProviderId': ['P1', 'P2', 'P1', 'P3', 'P1', 'P2', 'P4', 'P1', 'P3', 'P5'],
        'ProductId': ['Prod1', 'Prod2', 'Prod1', 'Prod3', 'Prod1', 'Prod2', 'Prod4', 'Prod1', 'Prod3', 'Prod5'],
        'ProductCategory': ['airtime', 'financial_services', 'airtime', 'utility_bill', 'airtime', 'financial_services', 'airtime', 'airtime', 'utility_bill', 'other'],
        'ChannelId': ['Ch1', 'Ch2', 'Ch1', 'Ch3', 'Ch1', 'Ch2', 'Ch1', 'Ch1', 'Ch3', 'Ch1'],
        'Amount': [100, 200, 150, 500, 120, 220, 180, 110, 550, 300],
        'Value': [100, 200, 150, 500, 120, 220, 180, 110, 550, 300],
        'TransactionStartTime': pd.to_datetime([
            '2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00',
            '2023-01-04 13:00:00', '2023-01-05 14:00:00', '2023-01-06 15:00:00',
            '2023-01-07 16:00:00', '2023-01-08 17:00:00', '2023-01-09 18:00:00',
            '2023-01-10 19:00:00'
        ]),
        'PricingStrategy': [1, 2, 1, 3, 1, 2, 1, 1, 3, 4],
        'FraudResult': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0] # Some fraud for testing
    }
    return pd.DataFrame(data)

def test_engineer_proxy_target_output_shape(sample_raw_df):
    """Test that engineer_proxy_target adds the 'is_high_risk' column."""
    df_with_proxy = engineer_proxy_target(sample_raw_df)
    assert 'is_high_risk' in df_with_proxy.columns
    assert df_with_proxy['is_high_risk'].dtype == 'int64'

def test_preprocess_data_output_shape(sample_raw_df):
    """Test that preprocess_data returns a DataFrame with expected columns and no NaNs."""
    # First, engineer the proxy target as preprocess_data expects it
    df_with_proxy = engineer_proxy_target(sample_raw_df)

    # Now, prepare for preprocess_data.
    # The preprocess_data function expects the raw_df structure for aggregations.
    # So, we pass the original raw_df to it, and handle the target separately.
    # This is a bit tricky due to how `preprocess_data` was designed.
    # For this test, we'll ensure it runs and produces a valid output.

    # Simulate the data preparation before calling preprocess_data as in model_training.py
    target_is_high_risk = df_with_proxy['is_high_risk']
    original_fraud_result = df_with_proxy['FraudResult'] # This is now a feature

    df_temp_for_preprocessing = df_with_proxy.drop(columns=['is_high_risk', 'FraudResult'])
    processed_df = preprocess_data(df_temp_for_preprocessing)

    assert isinstance(processed_df, pd.DataFrame)
    assert processed_df.shape[0] == sample_raw_df.shape[0]
    # Check for NaNs in the processed features
    assert processed_df.isnull().sum().sum() == 0

def test_preprocess_data_column_types(sample_raw_df):
    """Test that numerical features are numeric and categorical are encoded."""
    df_with_proxy = engineer_proxy_target(sample_raw_df)
    target_is_high_risk = df_with_proxy['is_high_risk']
    original_fraud_result = df_with_proxy['FraudResult']

    df_temp_for_preprocessing = df_with_proxy.drop(columns=['is_high_risk', 'FraudResult'])
    processed_df = preprocess_data(df_temp_for_preprocessing)

    # Check that all columns are numerical after preprocessing
    for col in processed_df.columns:
        assert pd.api.types.is_numeric_dtype(processed_df[col]), f"Column {col} is not numeric"