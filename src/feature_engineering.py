import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
import joblib
import os

#load data

df = pd.read_csv("../data/raw/data.csv")


df.head()
df.info()
df.describe()



# Load raw CSV data from the given file path.

def load_csv_data(file_path: str = "../data/raw/data.csv") -> pd.DataFrame:
    
    return pd.read_csv(file_path)

class FeatureEngineer:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.pipeline = None

    def _build_pipeline(self, df: pd.DataFrame):
        # Placeholder for building pipeline logic
        pass


def _build_pipeline(self, df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if self.target_column in numeric_cols:
        print(f"{self.target_column} is numeric")
    elif self.target_column in categorical_cols:
        print(f"{self.target_column} is categorical")
    else:
        print("Target column type not identified")


fe = FeatureEngineer(target_column='loan_amount')
fe._build_pipeline(df)  # assuming df is already loaded


# Define pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 2. Define the target column
target_column = 'loan_amount'  # change this if needed

# --- 1. Feature Engineering: Time-Based Features ---
    # Convert 'TransactionStartTime' to datetime objects
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['TransactionHour'] = df['TransactionStartTime'].dt.hour
df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
df['TransactionMonth'] = df['TransactionStartTime'].dt.month
df['TransactionDay'] = df['TransactionStartTime'].dt.day # Add day of month as well



    # --- 2. Drop Irrelevant and Redundant Columns ---
    # Drop identifiers and columns with single unique values identified in EDA
    # TransactionId and BatchId are unique identifiers, not predictive features.
    # CurrencyCode and CountryCode have only one unique value, providing no variance.
    # TransactionStartTime is replaced by derived temporal features.
columns_to_drop = [
        'TransactionId', 'BatchId', 'CurrencyCode', 'CountryCode',
        'TransactionStartTime'
    ]
df = df.drop(columns=columns_to_drop)

# --- 3. Feature Engineering: Aggregate Features for High-Cardinality IDs ---
    # These features aim to capture historical behavior, particularly for potential repeat offenders.
    # Note: For strict time-series modeling, these aggregates should be computed on a
    # rolling basis or based on data preceding the current transaction to avoid data leakage.
    # For a general "model-ready format" script, we compute them globally.

    # Total transaction count for each Account, Customer, Subscription
df['Account_TransactionCount'] = df.groupby('AccountId')['AccountId'].transform('count')
df['Customer_TransactionCount'] = df.groupby('CustomerId')['CustomerId'].transform('count')
df['Subscription_TransactionCount'] = df.groupby('SubscriptionId')['SubscriptionId'].transform('count')

    # Fraudulent transaction count for each Account, Customer, Subscription
df['Account_FraudCount'] = df.groupby('AccountId')['FraudResult'].transform('sum')
df['Customer_FraudCount'] = df.groupby('CustomerId')['FraudResult'].transform('sum')
df['Subscription_FraudCount'] = df.groupby('SubscriptionId')['FraudResult'].transform('sum')

    # Fraud rate for each Account, Customer, Subscription (to avoid division by zero for new IDs, use fillna(0))
    # It's important to calculate fraud rate correctly to avoid division by zero or NaN issues
    # First, calculate sum of FraudResult, then count of transactions, then divide.
    # This also helps ensure that new IDs (not seen during training) have a default rate, e.g., 0.
account_fraud_stats = df.groupby('AccountId')['FraudResult'].agg(['sum', 'count'])
df['Account_FraudRate'] = df['AccountId'].map(
        (account_fraud_stats['sum'] / account_fraud_stats['count']).fillna(0)
    )

customer_fraud_stats = df.groupby('CustomerId')['FraudResult'].agg(['sum', 'count'])
df['Customer_FraudRate'] = df['CustomerId'].map(
        (customer_fraud_stats['sum'] / customer_fraud_stats['count']).fillna(0)
    )

subscription_fraud_stats = df.groupby('SubscriptionId')['FraudResult'].agg(['sum', 'count'])
df['Subscription_FraudRate'] = df['SubscriptionId'].map(
        (subscription_fraud_stats['sum'] / subscription_fraud_stats['count']).fillna(0)
    )

# --- 4. Define Feature Sets for Preprocessing ---
    # The target variable 'FraudResult' should not be part of the features (X) during preprocessing.
target = df['FraudResult']
features_df = df.drop(columns=['FraudResult'])

    # Separate numerical and categorical features for transformation
    # Note: High-cardinality IDs are now used to generate numerical aggregates and will be dropped
    # as their raw string values are too many for direct one-hot encoding in a simple pipeline.
    # If they were to be kept as categorical, Weight of Evidence (WoE) or target encoding would be better.
numerical_features = [
        'Amount', 'Value', 'TransactionHour', 'TransactionDayOfWeek', 'TransactionMonth', 'TransactionDay',
        'Account_TransactionCount', 'Customer_TransactionCount', 'Subscription_TransactionCount',
        'Account_FraudCount', 'Customer_FraudCount', 'Subscription_FraudCount',
        'Account_FraudRate', 'Customer_FraudRate', 'Subscription_FraudRate'
    ]

    # Ensure all numerical features exist after dropping/creating
numerical_features = [f for f in numerical_features if f in features_df.columns]

categorical_features = [
        'ProductCategory', 'ChannelId', 'ProviderId', 'PricingStrategy'
    ]
    # Ensure all categorical features exist
categorical_features = [f for f in categorical_features if f in features_df.columns]


# --- 5. Create Preprocessing Pipeline (ColumnTransformer) ---
    # Numerical pipeline: Impute NaNs (e.g., from std of single-item groups), then transform for skewness, then standardize.
numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Fills NaNs, e.g., from std of single-item groups
        ('power', PowerTransformer(method='yeo-johnson', standardize=True)), # Handles skewness and scales
        ('scaler', StandardScaler()) # Ensures mean 0, std 1 after power transform
    ])

    # Categorical pipeline: One-Hot Encode nominal features.
    # Label Encoding is not applied here as the current categorical features are nominal.
    # If ordinal features were present, LabelEncoder would be used for them.
categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though none expected after selection
    )

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features = cat_encoder.get_feature_names_out()


feature_names_transformed = list(numerical_features) + list(cat_features)


print(cat_encoder.feature_names_in_)
print(categorical_features)

cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()



# Transform features
transformed_features_array = preprocessor.fit_transform(features_df)

# Get transformed column names
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features = cat_encoder.get_feature_names_out()

# Combine with numerical column names
feature_names_transformed = list(numerical_features) + list(cat_features)

# Reconstruct the DataFrame
#processed_df = pd.DataFrame(transformed_features_array, columns=feature_names_transformed, index=features_df.index)


print("Shape of transformed array:", transformed_features_array.shape)
print("Number of numerical features:", len(numerical_features))
print("Number of categorical dummy features:", len(cat_features))
print("Total column names:", len(numerical_features) + len(cat_features))
