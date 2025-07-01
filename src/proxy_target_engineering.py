import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# Suppress KMeans deprecation warning for n_init
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

def engineer_proxy_target(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers a proxy target variable 'is_high_risk' by calculating RFM metrics,
    clustering customers, and identifying the least engaged (high-risk) segment.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame loaded from data.csv.
                               Must contain 'CustomerId', 'TransactionStartTime', and 'Value'.

    Returns:
        pd.DataFrame: The DataFrame with the new 'is_high_risk' column merged in.
                      This DataFrame is ready to be passed to the main feature engineering
                      pipeline (e.g., preprocess_data from Task 3).
    """

    df = df_raw.copy()

    # Ensure TransactionStartTime is datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Load Data

df = pd.read_csv('../data/raw/data.csv')  # replace with your correct path


# Ensure datetime format
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
# Now calculate snapshot_date
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)


# --- 1. Calculate RFM Metrics ---
    # Define a snapshot date (e.g., one day after the latest transaction in the dataset)


    # Calculate RFM for each CustomerId
rfm_df = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
        Frequency=('TransactionId', 'count'), # Count of transactions
        Monetary=('Value', 'sum') # Sum of transaction values
    ).reset_index()

print(df['TransactionStartTime'].dtype)


# --- 2. Cluster Customers using K-Means ---
    # Select RFM features for clustering
rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]

    # Pre-process (scale) RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

    # Apply K-Means clustering
    # Using n_init='auto' for modern sklearn versions
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# --- 3. Define and Assign the "High-Risk" Label ---
    # Analyze cluster centroids to identify the high-risk cluster.
    # High-risk customers are typically characterized by:
    # - High Recency (haven't transacted recently)
    # - Low Frequency (don't transact often)
    # - Low Monetary (low total spending)

    # Get cluster centroids (in scaled space)
centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=rfm_features.columns)

    # To interpret, we can inverse transform centroids or just look at relative values.
    # For simplicity, let's look at the scaled centroids directly to find the "least engaged"
    # Least engaged = highest Recency_scaled, lowest Frequency_scaled, lowest Monetary_scaled

    # Calculate a 'risk score' for each centroid (lower score = higher risk)
    # Example: (Recency_scaled) - (Frequency_scaled) - (Monetary_scaled)
    # Or simply sort by Recency (desc), then Frequency (asc), then Monetary (asc)
    # For a robust approach, we'll identify the cluster with the highest mean Recency
    # and lowest mean Frequency/Monetary from the *original* RFM values.

cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(
        by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]
    )
print("Cluster Summary (Mean RFM Values):\n", cluster_summary.to_markdown(numalign="left", stralign="left"))

    # The first cluster in the sorted summary is likely the high-risk one
high_risk_cluster_label = cluster_summary.index[0]
print(f"\nIdentified High-Risk Cluster Label: {high_risk_cluster_label}")

    # Create the 'is_high_risk' column
rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_label).astype(int)


df = pd.read_csv('../data/raw/data.csv')  # or wherever your raw data is
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])


# Ensure TransactionStartTime is datetime
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

# Snapshot date: 1 day after last transaction
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

# Calculate RFM features
rfm_df = df.groupby('CustomerId').agg(
    Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
    Frequency=('TransactionId', 'count'),
    Monetary=('Value', 'sum')
).reset_index()


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)


cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_summary = cluster_summary.sort_values(by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True])
high_risk_cluster_label = cluster_summary.index[0]

rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_label).astype(int)


df = df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# Load raw transactions
df = pd.read_csv('../data/raw/data.csv')
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

# Assume RFM and is_high_risk were calculated
# rfm_df must contain 'CustomerId' and 'is_high_risk'
df = df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

def engineer_proxy_target(df):
    # Do your processing
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm_df = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Value', 'sum')
    ).reset_index()

    # (Scaling + Clustering...)
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(
        by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True])
    high_risk_cluster_label = cluster_summary.index[0]

    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_label).astype(int)

    # Merge with original df
    df = df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

    return df  # ✅ Correct place for return
