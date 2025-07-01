{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "be3b8129",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "import warnings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b02283cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Suppress KMeans deprecation warning for n_init\n",
    "warnings.filterwarnings(\"ignore\", category=FutureWarning, module=\"sklearn.cluster._kmeans\")\n",
    "\n",
    "def engineer_proxy_target(df_raw: pd.DataFrame) -> pd.DataFrame:\n",
    "    \"\"\"\n",
    "    Engineers a proxy target variable 'is_high_risk' by calculating RFM metrics,\n",
    "    clustering customers, and identifying the least engaged (high-risk) segment.\n",
    "\n",
    "    Args:\n",
    "        df_raw (pd.DataFrame): The raw DataFrame loaded from data.csv.\n",
    "                               Must contain 'CustomerId', 'TransactionStartTime', and 'Value'.\n",
    "\n",
    "    Returns:\n",
    "        pd.DataFrame: The DataFrame with the new 'is_high_risk' column merged in.\n",
    "                      This DataFrame is ready to be passed to the main feature engineering\n",
    "                      pipeline (e.g., preprocess_data from Task 3).\n",
    "    \"\"\"\n",
    "\n",
    "    df = df_raw.copy()\n",
    "\n",
    "    # Ensure TransactionStartTime is datetime\n",
    "    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7c2255f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.read_csv('../data/raw/data.csv')  # replace with your correct path\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "39d78b75",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ensure datetime format\n",
    "df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])\n",
    "# Now calculate snapshot_date\n",
    "snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "850ae4f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 1. Calculate RFM Metrics ---\n",
    "    # Define a snapshot date (e.g., one day after the latest transaction in the dataset)\n",
    "\n",
    "\n",
    "    # Calculate RFM for each CustomerId\n",
    "rfm_df = df.groupby('CustomerId').agg(\n",
    "        Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),\n",
    "        Frequency=('TransactionId', 'count'), # Count of transactions\n",
    "        Monetary=('Value', 'sum') # Sum of transaction values\n",
    "    ).reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a6432563",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "datetime64[ns, UTC]\n"
     ]
    }
   ],
   "source": [
    "print(df['TransactionStartTime'].dtype)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "3a751857",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 2. Cluster Customers using K-Means ---\n",
    "    # Select RFM features for clustering\n",
    "rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]\n",
    "\n",
    "    # Pre-process (scale) RFM features\n",
    "scaler = StandardScaler()\n",
    "rfm_scaled = scaler.fit_transform(rfm_features)\n",
    "\n",
    "    # Apply K-Means clustering\n",
    "    # Using n_init='auto' for modern sklearn versions\n",
    "kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')\n",
    "rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a1411291",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cluster Summary (Mean RFM Values):\n",
      " | Cluster   | Recency   | Frequency   | Monetary   |\n",
      "|:----------|:----------|:------------|:-----------|\n",
      "| 0         | 61.7157   | 7.6892      | 89251.7    |\n",
      "| 1         | 29        | 4091        | 1.049e+08  |\n",
      "| 2         | 12.6353   | 34.925      | 309685     |\n",
      "\n",
      "Identified High-Risk Cluster Label: 0\n"
     ]
    }
   ],
   "source": [
    "# --- 3. Define and Assign the \"High-Risk\" Label ---\n",
    "    # Analyze cluster centroids to identify the high-risk cluster.\n",
    "    # High-risk customers are typically characterized by:\n",
    "    # - High Recency (haven't transacted recently)\n",
    "    # - Low Frequency (don't transact often)\n",
    "    # - Low Monetary (low total spending)\n",
    "\n",
    "    # Get cluster centroids (in scaled space)\n",
    "centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=rfm_features.columns)\n",
    "\n",
    "    # To interpret, we can inverse transform centroids or just look at relative values.\n",
    "    # For simplicity, let's look at the scaled centroids directly to find the \"least engaged\"\n",
    "    # Least engaged = highest Recency_scaled, lowest Frequency_scaled, lowest Monetary_scaled\n",
    "\n",
    "    # Calculate a 'risk score' for each centroid (lower score = higher risk)\n",
    "    # Example: (Recency_scaled) - (Frequency_scaled) - (Monetary_scaled)\n",
    "    # Or simply sort by Recency (desc), then Frequency (asc), then Monetary (asc)\n",
    "    # For a robust approach, we'll identify the cluster with the highest mean Recency\n",
    "    # and lowest mean Frequency/Monetary from the *original* RFM values.\n",
    "\n",
    "cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(\n",
    "        by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]\n",
    "    )\n",
    "print(\"Cluster Summary (Mean RFM Values):\\n\", cluster_summary.to_markdown(numalign=\"left\", stralign=\"left\"))\n",
    "\n",
    "    # The first cluster in the sorted summary is likely the high-risk one\n",
    "high_risk_cluster_label = cluster_summary.index[0]\n",
    "print(f\"\\nIdentified High-Risk Cluster Label: {high_risk_cluster_label}\")\n",
    "\n",
    "    # Create the 'is_high_risk' column\n",
    "rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_label).astype(int)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9769b6d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.read_csv('../data/raw/data.csv')  # or wherever your raw data is\n",
    "df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "0550171f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',\n",
      "       'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',\n",
      "       'ProductCategory', 'ChannelId', 'Amount', 'Value',\n",
      "       'TransactionStartTime', 'PricingStrategy', 'FraudResult'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(df.columns)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "205fec0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ensure TransactionStartTime is datetime\n",
    "df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])\n",
    "\n",
    "# Snapshot date: 1 day after last transaction\n",
    "snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)\n",
    "\n",
    "# Calculate RFM features\n",
    "rfm_df = df.groupby('CustomerId').agg(\n",
    "    Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),\n",
    "    Frequency=('TransactionId', 'count'),\n",
    "    Monetary=('Value', 'sum')\n",
    ").reset_index()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ccb4c92b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "\n",
    "scaler = StandardScaler()\n",
    "rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])\n",
    "\n",
    "kmeans = KMeans(n_clusters=4, random_state=42)\n",
    "rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7a986da6",
   "metadata": {},
   "outputs": [],
   "source": [
    "cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()\n",
    "cluster_summary = cluster_summary.sort_values(by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True])\n",
    "high_risk_cluster_label = cluster_summary.index[0]\n",
    "\n",
    "rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_label).astype(int)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "265fcf27",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "67a1f142",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load raw transactions\n",
    "df = pd.read_csv('../data/raw/data.csv')\n",
    "df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])\n",
    "\n",
    "# Assume RFM and is_high_risk were calculated\n",
    "# rfm_df must contain 'CustomerId' and 'is_high_risk'\n",
    "df = df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d035789e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def engineer_proxy_target(df):\n",
    "    # Do your processing\n",
    "    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])\n",
    "\n",
    "    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)\n",
    "\n",
    "    rfm_df = df.groupby('CustomerId').agg(\n",
    "        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),\n",
    "        Frequency=('TransactionId', 'count'),\n",
    "        Monetary=('Value', 'sum')\n",
    "    ).reset_index()\n",
    "\n",
    "    # (Scaling + Clustering...)\n",
    "    from sklearn.preprocessing import StandardScaler\n",
    "    from sklearn.cluster import KMeans\n",
    "\n",
    "    scaler = StandardScaler()\n",
    "    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])\n",
    "\n",
    "    kmeans = KMeans(n_clusters=4, random_state=42)\n",
    "    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)\n",
    "\n",
    "    cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(\n",
    "        by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True])\n",
    "    high_risk_cluster_label = cluster_summary.index[0]\n",
    "\n",
    "    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_label).astype(int)\n",
    "\n",
    "    # Merge with original df\n",
    "    df = df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')\n",
    "\n",
    "    return df  # ✅ Correct place for return\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
