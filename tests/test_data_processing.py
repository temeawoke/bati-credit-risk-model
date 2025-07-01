# Test imputation logic
from src.feature_engineering import impute_missing

def test_impute_missing():
    df = pd.DataFrame({"val": [1.0, None, 3.0]})
    df = impute_missing(df, method="mean")
    assert df["val"].isna().sum() == 0

# Test aggregate feature computation
from src.data.feature_engineering import compute_transaction_stats

def test_compute_transaction_stats():
    df = pd.DataFrame({"CustomerId": [1, 1, 2], "Amount": [10, 20, 30]})
    agg = compute_transaction_stats(df)
    assert "TotalTransactionAmount" in agg.columns