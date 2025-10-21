import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from Preprocessing import final_dataset

# ==================== STEP 7: Helper Functions for GMM Binarization ====================

def log_transform(values):
    """
    Apply log1p (log(1 + x)) transformation to:
    - Stabilize variance
    - Reduce skewness in expression values
    - Make the data more normally distributed for GMM fitting
    """
    return np.log1p(values)


def gmm_threshold(values):
    """
    Fit a 2-component Gaussian Mixture Model (GMM) to a single gene’s values
    and determine a threshold separating low vs. high expression levels.

    Process:
    - Assume two distributions: low expression (component 1) and high expression (component 2)
    - Fit GMM to the data and find the intersection point where the probability is 0.5
    - Return that point as the binarization threshold
    """
    vals = values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(vals)

    means = gmm.means_.flatten()
    order = np.argsort(means)  # Sort components from low to high expression

    # Generate a smooth range of values between min and max observed values
    xs = np.linspace(vals.min(), vals.max(), 1000).reshape(-1, 1)

    # Probability of being in the "high expression" component
    probs = gmm.predict_proba(xs)[:, order[1]]

    # Find the threshold where probability ≈ 0.5
    thr = xs[np.argmin(np.abs(probs - 0.5))][0]
    return float(thr)


def binarize_gene(values):
    """
    Binarize one gene’s expression values using the GMM-based threshold.

    Steps:
    1. Log-transform the expression values
    2. Compute threshold using GMM
    3. Convert values > threshold to 1 (high expression), else 0 (low expression)
    4. If GMM fails to converge, use median as fallback threshold
    """
    vals = log_transform(values)
    try:
        thr = gmm_threshold(vals)
    except Exception:
        thr = np.median(vals)
    binary = (vals > thr).astype(int)
    return binary, thr


def binarize_dataframe(df):
    """
    Apply GMM binarization to all genes (columns) in parallel.

    - Uses joblib for parallelization → faster for large datasets
    - Returns:
        - binarized DataFrame with 0/1 values
        - Series of per-gene thresholds
    """
    results = Parallel(n_jobs=-1)(
        delayed(binarize_gene)(df.iloc[:, i].values) for i in range(df.shape[1])
    )

    binarized = pd.DataFrame(
        np.column_stack([r[0] for r in results]),
        index=df.index,
        columns=df.columns,
    )

    thresholds = pd.Series(
        [r[1] for r in results],
        index=df.columns,
        name="threshold"
    )

    return binarized, thresholds

# ==================== STEP 7A: Apply GMM Binarization ====================
# - Separate gene expression features and labels
# - Apply GMM binarization to convert continuous values into binary states (0/1)
# - Save binarized dataset for downstream analysis

X = final_dataset.drop(columns=["Label"])  # Features (gene expression)
y = final_dataset["Label"]                 # Target labels (cancer types)

print("\nBinarizing data...")
binarized_X, thresholds = binarize_dataframe(X)

# Merge binarized features with labels
binarized_df = pd.concat([binarized_X, y], axis=1)

# Save binarized data
binarized_df.to_csv("binarized_data.csv", index=False)

# ==================== STEP 7B: Summary ====================
# - Print shape and preview to verify the binarization result
print(f"Binarized data shape: {binarized_df.shape}")
print(binarized_df.head())
