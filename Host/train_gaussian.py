import os
import numpy as np
from data_loader import get_file_lists, extract_features_gaussian, IDS

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_gaussian(train_files):
    """
    Fit multivariate Gaussian on normal training features.
    Returns mu (68,) and sigma_inv (68,68).
    """
    X = np.stack([extract_features_gaussian(f) for f in train_files])  # (N, 68)
    mu    = X.mean(axis=0)                                              # (68,)
    sigma = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])                   # (68,68)
    sigma = (sigma + sigma.T) / 2                                       # ensure symmetry
    sigma_inv = np.linalg.inv(sigma)
    return mu, sigma_inv


def mahalanobis_score(feat, mu, sigma_inv):
    d = feat - mu
    return float(d @ sigma_inv @ d)


if __name__ == "__main__":
    for mid in IDS:
        print(f"\nTraining Gaussian — {mid}")
        train_files, _, _ = get_file_lists(mid)
        mu, sigma_inv = train_gaussian(train_files)

        np.save(os.path.join(MODELS_DIR, f"gaussian_mu_{mid}.npy"), mu)
        np.save(os.path.join(MODELS_DIR, f"gaussian_sigma_inv_{mid}.npy"), sigma_inv)
        print(f"  Saved mu and sigma_inv for {mid}")

    print("\nGaussian training complete.")
