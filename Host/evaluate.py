import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

from data_loader import get_file_lists, extract_features_gaussian, extract_features_hitachi, IDS
from train_gaussian import mahalanobis_score

MODELS_DIR = "models"


def evaluate_gaussian(mid):
    mu        = np.load(os.path.join(MODELS_DIR, f"gaussian_mu_{mid}.npy"))
    sigma_inv = np.load(os.path.join(MODELS_DIR, f"gaussian_sigma_inv_{mid}.npy"))

    _, test_norm, test_abn = get_file_lists(mid)
    scores, labels, times = [], [], []

    for path in test_norm:
        t0 = time.perf_counter()
        feat = extract_features_gaussian(path)
        s    = mahalanobis_score(feat, mu, sigma_inv)
        times.append((time.perf_counter() - t0) * 1000)
        scores.append(s); labels.append(0)

    for path in test_abn:
        t0 = time.perf_counter()
        feat = extract_features_gaussian(path)
        s    = mahalanobis_score(feat, mu, sigma_inv)
        times.append((time.perf_counter() - t0) * 1000)
        scores.append(s); labels.append(1)

    auc = roc_auc_score(labels, scores)
    return auc, np.mean(times)


def evaluate_hitachi(mid):
    model_path = os.path.join(MODELS_DIR, f"hitachi_{mid}.h5")
    model = tf.keras.models.load_model(model_path)

    _, test_norm, test_abn = get_file_lists(mid)
    scores, labels, times = [], [], []

    for path in test_norm:
        t0     = time.perf_counter()
        frames = extract_features_hitachi(path).astype(np.float32)
        pred   = model.predict(frames, verbose=0)
        s      = float(np.mean(np.mean((frames - pred) ** 2, axis=1)))
        times.append((time.perf_counter() - t0) * 1000)
        scores.append(s); labels.append(0)

    for path in test_abn:
        t0     = time.perf_counter()
        frames = extract_features_hitachi(path).astype(np.float32)
        pred   = model.predict(frames, verbose=0)
        s      = float(np.mean(np.mean((frames - pred) ** 2, axis=1)))
        times.append((time.perf_counter() - t0) * 1000)
        scores.append(s); labels.append(1)

    auc = roc_auc_score(labels, scores)
    return auc, np.mean(times)


if __name__ == "__main__":
    print(f"\n{'ID':<8} {'Gaussian AUC':>14} {'G time(ms)':>11} {'Hitachi AUC':>13} {'H time(ms)':>11}")
    print("-" * 62)

    g_aucs, h_aucs = [], []
    for mid in IDS:
        g_auc, g_t = evaluate_gaussian(mid)
        h_auc, h_t = evaluate_hitachi(mid)
        g_aucs.append(g_auc); h_aucs.append(h_auc)
        print(f"{mid:<8} {g_auc:>14.4f} {g_t:>11.2f} {h_auc:>13.4f} {h_t:>11.2f}")

    print("-" * 62)
    print(f"{'Macro avg':<8} {np.mean(g_aucs):>14.4f} {'':>11} {np.mean(h_aucs):>13.4f}")
