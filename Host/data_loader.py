import os
import numpy as np
import librosa

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_ROOT   = r"C:\Users\aazry\Desktop\6_dB_fan\fan"
IDS         = ["id_00", "id_02", "id_04", "id_06"]
SR          = 16000
N_FFT       = 1024
HOP_LENGTH  = 512
N_MELS      = 64
N_FRAMES    = 5        # for Hitachi stacked frames
TRAIN_RATIO = 0.5      # 50/50 split — matches MATLAB code
RANDOM_SEED = 0
# ──────────────────────────────────────────────────────────────────────────────


def load_wav(path):
    """Load WAV, channel 1 only, resample to 16kHz."""
    y, sr = librosa.load(path, sr=SR, mono=False)
    if y.ndim > 1:
        y = y[0]   # channel 1 only — matches Hitachi baseline
    return y


def extract_features_gaussian(path):
    """
    68-dim feature vector per file (matches MATLAB Gaussian model):
      - 64 log-mel band means  (natural log)
      - 4 RMS stats: mean, std, max, min
    Returns shape (68,).
    """
    y = load_wav(path)
    y = y / (np.max(np.abs(y)) + 1e-10)   # normalize

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    logmel   = np.log(mel + 1e-6)          # natural log — matches MATLAB
    mel_feat = logmel.mean(axis=1)         # (64,)

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms_feat = np.array([rms.mean(), rms.std(), rms.max(), rms.min()])  # (4,)

    return np.concatenate([mel_feat, rms_feat])  # (68,)


def extract_features_hitachi(path):
    """
    Stacked log-mel frames per file (matches MATLAB Hitachi model):
      - 10*log10 power mel spectrogram
      - Stack 5 consecutive frames → 320-dim vectors
    Returns shape (n_vectors, 320).
    """
    y = load_wav(path)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0
    )
    logmel = 10.0 * np.log10(mel + 2.22e-16)  # dB scale — matches MATLAB

    T       = logmel.shape[1]
    n_vecs  = T - N_FRAMES + 1
    if n_vecs < 1:
        return logmel.T.reshape(1, -1)

    frames = np.zeros((n_vecs, N_MELS * N_FRAMES))
    for t in range(n_vecs):
        frames[t] = logmel[:, t:t + N_FRAMES].flatten(order='F')  # column-major matches MATLAB

    return frames  # (n_vectors, 320)


def get_file_lists(machine_id):
    """
    Returns train/test file lists for a given ID.
    50/50 split on normal files, all abnormal files go to test.
    """
    normal_dir   = os.path.join(DATA_ROOT, machine_id, "normal")
    abnormal_dir = os.path.join(DATA_ROOT, machine_id, "abnormal")

    normal_files   = sorted([os.path.join(normal_dir, f)
                              for f in os.listdir(normal_dir) if f.endswith('.wav')])
    abnormal_files = sorted([os.path.join(abnormal_dir, f)
                              for f in os.listdir(abnormal_dir) if f.endswith('.wav')])

    rng   = np.random.default_rng(RANDOM_SEED)
    perm  = rng.permutation(len(normal_files))
    n_tr  = int(len(normal_files) * TRAIN_RATIO)

    train_files     = [normal_files[i] for i in perm[:n_tr]]
    test_norm_files = [normal_files[i] for i in perm[n_tr:]]

    return train_files, test_norm_files, abnormal_files


if __name__ == "__main__":
    # Quick sanity check
    train, test_norm, test_abn = get_file_lists("id_00")
    print(f"id_00 — train: {len(train)}, test normal: {len(test_norm)}, test abnormal: {len(test_abn)}")

    feat_g = extract_features_gaussian(train[0])
    print(f"Gaussian feature shape: {feat_g.shape}")   # expect (68,)

    feat_h = extract_features_hitachi(train[0])
    print(f"Hitachi feature shape:  {feat_h.shape}")   # expect (n_vectors, 320)
