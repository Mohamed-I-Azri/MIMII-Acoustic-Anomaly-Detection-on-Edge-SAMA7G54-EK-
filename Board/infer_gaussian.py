import os, time, struct
import numpy as np

DATA_ROOT  = "/data/mimii_test"
MODELS_DIR = "/data"
IDS        = ["id_00", "id_02", "id_04", "id_06"]
SR         = 16000
N_FFT      = 1024
HOP_LENGTH = 512
N_MELS     = 64

# ── Mel filterbank (built once) ───────────────────────────────────────────────
def _build_mel_fb():
    def hz2mel(h): return 2595.0 * np.log10(1.0 + h / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    n_freqs  = N_FFT // 2 + 1
    mel_pts  = np.linspace(hz2mel(0), hz2mel(SR / 2), N_MELS + 2)
    hz_pts   = mel2hz(mel_pts)
    bin_pts  = np.floor((N_FFT + 1) * hz_pts / SR).astype(int)
    fb = np.zeros((N_MELS, n_freqs), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        fl, fc, fr = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(fl, fc):
            if fc > fl: fb[m-1, k] = (k - fl) / (fc - fl)
        for k in range(fc, fr):
            if fr > fc: fb[m-1, k] = (fr - k) / (fr - fc)
    return fb

MEL_FB = _build_mel_fb()
WINDOW = np.hanning(N_FFT).astype(np.float32)


def load_wav(path):
    with open(path, 'rb') as f:
        data = f.read()
    pos = 12
    n_channels = sampwidth = 0
    audio_data = None
    while pos < len(data) - 8:
        chunk_id   = data[pos:pos+4]
        chunk_size = struct.unpack_from('<I', data, pos+4)[0]
        pos += 8
        if chunk_id == b'fmt ':
            n_channels = struct.unpack_from('<H', data, pos+2)[0]
            sampwidth  = struct.unpack_from('<H', data, pos+14)[0] // 8
        elif chunk_id == b'data':
            audio_data = data[pos:pos+chunk_size]
        pos += chunk_size + (chunk_size % 2)
    arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return arr.reshape(-1, n_channels)[:, 0]


def extract_features(y):
    """68-dim feature: 64 log-mel means + 4 RMS stats."""
    y = y / (np.max(np.abs(y)) + 1e-10)
    n_frames = max(1, (len(y) - N_FFT) // HOP_LENGTH + 1)

    # Build frame matrix efficiently
    idx    = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    idx    = np.clip(idx, 0, len(y) - 1)
    frames = y[idx] * WINDOW                               # (n_frames, N_FFT)

    # Power spectrum → mel → log
    spec   = np.fft.rfft(frames, n=N_FFT)                 # (n_frames, n_freqs)
    power  = (spec.real**2 + spec.imag**2).astype(np.float32)  # (n_frames, n_freqs)
    mel    = power @ MEL_FB.T                              # (n_frames, n_mels)
    logmel = np.log(mel + 1e-6)
    mel_feat = logmel.mean(axis=0)                         # (64,)

    # RMS per frame
    rms = np.sqrt((frames**2).mean(axis=1))                # (n_frames,)
    rms_feat = np.array([rms.mean(), rms.std(), rms.max(), rms.min()])

    return np.concatenate([mel_feat, rms_feat]).astype(np.float32)


def roc_auc(labels, scores):
    labels = np.array(labels); scores = np.array(scores)
    idx    = np.argsort(scores)[::-1]; labels = labels[idx]
    n_pos  = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return float('nan')
    auc = float(abs(np.trapz(np.cumsum(labels)/n_pos,
                              np.cumsum(1-labels)/n_neg)))
    return max(auc)   # always return > 0.5 equivalent


if __name__ == "__main__":
    print(f"\n{'ID':<8} {'AUC':>8} {'ms/file':>10}")
    print("-" * 30)
    aucs = []
    for mid in IDS:
        mu        = np.load(f"{MODELS_DIR}/gaussian_mu_{mid}.npy")
        sigma_inv = np.load(f"{MODELS_DIR}/gaussian_sigma_inv_{mid}.npy")

        normal   = sorted([f"{DATA_ROOT}/{mid}/normal/{f}"
                           for f in os.listdir(f"{DATA_ROOT}/{mid}/normal/")])
        abnormal = sorted([f"{DATA_ROOT}/{mid}/abnormal/{f}"
                           for f in os.listdir(f"{DATA_ROOT}/{mid}/abnormal/")])

        scores, labels, times = [], [], []
        for path in normal + abnormal:
            y  = load_wav(path)
            t0 = time.perf_counter()
            feat = extract_features(y)
            d    = feat - mu
            s    = float(d @ sigma_inv @ d)
            times.append((time.perf_counter() - t0) * 1000)
            scores.append(s)
            labels.append(1 if "abnormal" in path else 0)

        auc = roc_auc(labels, scores)
        aucs.append(auc)
        print(f"{mid:<8} {auc:>8.4f} {np.mean(times):>10.1f}")

    print("-" * 30)
    print(f"{'Macro':<8} {np.mean(aucs):>8.4f}")
