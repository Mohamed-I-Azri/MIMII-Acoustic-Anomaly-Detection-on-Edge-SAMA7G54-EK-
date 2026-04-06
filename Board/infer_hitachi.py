import os, time, struct
import numpy as np
import tflite_runtime.interpreter as tflite

DATA_ROOT  = "/data/mimii_test"
MODELS_DIR = "/data"
IDS        = ["id_00", "id_02", "id_04", "id_06"]
SR         = 16000
N_FFT      = 1024
HOP_LENGTH = 512
N_MELS     = 64
N_FRAMES   = 5

# ── Mel filterbank ─────────────────────────────────────────────────────────────
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
    """Stacked log-mel frames: (n_vecs, 320) — matches Hitachi baseline."""
    n_stft = max(1, (len(y) - N_FFT) // HOP_LENGTH + 1)

    idx    = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_stft)[:, None]
    idx    = np.clip(idx, 0, len(y) - 1)
    frames = y[idx] * WINDOW                                    # (n_stft, N_FFT)

    spec   = np.fft.rfft(frames, n=N_FFT)
    power  = (spec.real**2 + spec.imag**2).astype(np.float32)  # (n_stft, n_freqs)
    mel    = power @ MEL_FB.T                                   # (n_stft, n_mels)
    logmel = 10.0 * np.log10(mel + 2.22e-16)                   # dB, (n_stft, n_mels)
    logmel = logmel.T                                           # (n_mels, n_stft)

    T      = logmel.shape[1]
    n_vecs = T - N_FRAMES + 1
    if n_vecs < 1:
        # Pad with silence so we can still produce one valid 320-dim vector
        pad    = np.full((N_MELS, N_FRAMES - T), -80.0, dtype=np.float32)
        logmel = np.concatenate([logmel, pad], axis=1)
        n_vecs = 1

    # Stack 5 consecutive frames column-major (matches MATLAB flatten)
    vecs = np.zeros((n_vecs, N_MELS * N_FRAMES), dtype=np.float32)
    for t in range(n_vecs):
        vecs[t] = logmel[:, t:t + N_FRAMES].flatten(order='F')
    return vecs   # (n_vecs, 320)


def score_file(interpreter, frames):
    """Mean MSE reconstruction error over all frames."""
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    mse = np.zeros(len(frames), dtype=np.float32)
    for i, frame in enumerate(frames):
        interpreter.set_tensor(inp_idx, frame.reshape(1, -1))
        interpreter.invoke()
        pred   = interpreter.get_tensor(out_idx)[0]
        mse[i] = np.mean((frame - pred) ** 2)
    return float(mse.mean())


def roc_auc(labels, scores):
    labels = np.array(labels); scores = np.array(scores)
    idx    = np.argsort(scores)[::-1]; labels = labels[idx]
    n_pos  = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return float('nan')
    auc = float(abs(np.trapz(np.cumsum(labels)/n_pos,
                              np.cumsum(1-labels)/n_neg)))
    return max(auc, 1 - auc)


if __name__ == "__main__":
    print(f"\n{'ID':<8} {'AUC':>8} {'ms/file':>10}")
    print("-" * 30)
    aucs = []
    for mid in IDS:
        interp = tflite.Interpreter(
            model_path=f"{MODELS_DIR}/hitachi_{mid}.tflite")
        interp.allocate_tensors()

        normal   = sorted([f"{DATA_ROOT}/{mid}/normal/{f}"
                           for f in os.listdir(f"{DATA_ROOT}/{mid}/normal/")])
        abnormal = sorted([f"{DATA_ROOT}/{mid}/abnormal/{f}"
                           for f in os.listdir(f"{DATA_ROOT}/{mid}/abnormal/")])

        scores, labels, times = [], [], []
        for path in normal + abnormal:
            y      = load_wav(path)
            t0     = time.perf_counter()
            frames = extract_features(y)
            s      = score_file(interp, frames)
            times.append((time.perf_counter() - t0) * 1000)
            scores.append(s)
            labels.append(1 if "abnormal" in path else 0)

        auc = roc_auc(labels, scores)
        aucs.append(auc)
        print(f"{mid:<8} {auc:>8.4f} {np.mean(times):>10.1f}")

    print("-" * 30)
    print(f"{'Macro':<8} {np.mean(aucs):>8.4f}")
