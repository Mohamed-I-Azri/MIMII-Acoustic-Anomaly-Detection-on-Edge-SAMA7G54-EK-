import os
import numpy as np
import tensorflow as tf
from data_loader import get_file_lists, extract_features_hitachi, IDS

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

INPUT_DIM  = 320
EPOCHS     = 50
BATCH_SIZE = 512
LR         = 0.001


def build_autoencoder():
    """Exact Hitachi architecture: 320->128->128->128->128->128->320"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(INPUT_DIM)               # linear output
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
    return model


def train_hitachi(train_files):
    """Extract all frame vectors from training files and train autoencoder."""
    X_list = [extract_features_hitachi(f) for f in train_files]
    X = np.vstack(X_list).astype(np.float32)   # (total_frames, 320)
    print(f"  Training frames: {X.shape[0]}")

    model = build_autoencoder()
    model.fit(X, X,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              shuffle=True,
              verbose=1)
    return model


def score_file_hitachi(model, path):
    """Mean MSE reconstruction error over all frames of a file."""
    frames = extract_features_hitachi(path).astype(np.float32)  # (n_vecs, 320)
    pred   = model.predict(frames, verbose=0)
    per_frame_mse = np.mean((frames - pred) ** 2, axis=1)       # (n_vecs,)
    return float(np.mean(per_frame_mse))


def convert_to_tflite(model, mid):
    """Convert Keras model to TFLite flatbuffer."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_path = os.path.join(MODELS_DIR, f"hitachi_{mid}.tflite")
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  TFLite model saved: {out_path}")


if __name__ == "__main__":
    for mid in IDS:
        print(f"\nTraining Hitachi — {mid}")
        train_files, _, _ = get_file_lists(mid)
        model = train_hitachi(train_files)

        # Save Keras model
        model.save(os.path.join(MODELS_DIR, f"hitachi_{mid}.h5"))

        # Convert to TFLite
        convert_to_tflite(model, mid)

    print("\nHitachi training complete.")
