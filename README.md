# MIMII Anomaly Detection

Acoustic anomaly detection for industrial fans using the MIMII dataset, deployed on the SAMA7G54-EK embedded board. Two models are compared: a Gaussian (Mahalanobis distance) classifier and a Hitachi-style autoencoder converted to TFLite.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [PC Setup (Windows)](#pc-setup-windows)
- [Training](#training)
- [Board Setup (SAMA7G54-EK)](#board-setup-sama7g54-ek)
- [Transfer and Inference](#transfer-and-inference)
- [Enable On-Board Microphones](#enable-on-board-microphones)
- [Results](#results)
- [Common Issues](#common-issues)
- [Quick Reference](#quick-reference)

---

## Project Overview

This project implements and evaluates two unsupervised anomaly detection approaches on the MIMII fan/6dB subset:

- **Gaussian model** — fits a multivariate Gaussian on log-mel and RMS features extracted from normal audio; anomaly score is the Mahalanobis distance. Fast and lightweight.
- **Hitachi autoencoder** — trains a dense autoencoder on spectral features; anomaly score is reconstruction error. Higher accuracy at the cost of inference time.

Both models are trained on a Windows PC using Python 3.8 and TensorFlow 2.8, then deployed to the SAMA7G54-EK (ARM Cortex-A7) running Linux4SAM, using numpy and tflite-runtime for inference.

---

## Repository Structure

```
mimii-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
│
├── Host/
│   ├── data_loader.py          # Feature extraction (librosa-based)
│   ├── train_gaussian.py       # Fit Gaussian model -> .npy files
│   ├── train_hitachi.py        # Train autoencoder -> .h5 + .tflite
│   └── evaluate.py             # AUC comparison on PC
│
├── Board/
│   ├── infer_gaussian.py       # Pure numpy Gaussian inference
│   └── infer_hitachi.py        # TFLite Hitachi inference
│
├── models/                     # Trained artifacts (git-ignored)
│   ├── gaussian_mu_id_00.npy
│   ├── gaussian_sigma_inv_id_00.npy
│   ├── ...
│   ├── hitachi_id_00.tflite
│   └── ...
│
├── numpy-1.24.4-cp38-cp38-linux_armv7l.whl
├── libopenblas.so.0
├── libgfortran.so.5
├── libgcc_s.so.1
├── libgomp.so.1
└── libstdc++.so.6
```

---

## Requirements

**Python version: 3.8.10 (mandatory — must match the board)**

```
numpy
tensorflow==2.8.0
soundfile
matplotlib
protobuf==3.20.3
```

> `protobuf==3.20.3` is required. Newer versions cause a Descriptors error on TensorFlow 2.8 import.

---

## PC Setup (Windows)

### Step 1 — Install pyenv-win

Open PowerShell as Administrator:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-WebRequest -UseBasicParsing -Uri `
  "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" `
  -OutFile "./install-pyenv-win.ps1"
& "./install-pyenv-win.ps1"
```

Close and reopen PowerShell. If `pyenv` is not found, add it to PATH manually:

```powershell
[System.Environment]::SetEnvironmentVariable("PYENV", "$env:USERPROFILE\.pyenv\pyenv-win", "User")
$path = [System.Environment]::GetEnvironmentVariable("PATH", "User")
[System.Environment]::SetEnvironmentVariable("PATH",
  "$env:USERPROFILE\.pyenv\pyenv-win\bin;$env:USERPROFILE\.pyenv\pyenv-win\shims;$path", "User")
```

Verify:

```powershell
pyenv --version
```

### Step 2 — Install Python 3.8.10

```powershell
pyenv install 3.8.10
```

> Do not use Python 3.9, 3.10, 3.11, or 3.12. The version must match the board.

### Step 3 — Create Project and Virtual Environment

```powershell
cd C:\Users\<your_username>
mkdir mimii_project
cd mimii_project
pyenv local 3.8.10
C:\Users\<your_username>\.pyenv\pyenv-win\versions\3.8.10\python.exe -m venv venv
venv\Scripts\activate
python --version   # must show Python 3.8.10
```

> If `python --version` shows 3.12, the venv picked up the global Python. Deactivate and use the full pyenv path shown above.

### Step 4 — Install Packages

```powershell
pip install numpy librosa scikit-learn tensorflow==2.8.0 soundfile matplotlib
pip install protobuf==3.20.3
```

Verify:

```powershell
python -c "import numpy, librosa, sklearn, tensorflow, soundfile, matplotlib; print('all good')"
```

### Step 5 — Dataset

Download the MIMII dataset from [Zenodo](https://zenodo.org). Only the `fan/6dB` subset is needed. Place it at:

```
C:\Users\<your_username>\Desktop\6_dB_fan\fan\
```

Expected structure:

```
fan/
├── id_00/
│   ├── normal/     (~1000 WAV files)
│   └── abnormal/   (~350-400 WAV files)
├── id_02/
├── id_04/
└── id_06/
```

### Step 6 — Configure Data Loader

Open `Host/data_loader.py` and set:

```python
DATA_ROOT = r"C:\Users\<your_username>\Desktop\6_dB_fan\fan"
```

### Step 7 — Verify Data Loader

```powershell
python Host/data_loader.py
```

Expected output:

```
id_00 -- train: 505, test normal: 506, test abnormal: 407
Gaussian feature shape: (68,)
Hitachi feature shape: (309, 320)
```

---

## Training

### Step 8 — Train Gaussian Model

```powershell
python Host/train_gaussian.py
```

Produces 8 `.npy` files in `models/` (mu + sigma_inv per machine ID). Completes in under 1 minute.

### Step 9 — Train Hitachi Autoencoder

```powershell
python Host/train_hitachi.py
```

Produces 4 `.h5` files and 4 `.tflite` files in `models/`. Takes 10-20 minutes (50 epochs x 4 IDs).

> The `.tflite` files are transferred to the board. The `.h5` files are kept as backup for re-conversion.

### Step 10 — Evaluate on PC

```powershell
python Host/evaluate.py
```

Expected output:

```
ID        Gaussian AUC   G time(ms)   Hitachi AUC   H time(ms)
--------------------------------------------------------------
id_00     0.9348         17.32        0.7712         37.16
id_02     0.9977         17.36        0.9882         42.68
id_04     0.9767         20.71        0.9412         42.49
id_06     0.9999         19.60        0.9947         42.89
--------------------------------------------------------------
Macro avg 0.9773                      0.9238
```

---

## Board Setup (SAMA7G54-EK)

### Step 11 — Flash SD Card

Download the Linux4SAM 2022.04 headless image for SAMA7G5-EK:

```
SAMA7G5-EK-linux4sam-poky-sama7g5ek-headless-2022.04.img.bz2
```

From: https://www.linux4sam.org — SAMA7G5-EK demo images — 2022.04 archive

Flash to SD card using [Balena Etcher](https://etcher.balena.io). Etcher handles `.bz2` directly.

### Step 12 — Create /data Partition

Before inserting the SD card into the board, use DiskGenius (Windows) to create a new ext4 partition in the unallocated space (28+ GB). This becomes the `/data` working partition.

> Do NOT resize the existing root partition (partition 2, ~533MB). Only create a NEW partition in unallocated space. Resizing root will break the bootloader.

### Step 13 — First Boot

Insert SD card, connect Ethernet, connect USB serial. Open PuTTY:

- Connection type: Serial
- Serial line: COMx (check Device Manager for J-Link CDC port)
- Speed: 115200

Power on. Login as `root` (no password). Set date and get network:

```sh
date -s "2026-03-24 09:00:00"
udhcpc -i eth0
ping -c 2 google.com
```

> The board clock resets to 2018 on every reboot. Always run `date -s` before any pip commands or SSL verification will fail.

### Step 14 — Mount /data Partition

```sh
mkdir -p /data
mount /dev/mmcblk1p3 /data
echo '/dev/mmcblk1p3 /data ext4 defaults 0 0' >> /etc/fstab
df -h /data   # should show ~27GB available
```

> Your SD card partition number may differ. Run `lsblk` to confirm the correct device name.

### Step 15 — Install tflite-runtime

```sh
pip3 install tflite-runtime==2.11.0 --no-deps
```

> Always use `--no-deps`. Without it, pip tries to build numpy from source, which fails (no gcc on board).

### Step 16 — Cross-Compile numpy on PC

On your PC (requires Docker Desktop with WSL2):

```powershell
docker run --rm --platform linux/arm/v7 -v C:\path\to\wheels:/wheels python:3.8-bullseye bash -c \
  "apt-get update -q && apt-get install -y -q gcc g++ gfortran libopenblas-dev && \
   pip wheel numpy --no-deps -w /wheels -q && \
   cp /usr/lib/arm-linux-gnueabihf/libopenblas*.so* /wheels/ && \
   cp /usr/lib/arm-linux-gnueabihf/libgfortran*.so* /wheels/ && \
   cp /usr/lib/gcc/arm-linux-gnueabihf/10/libgcc_s.so /wheels/ && \
   cp /usr/lib/arm-linux-gnueabihf/libgomp*.so* /wheels/ && \
   cp /usr/lib/arm-linux-gnueabihf/libstdc++*.so* /wheels/ && echo DONE"
```

> Use `python:3.8-bullseye` (glibc 2.31), not `python:3.8-slim` or `python:3.8-bookworm` (glibc 2.34). The board has glibc 2.31 and wheels built against newer glibc will fail to load.

### Step 17 — Setup SSH Key (Passwordless Transfer)

On PC:

```powershell
ssh-keygen -t ed25519 -f C:\Users\<user>\.ssh\sama7g54 -N ""
Get-Content C:\Users\<user>\.ssh\sama7g54.pub | ssh root@192.168.100.46 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

On board:

```sh
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

Test:

```powershell
ssh -i C:\Users\<user>\.ssh\sama7g54 -o IdentitiesOnly=yes root@192.168.100.46 "echo ok"
```

### Step 18 — Transfer numpy Wheel and Libraries

From PC PowerShell:

```powershell
scp -O -i C:\Users\<user>\.ssh\sama7g54 -o IdentitiesOnly=yes wheels\numpy-1.24.4-cp38-cp38-linux_armv7l.whl root@192.168.100.46:/data/
scp -O -i C:\Users\<user>\.ssh\sama7g54 -o IdentitiesOnly=yes wheels\libopenblas.so.0   root@192.168.100.46:/usr/lib/libopenblas.so.0
scp -O -i C:\Users\<user>\.ssh\sama7g54 -o IdentitiesOnly=yes wheels\libgfortran.so.5   root@192.168.100.46:/usr/lib/libgfortran.so.5
scp -O -i C:\Users\<user>\.ssh\sama7g54 -o IdentitiesOnly=yes wheels\libgcc_s.so        root@192.168.100.46:/usr/lib/libgcc_s.so.1
scp -O -i C:\Users\<user>\.ssh\sama7g54 -o IdentitiesOnly=yes wheels\libgomp.so.1       root@192.168.100.46:/usr/lib/libgomp.so.1
scp -O -i C:\Users\<user>\.ssh\sama7g54 -o IdentitiesOnly=yes wheels\libstdc++.so.6     root@192.168.100.46:/usr/lib/libstdc++.so.6
```

### Step 19 — Install numpy on Board

```sh
pip3 install /data/numpy-1.24.4-cp38-cp38-linux_armv7l.whl
ldconfig
```

Create gnueabi symlinks (the board Python expects gnueabi, but the wheel is gnueabihf):

```sh
find /usr/lib/python3.8/site-packages/numpy -name "*.cpython-38-arm-linux-gnueabihf.so" | while read f; do
  ln -sf "$f" "${f/gnueabihf/gnueabi}"
done
```

Verify:

```sh
python3 -c "import numpy, tflite_runtime; print('numpy:', numpy.__version__, '| tflite:', tflite_runtime.__version__)"
# Expected: numpy: 1.24.4 | tflite: 2.11.0
```

> If numpy imports with `No module named numpy.core._multiarray_umath`, the symlink step was missed or incomplete. Re-run the find/ln command.

---

## Transfer and Inference

### Step 20 — Transfer Model Files

```sh
# Gaussian parameters (8 files)
for id in id_00 id_02 id_04 id_06; do
  scp -O -i ~/.ssh/sama7g54 -o IdentitiesOnly=yes models/gaussian_mu_$id.npy        root@192.168.100.46:/data/
  scp -O -i ~/.ssh/sama7g54 -o IdentitiesOnly=yes models/gaussian_sigma_inv_$id.npy root@192.168.100.46:/data/
done

# TFLite models (4 files)
for id in id_00 id_02 id_04 id_06; do
  scp -O -i ~/.ssh/sama7g54 -o IdentitiesOnly=yes models/hitachi_$id.tflite root@192.168.100.46:/data/
done
```

### Step 21 — Transfer Inference Scripts

```sh
scp -O -i ~/.ssh/sama7g54 -o IdentitiesOnly=yes Board/infer_gaussian.py root@192.168.100.46:/data/
scp -O -i ~/.ssh/sama7g54 -o IdentitiesOnly=yes Board/infer_hitachi.py  root@192.168.100.46:/data/
```

### Step 22 — Transfer Test Audio Files

On PC, generate and zip test files:

```python
python -c "
from Host.data_loader import get_file_lists, IDS
import os, zipfile
for mid in IDS:
    _, test_norm, test_abn = get_file_lists(mid)
    with zipfile.ZipFile(f'test_{mid}.zip', 'w') as z:
        for f in test_norm[:30]:
            z.write(f, f'{mid}/normal/{os.path.basename(f)}')
        for f in test_abn[:30]:
            z.write(f, f'{mid}/abnormal/{os.path.basename(f)}')
"
```

Transfer and extract on board:

```sh
# On PC
for id in id_00 id_02 id_04 id_06; do
  scp -O -i ~/.ssh/sama7g54 -o IdentitiesOnly=yes test_$id.zip root@192.168.100.46:/data/
done

# On board
mkdir -p /data/mimii_test
for id in id_00 id_02 id_04 id_06; do
  python3 -c "import zipfile; zipfile.ZipFile('/data/test_$id.zip').extractall('/data/mimii_test/')"
done
```

> Use Python's zipfile module, not `unzip`. BusyBox unzip segfaults on large archives.

### Step 23 — Run Inference

```sh
python3 /data/infer_gaussian.py
python3 /data/infer_hitachi.py
```

---

## Enable On-Board Microphones

These steps are optional and only needed if you want to capture live audio from the board's PDMC MEMS microphones (Knowles SPK0641HT4H-1).

### Step 24 — Close Jumper J3

Physically close jumper J3 on the SAMA7G54-EK board to power the microphones. Without this, `arecord` will fail with `BEs uses 0 channels`.

### Step 25 — Apply Device Tree Overlay

Reboot and interrupt U-Boot by pressing ENTER. Then edit the boot command:

```
edit bootcmd
```

Change:

```
fatload mmc 1:1 0x63000000 sama7g5ek.itb; bootm 0x63000000#kernel_dtb
```

To:

```
fatload mmc 1:1 0x63000000 sama7g5ek.itb; bootm 0x63000000#kernel_dtb#pdmc0
```

Save and boot:

```
saveenv
boot
```

### Step 26 — Record Audio

```sh
arecord -D hw:0,0 -f S16_LE -r 16000 -c 4 -d 10 /data/recording.wav
```
## Recording results
After comparing the original audio with the recorded audio using MATLAB these are the expected results forthe example provided:

--- RMS Energy ---
Original:  0.2245
Recorded:  0.2430

--- Dominant Frequency ---
Original:  237.00 Hz
Recorded:  237.00 Hz

--- Spectral Centroid ---
Original:  1520.36 Hz
Recorded:  1420.73 Hz

--- Crest Factor ---
Original:  4.4553
Recorded:  4.1159

--- Spectral MSE (0 = identical) ---
Score: 0.00000065

--- Spectral Correlation (1 = identical) ---
Correlation: 0.7332


---

## Results

### PC

| ID       | Gaussian AUC | G Time (ms) | Hitachi AUC | H Time (ms) |
|----------|-------------|-------------|-------------|-------------|
| id_00    | 0.9348      | ~17         | 0.7712      | ~37         |
| id_02    | 0.9977      | ~17         | 0.9882      | ~43         |
| id_04    | 0.9767      | ~21         | 0.9412      | ~42         |
| id_06    | 0.9999      | ~20         | 0.9947      | ~43         |
| Macro avg| 0.9773      | ~19         | 0.9238      | ~41         |

### Board (SAMA7G54-EK, ARM Cortex-A7 @ 800MHz)

| Model              | Macro AUC | Avg Time (ms/file) |
|--------------------|-----------|--------------------|
| Gaussian           | 0.9472    | ~145               |
| Hitachi TFLite     | 0.9833    | ~1025              |

AUC values closely match PC results. Inference time is 10-15x slower due to the ARM Cortex-A7 vs a modern x86 CPU.

---

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| pyenv not found after install | PATH not updated | Manually add pyenv/bin and pyenv/shims to user PATH |
| venv uses Python 3.12 not 3.8 | pyenv shims not active in venv creation | Use full path: `.pyenv/versions/3.8.10/python.exe -m venv venv` |
| TF import error: Descriptors cannot be created | protobuf version too new | `pip install protobuf==3.20.3` |
| pip fails with SSL error on board | Board clock is wrong (year 2018) | `date -s "2026-03-24 09:00:00"` before any pip command |
| numpy build fails on board | No gcc on Yocto image | Use `--no-deps`; install pre-built wheel from Docker |
| numpy imports but crashes with multiarray error | gnueabihf vs gnueabi mismatch | Run the symlink creation command (Step 19) |
| glibc version mismatch loading numpy .so | Wheel built against glibc 2.34, board has 2.31 | Use `python:3.8-bullseye` Docker image (not slim/bookworm) |
| tflite set_tensor dimension mismatch | Stale inference script on board | Re-transfer infer_hitachi.py; confirm frame shape is (1, 320) |
| arecord: BEs uses 0 channels | J3 not closed or pdmc0 overlay not applied | Close jumper J3 AND apply pdmc0 device tree overlay |
| SCP connection drops mid-transfer | SSH keepalive timeout | Use per-file zip archives transferred individually |
| Root filesystem full | Linux4SAM root partition is only ~500MB | Use /data partition for all files |
| unzip segfaults on board | BusyBox unzip cannot handle large zips | Use Python: `python3 -c "import zipfile; zipfile.ZipFile('x.zip').extractall('dir/')"` |
| SSH still asks for password after key setup | Missing `-o IdentitiesOnly=yes` | Always use: `scp -O -i ~/.ssh/sama7g54 -o IdentitiesOnly=yes` |
| WAV files unreadable by Python wave module | MIMII uses WAVE_FORMAT_EXTENSIBLE (format 65534) | Use the custom struct-based `load_wav()` in the board scripts |

---

## Quick Reference

### Board Access

| | |
|---|---|
| Board IP | 192.168.100.46 |
| SSH | `ssh -i ~/.ssh/sama7g54 -o IdentitiesOnly=yes root@192.168.100.46` |
| Serial | PuTTY — Serial — COMx — 115200 baud |
| Login | root (no password) |

### Key Paths

| Location | Path |
|---|---|
| PC project | `C:\Users\<user>\mimii_project\` |
| PC models | `C:\Users\<user>\mimii_project\models\` |
| Board working dir | `/data/` |
| Board test audio | `/data/mimii_test/` |
| Board inference scripts | `/data/infer_gaussian.py`, `/data/infer_hitachi.py` |

### Key Commands

| Task | Command |
|---|---|
| Fix board clock | `date -s "2026-03-24 09:00:00"` |
| Get board IP | `udhcpc -i eth0 && ip addr show eth0` |
| Check disk space | `df -h` |
| Verify Python packages | `python3 -c "import numpy, tflite_runtime; print('ok')"` |
| Run Gaussian inference | `python3 /data/infer_gaussian.py` |
| Run Hitachi inference | `python3 /data/infer_hitachi.py` |
| Record audio | `arecord -D hw:0,0 -f S16_LE -r 16000 -c 4 -d 10 /data/rec.wav` |

