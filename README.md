# 🧠 Muse 2 Emotion Classification & PAD Decoding (`muse-emotion-classification`)

> **NeuroTech@UIUC — Neuromarketing Project (Fall 2026)**  
> Real-time and offline pipeline converting 4-channel **Muse 2 EEG** streams (`TP9`, `AF7`, `AF8`, `TP10`) into **3D PAD (Pleasure, Arousal, Dominance)** coordinates and discrete emotional state tokens.

---

## 🗺️ Pipeline Architecture

```
 ┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
 │ 1. Raw Muse 2 CSVs     │ ───> │ 2. Signal Conditioning  │ ───> │ 3. Sliding Epoching     │
 │ & Rail Interpolation   │      │ (0.5-45Hz FIR + ASR)    │      │ (2.0s Window, 0.5s Step)│
 └────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                               │
                                                                               ▼
 ┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
 │ 5. Discrete Emotion    │ <─── │ 4. Model A Multi-Output │ <─── │ Slide 6-9 Spectral &    │
 │ Token Classification   │      │ Ridge PAD Regression    │      │ FAA Asymmetry Features  │
 └────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

---

## 📂 Repository Structure

```text
muse emotion classification/
├── muse_eeg.py                 # Live Muse 2 LSL stream recorder (start / stop / plot)
├── run_pipeline.py             # End-to-end pipeline runner across all 5 steps
├── admaster.xlsx               # Ground-truth ad links, PAD ratings, and keywords
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Excludes venv/ and temporary session lockfiles
├── recordings/                 # Raw Muse 2 EEG CSV recordings & waveform plots
└── src/                        # Modular 5-step pipeline for the technical team
    ├── __init__.py             # Global constants (CHANNELS, SFREQ = 256.0 Hz)
    ├── step1_loader.py         # Freshman 1: Prefix matcher & +-1000uV rail dropout interpolation
    ├── step2_clean.py          # Freshman 2: Bandpass (0.5-45Hz), 60Hz notch & EducationalASR
    ├── step3_features.py       # Freshman 3: 2.0s sliding window, Welch PSD, FAA & Arousal ratios
    ├── step4_labels.py         # Freshman 4: admaster.xlsx reader & [-1, 1] PAD normalization
    └── step5_model.py          # Freshman 5: Multi-Output Ridge regressor & discrete emotion tokens
```

---

## 🚀 Quickstart

### 1. Activate Virtual Environment & Install Dependencies
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Emotion Pipeline
```powershell
python run_pipeline.py
```
This script automatically:
1. Matches the `EEGid` entries in `admaster.xlsx` to the CSV recordings in `recordings/`.
2. Repairs $\pm 1000\mu\text{V}$ dry-electrode contact dropouts.
3. Applies zero-phase bandpass ($0.5–45\text{ Hz}$), $60\text{ Hz}$ notch filtering, and **Artifact Subspace Reconstruction (ASR)**.
4. Slices the recordings into **2.0-second epochs** ($75\%$ overlap) and computes **Frontal Alpha Asymmetry (FAA)**, **Left Frontal Theta**, **Arousal Index**, and **Dominance Index**.
5. Trains and cross-validates **Model A** (`Ridge` multi-output regressor) and outputs epoch-by-epoch predictions to `outputs/master_epochs_predictions.csv`.

---

### 3. Record New Live Muse 2 Sessions (BlueMuse LSL)
Ensure **BlueMuse** is streaming on Windows, then run:
```powershell
python muse_eeg.py start
# ... participant views advertisement stimulus ...
python muse_eeg.py stop
```

---

## 🧬 Neuroscience Feature Mapping (From 10/19 Lecture Slides)

| PAD Dimension | Primary Muse 2 Channels | Mathematical Feature | Physiological Meaning |
| :--- | :--- | :--- | :--- |
| **Pleasure (Valence)** | `AF7` (Left), `AF8` (Right) | $\text{FAA} = \ln(\alpha_{\text{AF8}}) - \ln(\alpha_{\text{AF7}})$ and $\ln(\theta_{\text{AF7}})$ | Lower left-forehead Alpha ($\text{FAA} > 0$) + higher left Theta indicates approach motivation & liking. |
| **Arousal (Energy)** | `TP9`, `AF7`, `AF8`, `TP10` | $\ln\left(\frac{\beta + \gamma}{\alpha}\right)$ | Alpha suppression coupled with high-frequency Beta ($13–30\text{ Hz}$) & Gamma ($>30\text{ Hz}$) spikes. |
| **Dominance (Control)** | Prefrontal vs. Temporal | $\ln\left(\frac{\beta_{\text{AF7,AF8}}}{\alpha_{\text{TP9,TP10}}}\right)$ | Distributed prefrontal executive control relative to temporoparietal gating. |

---

## 👥 Team Workflow & Git Hygiene

Each member works on their assigned module in `src/` on a feature branch:
```bash
git checkout -b feat/step1-loader
# edit src/step1_loader.py and test with `python run_pipeline.py`
git add src/step1_loader.py
git commit -m "Implement rail dropout interpolation in step1_loader"
git push origin feat/step1-loader
```
