"""Step 3: Sliding-Window Epoching & Slide 6-9 Spectral Feature Extraction.

Owner: Member 3
Responsibilities:
  1. Slice continuous cleaned Muse recordings into 2.0-second epochs (512 samples)
     with a 0.5-second step (75% overlap).
  2. Extract Welch Power Spectral Density (PSD) across the 5 bands from Slide 6:
     - Delta (0.5 - 4 Hz): Reward processing & baseline
     - Theta (4 - 8 Hz): Memory encoding, workload, & left-frontal preference
     - Alpha (8 - 12 Hz): Cortical idling & relaxation
     - Beta  (13 - 30 Hz): Active analytical focus & alertness
     - Gamma (30 - 45 Hz): Sensory binding & peak emotional intensity
  3. Compute the PAD indices from Slides 7-9:
     - Frontal Alpha Asymmetry (FAA = ln(Alpha_AF8) - ln(Alpha_AF7)) -> Pleasure
     - Left Frontal Theta (ln(Theta_AF7)) -> Pleasant engagement
     - Arousal Index ((Beta + Gamma) / Alpha) -> Arousal
     - Dominance Index (Frontal-Temporoparietal Beta/Alpha balance) -> Dominance
"""

import numpy as np
import pandas as pd
import scipy.signal

from . import CHANNELS, SFREQ


def extract_epoch_features(epoch_data: np.ndarray, sfreq: float = SFREQ) -> dict:
    """Compute Slide 6-9 spectral & PAD features for a single epoch (shape: 4 x N).

    Channel indices: 0=TP9, 1=AF7, 2=AF8, 3=TP10
    """
    nperseg = min(int(sfreq), epoch_data.shape[1])
    freqs, psd = scipy.signal.welch(epoch_data, fs=sfreq, nperseg=nperseg, axis=1)

    def band_power(fmin: float, fmax: float) -> np.ndarray:
        mask = (freqs >= fmin) & (freqs < fmax)
        return np.mean(psd[:, mask], axis=1) + 1e-8

    delta = band_power(0.5, 4.0)
    theta = band_power(4.0, 8.0)
    alpha = band_power(8.0, 12.0)
    beta = band_power(13.0, 30.0)
    gamma = band_power(30.0, 45.0)

    # Slide 7: Pleasure -> Frontal Alpha Asymmetry (AF8 vs AF7) & Left Frontal Theta (AF7)
    faa = float(np.log(alpha[2]) - np.log(alpha[1]))
    left_frontal_theta = float(np.log(theta[1]))

    # Slide 8: Arousal -> Fast waves (Beta + Gamma) relative to slow Alpha
    arousal_index = float(np.log((np.mean(beta) + np.mean(gamma)) / np.mean(alpha)))

    # Slide 9: Dominance -> Prefrontal (AF7/AF8) Beta control vs Temporoparietal (TP9/TP10) Alpha
    frontal_beta = 0.5 * (beta[1] + beta[2])
    temporal_alpha = 0.5 * (alpha[0] + alpha[3])
    dominance_index = float(np.log(frontal_beta / temporal_alpha))

    return {
        "FAA": faa,
        "Left_Frontal_Theta": left_frontal_theta,
        "Arousal_Index": arousal_index,
        "Dominance_Index": dominance_index,
        "Delta_Mean": float(np.log(np.mean(delta))),
        "Theta_Mean": float(np.log(np.mean(theta))),
        "Alpha_Mean": float(np.log(np.mean(alpha))),
        "Beta_Mean": float(np.log(np.mean(beta))),
        "Gamma_Mean": float(np.log(np.mean(gamma))),
        "Alpha_AF7": float(np.log(alpha[1])),
        "Alpha_AF8": float(np.log(alpha[2])),
        "Beta_AF7": float(np.log(beta[1])),
        "Beta_AF8": float(np.log(beta[2])),
    }


def slice_and_extract_recording(
    df_clean: pd.DataFrame,
    eeg_id: str,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    sfreq: float = SFREQ,
) -> pd.DataFrame:
    """Slice a cleaned recording into overlapping epochs and return feature table."""
    data = df_clean[CHANNELS].to_numpy(dtype=float).T  # (4, n_samples)
    n_samples = data.shape[1]
    win_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    rows = []
    epoch_idx = 0
    for start in range(0, n_samples - win_samples + 1, step_samples):
        stop = start + win_samples
        epoch_matrix = data[:, start:stop]
        feats = extract_epoch_features(epoch_matrix, sfreq=sfreq)
        feats["EEGid"] = eeg_id
        feats["Epoch_ID"] = epoch_idx
        feats["Start_Time_s"] = round(start / sfreq, 2)
        rows.append(feats)
        epoch_idx += 1

    return pd.DataFrame(rows)
