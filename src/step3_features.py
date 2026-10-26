"""Step 3: 10s Baseline Subtraction, Sliding-Window Epoching & 13 Spectral Features.

Owner: Member 3
Responsibilities:
  1. Separate the first `baseline_sec` (default 10.0s = 2560 samples) of each
     recording as the Pre-Stimulus Resting Baseline (Slide 12 of 10/19 meeting).
  2. Compute the participant's resting baseline for all 13 features across the
     first 10 seconds (`t = 0s .. 10s`).
  3. Slice the active advertisement viewing period (`t >= 10s`) into overlapping
     2.0-second epochs (`step_sec = 0.5s`) and subtract the 10s resting baseline
     from every epoch so all 13 features measure the *stimulus-evoked change*
     (`Delta_Feature = Stimulus_Epoch_Feature - Baseline_10s_Feature`).
"""

import numpy as np
import pandas as pd
import scipy.signal

from . import CHANNELS, SFREQ

FEATURE_KEYS = [
    "FAA",
    "Left_Frontal_Theta",
    "Arousal_Index",
    "Dominance_Index",
    "Delta_Mean",
    "Theta_Mean",
    "Alpha_Mean",
    "Beta_Mean",
    "Gamma_Mean",
    "Alpha_AF7",
    "Alpha_AF8",
    "Beta_AF7",
    "Beta_AF8",
]


def extract_epoch_features(epoch_data: np.ndarray, sfreq: float = SFREQ) -> dict:
    """Compute Slide 6-9 spectral & PAD features for a single segment (shape: 4 x N).

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
    baseline_sec: float = 10.0,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    subtract_baseline: bool = True,
    sfreq: float = SFREQ,
) -> pd.DataFrame:
    """Use the first `baseline_sec` (10s) as resting baseline, then slice `t >= baseline_sec`."""
    data = df_clean[CHANNELS].to_numpy(dtype=float).T  # (4, n_samples)
    n_samples = data.shape[1]
    base_samples = int(baseline_sec * sfreq)
    win_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    if n_samples <= base_samples + win_samples:
        raise ValueError(
            f"Recording {eeg_id} ({n_samples/sfreq:.1f}s) is shorter than "
            f"baseline_sec ({baseline_sec}s) + window_sec ({window_sec}s)."
        )

    # 1. Compute pre-stimulus resting baseline features on [0 .. baseline_sec]
    baseline_matrix = data[:, :base_samples]
    baseline_feats = extract_epoch_features(baseline_matrix, sfreq=sfreq)

    # 2. Slice active stimulus period starting at `baseline_sec` (t >= 10.0s)
    rows = []
    epoch_idx = 0
    for start in range(base_samples, n_samples - win_samples + 1, step_samples):
        stop = start + win_samples
        epoch_matrix = data[:, start:stop]
        raw_feats = extract_epoch_features(epoch_matrix, sfreq=sfreq)

        if subtract_baseline:
            # Baseline-corrected change in log-power / asymmetry relative to first 10s
            feats = {k: raw_feats[k] - baseline_feats[k] for k in FEATURE_KEYS}
        else:
            feats = dict(raw_feats)

        feats["EEGid"] = eeg_id
        feats["Epoch_ID"] = epoch_idx
        feats["Start_Time_s"] = round(start / sfreq, 2)
        feats["Stimulus_Time_s"] = round((start - base_samples) / sfreq, 2)
        rows.append(feats)
        epoch_idx += 1

    return pd.DataFrame(rows)
