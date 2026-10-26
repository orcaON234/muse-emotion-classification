"""Step 2: Zero-Phase Bandpass/Notch Filtering & ASR Subspace Cleaning.

Owner: Member 2
Responsibilities:
  1. Apply 0.5 - 45.0 Hz zero-phase Butterworth/FIR bandpass filter to remove
     DC drift (<0.5 Hz) and high-frequency muscle tension (>45 Hz).
  2. Apply 60.0 Hz notch filter to eliminate powerline interference.
  3. Calibrate Artifact Subspace Reconstruction (EducationalASR) specifically on
     the first `baseline_sec` (default 10.0s) of pre-stimulus resting EEG, then
     clean the entire recording without dropping trials.
"""

import numpy as np
import pandas as pd
import scipy.signal

from . import CHANNELS, SFREQ


def bandpass_and_notch_filter(
    data: np.ndarray,
    sfreq: float = SFREQ,
    low_hz: float = 0.5,
    high_hz: float = 45.0,
    notch_hz: float = 60.0,
) -> np.ndarray:
    """Apply zero-phase bandpass (0.5-45 Hz) and 60 Hz notch filter.

    Args:
        data: Array of shape (n_channels, n_samples)
    Returns:
        filtered: Array of shape (n_channels, n_samples)
    """
    sos_bp = scipy.signal.butter(
        4, [low_hz, high_hz], btype="bandpass", fs=sfreq, output="sos"
    )
    filtered = scipy.signal.sosfiltfilt(sos_bp, data, axis=1)

    b_notch, a_notch = scipy.signal.iirnotch(w0=notch_hz, Q=30.0, fs=sfreq)
    filtered = scipy.signal.filtfilt(b_notch, a_notch, filtered, axis=1)
    return filtered


class EducationalASR:
    """Artifact Subspace Reconstruction (ASR) calibrated on pre-stimulus baseline."""

    def __init__(
        self,
        cutoff: float = 4.5,
        win_len_sec: float = 0.5,
        step_len_sec: float = 0.25,
    ):
        self.cutoff = cutoff
        self.win_len_sec = win_len_sec
        self.step_len_sec = step_len_sec
        self.calib_eigvals = None
        self.calib_eigvecs = None

    def fit(self, baseline_data: np.ndarray):
        """Calibrate baseline covariance on the pre-stimulus resting segment."""
        n_ch, n_samples = baseline_data.shape
        win_size = min(128, n_samples)
        rms_vals = []
        slices = []
        for start in range(0, n_samples - win_size + 1, win_size):
            seg = baseline_data[:, start : start + win_size]
            rms_vals.append(np.sqrt(np.mean(seg**2)))
            slices.append(seg)

        if slices:
            cutoff_rms = np.percentile(rms_vals, 70)
            clean_segs = [s for s, r in zip(slices, rms_vals) if r <= cutoff_rms]
            ref_data = np.concatenate(clean_segs, axis=1) if clean_segs else baseline_data
        else:
            ref_data = baseline_data

        cov = np.cov(ref_data)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        self.calib_eigvals = np.maximum(eigvals[idx], 1e-8)
        self.calib_eigvecs = eigvecs[:, idx]
        return self

    def transform(self, data: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
        """Reconstruct transient high-amplitude bursts in sliding windows."""
        n_ch, n_times = data.shape
        win_samples = int(self.win_len_sec * sfreq)
        step_samples = int(self.step_len_sec * sfreq)

        output = np.zeros_like(data)
        weights = np.zeros(n_times)
        hann = np.hanning(win_samples)

        M_calib = self.calib_eigvecs @ np.diag(1.0 / np.sqrt(self.calib_eigvals))

        for start in range(0, n_times - win_samples + 1, step_samples):
            stop = start + win_samples
            win = data[:, start:stop]

            proj = M_calib.T @ win
            comp_std = np.std(proj, axis=1)
            clean_mask = comp_std <= self.cutoff

            if np.all(clean_mask):
                recon_win = win
            elif not np.any(clean_mask):
                recon_win = win * 0.15
            else:
                V_clean = self.calib_eigvecs[:, clean_mask]
                recon_win = V_clean @ (V_clean.T @ win)

            output[:, start:stop] += recon_win * hann[None, :]
            weights[start:stop] += hann

        rem = weights == 0
        output[:, rem] = data[:, rem]
        weights[rem] = 1.0
        return output / weights[None, :]


def clean_eeg_dataframe(
    df_raw: pd.DataFrame,
    asr_cutoff: float = 4.5,
    baseline_sec: float = 10.0,
) -> pd.DataFrame:
    """Run filtering + ASR (calibrated on the first `baseline_sec` seconds)."""
    raw_matrix = df_raw[CHANNELS].to_numpy(dtype=float).T  # (4, n_samples)
    filtered = bandpass_and_notch_filter(raw_matrix, sfreq=SFREQ)

    baseline_samples = min(int(baseline_sec * SFREQ), filtered.shape[1])
    baseline_segment = filtered[:, :baseline_samples]

    asr = EducationalASR(cutoff=asr_cutoff)
    asr.fit(baseline_segment)
    cleaned_matrix = asr.transform(filtered, sfreq=SFREQ)

    df_clean = df_raw.copy()
    for idx, ch in enumerate(CHANNELS):
        df_clean[ch] = cleaned_matrix[idx, :]
    return df_clean
