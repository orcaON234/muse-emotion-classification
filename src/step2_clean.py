"""Step 2: Zero-Phase Bandpass/Notch Filtering & ASR Subspace Cleaning.

Owner: Freshman 2
Responsibilities:
  1. Apply 0.5 - 45.0 Hz zero-phase Butterworth/FIR bandpass filter to remove
     DC drift (<0.5 Hz) and high-frequency muscle tension (>45 Hz).
  2. Apply 60.0 Hz notch filter to eliminate powerline interference.
  3. Run Artifact Subspace Reconstruction (EducationalASR) to suppress transient
     jaw clenches (TP9/TP10) and forehead blinks (AF7/AF8) without dropping trials.
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
    """Artifact Subspace Reconstruction (ASR) for 4-channel Muse EEG."""

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

    def fit(self, data: np.ndarray):
        """Calibrate baseline covariance on lower-variance reference windows."""
        # Select windows in the lower 50th percentile of RMS amplitude as clean reference
        n_ch, n_samples = data.shape
        win_size = min(256, n_samples)
        rms_vals = []
        slices = []
        for start in range(0, n_samples - win_size + 1, win_size):
            seg = data[:, start : start + win_size]
            rms_vals.append(np.sqrt(np.mean(seg**2)))
            slices.append(seg)

        if slices:
            median_rms = np.percentile(rms_vals, 60)
            clean_segs = [s for s, r in zip(slices, rms_vals) if r <= median_rms]
            ref_data = np.concatenate(clean_segs, axis=1) if clean_segs else data
        else:
            ref_data = data

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

        # Fill tail samples if any
        rem = weights == 0
        output[:, rem] = data[:, rem]
        weights[rem] = 1.0
        return output / weights[None, :]


def clean_eeg_dataframe(
    df_raw: pd.DataFrame,
    asr_cutoff: float = 4.5,
) -> pd.DataFrame:
    """Run full filtering + ASR cleaning on a loaded Muse DataFrame."""
    raw_matrix = df_raw[CHANNELS].to_numpy(dtype=float).T  # (4, n_samples)
    filtered = bandpass_and_notch_filter(raw_matrix, sfreq=SFREQ)

    asr = EducationalASR(cutoff=asr_cutoff)
    asr.fit(filtered)
    cleaned_matrix = asr.transform(filtered, sfreq=SFREQ)

    df_clean = df_raw.copy()
    for idx, ch in enumerate(CHANNELS):
        df_clean[ch] = cleaned_matrix[idx, :]
    return df_clean
