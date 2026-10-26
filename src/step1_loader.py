"""Step 1: Recording Loader, Prefix Matcher & Rail-Dropout Interpolator.

Owner: Freshman 1
Responsibilities:
  1. Match `EEGid` prefixes from `admaster.xlsx` (e.g. `muse_20261019_165020.csv`)
     to the actual timestamped CSV files inside `recordings/`.
  2. Load the 4 active Muse EEG channels (`TP9`, `AF7`, `AF8`, `TP10`).
  3. Detect dry-electrode rail saturation (|voltage| >= 950 uV) and repair
     dropouts via linear interpolation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from . import CHANNELS, SFREQ

BASE_DIR = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = BASE_DIR / "recordings"


def find_recording_csv(eeg_id_prefix: str, recordings_dir: Path = RECORDINGS_DIR) -> Path:
    """Find the recording CSV matching an `EEGid` prefix from admaster.xlsx."""
    stem = Path(eeg_id_prefix).stem
    matches = sorted(recordings_dir.glob(f"{stem}*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No recording CSV found matching prefix '{stem}' in {recordings_dir}"
        )
    return matches[0]


def load_and_clean_rails(
    csv_path: Path,
    rail_threshold_uv: float = 950.0,
) -> tuple[pd.DataFrame, dict]:
    """Load a Muse CSV recording and interpolate rail-saturated samples.

    Returns:
        df_clean: DataFrame with columns ['time_s', 'TP9', 'AF7', 'AF8', 'TP10']
        stats: Dict summarizing sample count, duration, and % rail dropouts fixed
    """
    df = pd.read_csv(csv_path)
    keep_cols = ["time_s"] + [c for c in CHANNELS if c in df.columns]
    df_clean = df[keep_cols].copy()

    rail_counts = {}
    for ch in CHANNELS:
        mask = df_clean[ch].abs() >= rail_threshold_uv
        rail_counts[ch] = int(mask.sum())
        df_clean.loc[mask, ch] = np.nan
        df_clean[ch] = (
            df_clean[ch]
            .interpolate(method="linear", limit_direction="both")
            .bfill()
            .ffill()
            .fillna(0.0)
        )

    total_samples = len(df_clean)
    stats = {
        "file": csv_path.name,
        "samples": total_samples,
        "duration_s": round(total_samples / SFREQ, 2),
        "rail_dropouts_per_ch": rail_counts,
        "rail_pct": round(
            100.0 * sum(rail_counts.values()) / max(1, total_samples * len(CHANNELS)),
            2,
        ),
    }
    return df_clean, stats


if __name__ == "__main__":
    sample_csv = find_recording_csv("muse_20261019_165020.csv")
    df_out, info = load_and_clean_rails(sample_csv)
    print(f"[Step 1] Loaded {info['file']} ({info['duration_s']}s, {info['samples']} samples)")
    print(f"         Interpolated {info['rail_pct']}% rail-saturated samples: {info['rail_dropouts_per_ch']}")
