"""End-to-End Muse 2 Emotion Classification Pipeline Runner.

Runs all 5 freshman modules sequentially on the recordings listed in `admaster.xlsx`:
  Step 1: Match CSV filenames & interpolate +-1000 uV dry-electrode rail dropouts
  Step 2: Apply 0.5-45 Hz bandpass, 60 Hz notch, and ASR burst subspace cleaning
  Step 3: Slice into 2.0s epochs (0.5s step) & extract Slide 6-9 PAD features
  Step 4: Load & normalize `admaster.xlsx` PAD labels
  Step 5: Train Model A (Multi-Output Ridge) & map epochs to discrete emotion tokens

Usage:
    python run_pipeline.py
"""

from pathlib import Path
import pandas as pd

from src.step1_loader import find_recording_csv, load_and_clean_rails
from src.step2_clean import clean_eeg_dataframe
from src.step3_features import slice_and_extract_recording
from src.step4_labels import load_admaster_labels
from src.step5_model import train_and_evaluate_model_a

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("=" * 72)
    print("MUSE 2 EMOTION CLASSIFICATION PIPELINE (MODEL A BASELINE)")
    print("=" * 72)

    # Step 4: Load admaster.xlsx labels
    df_labels = load_admaster_labels(BASE_DIR / "admaster.xlsx")
    print(f"\n[1/5] Loaded {len(df_labels)} labeled ads from admaster.xlsx:")
    for _, row in df_labels.iterrows():
        print(
            f"  - {row['EEGid']} | P={row['P_norm']:+.2f}, A={row['A_norm']:+.2f}, "
            f"D={row['D_norm']:+.2f} | Ad: {row['adlink']}"
        )

    # Steps 1, 2, 3: Process each recording into epoch features
    all_epochs = []
    print("\n[2/5] Cleaning recordings (Rail interpolation + 0.5-45Hz FIR + ASR)...")
    for _, row in df_labels.iterrows():
        eeg_id = str(row["EEGid"]).strip()
        csv_path = find_recording_csv(eeg_id)
        df_railed, stats = load_and_clean_rails(csv_path)
        df_clean = clean_eeg_dataframe(df_railed, asr_cutoff=4.5)
        df_feats = slice_and_extract_recording(
            df_clean, eeg_id=eeg_id, window_sec=2.0, step_sec=0.5
        )
        print(
            f"  - {stats['file']} ({stats['duration_s']}s) -> "
            f"Repaired {stats['rail_pct']}% rail dropouts -> "
            f"Extracted {len(df_feats)} epochs"
        )
        all_epochs.append(df_feats)

    df_features = pd.concat(all_epochs, ignore_index=True)

    # Merge features with admaster PAD labels
    df_master = df_features.merge(
        df_labels[["EEGid", "adlink", "P_norm", "A_norm", "D_norm", "Keywords"]],
        on="EEGid",
        how="left",
    )
    print(f"\n[3/5] Built Master Dataset: {len(df_master)} total 2-second epochs.")

    # Step 5: Train & evaluate Model A
    print("\n[4/5] Training & Cross-Validating Model A (Multi-Output Ridge)...")
    _, df_results, metrics = train_and_evaluate_model_a(df_master)
    print(f"  - Pleasure MAE : {metrics['MAE_Pleasure']:.4f}")
    print(f"  - Arousal MAE  : {metrics['MAE_Arousal']:.4f}")
    print(f"  - Dominance MAE: {metrics['MAE_Dominance']:.4f}")
    print(f"  - Overall R^2  : {metrics['R2_Overall']:.4f}")

    # Summarize Discrete Emotion Token Predictions per Ad
    print("\n[5/5] Dominant Predicted Emotion Tokens per Recording:")
    for eeg_id, group in df_results.groupby("EEGid"):
        top_token = group["Predicted_Emotion_Token"].value_counts().idxmax()
        mean_p = group["Pred_P"].mean()
        mean_a = group["Pred_A"].mean()
        mean_d = group["Pred_D"].mean()
        print(
            f"  - {eeg_id}: [{mean_p:+.2f}, {mean_a:+.2f}, {mean_d:+.2f}] -> {top_token}"
        )

    out_csv = OUTPUT_DIR / "master_epochs_predictions.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved full epoch predictions to: {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
