"""End-to-End Muse 2 Emotion Classification Pipeline Runner.

Runs all 5 team modules sequentially on the recordings listed in `admaster.xlsx`,
using the exact Machine Learning approach from `Neuromarketing_Technical_Onboarding_Fall2026.ipynb`
(Module 6: StandardScaler + LogisticRegression with 5-Fold Stratified Cross-Validation):
  Step 1 (Member 1): Match CSV filenames & interpolate +-1000 uV rail dropouts
  Step 2 (Member 2): Apply 0.5-45 Hz bandpass, 60 Hz notch, and EducationalASR cleaning
  Step 3 (Member 3): Slice into epochs & extract the 13 spectral/FAA features
  Step 4 (Member 4): Load `admaster.xlsx` labels & create binary `Emotion_Valence` target
  Step 5 (Member 5): Train & evaluate LogisticRegression / SGD / SVM / RF classifier

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

# =============================================================================
# TUNABLE HYPERPARAMETERS (Change these to test how they affect accuracy/AUC!)
# =============================================================================
HYPERPARAMS = {
    "rail_threshold_uv": 950.0,  # Step 1: Cutoff (uV) for dry-sensor rail dropout
    "asr_cutoff": 4.5,           # Step 2: ASR burst threshold k (try 2.0, 4.5, 10.0)
    "window_sec": 2.0,           # Step 3: Epoch window size T in seconds (try 1.0, 2.0, 4.0)
    "step_sec": 0.5,             # Step 3: Sliding window step size in seconds (try 0.25, 0.5, 1.0)
    "model_type": "logreg",      # Step 5: 'logreg', 'sgd', 'svm', or 'rf'
    "C": 1.0,                    # Step 5: Regularization C (try 0.1, 1.0, 10.0)
    "learning_rate": 0.01,       # Step 5: Learning rate eta0 (used when model_type='sgd')
    "max_iter": 1000,            # Step 5: Max training iterations / passes
    "n_splits": 5,               # Step 5: Number of Stratified K-Fold CV splits
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("=" * 72)
    print("MUSE 2 EMOTION CLASSIFICATION PIPELINE (ONBOARDING ML APPROACH)")
    print("=" * 72)
    print("Active Hyperparameters:", HYPERPARAMS)

    # Step 4: Load admaster.xlsx labels
    df_labels = load_admaster_labels(BASE_DIR / "admaster.xlsx")
    print(f"\n[1/5] Loaded {len(df_labels)} labeled ads from admaster.xlsx:")
    for _, row in df_labels.iterrows():
        valence_str = "Positive/Approach (1)" if row["Emotion_Valence"] == 1 else "Negative/Withdrawal (0)"
        print(
            f"  - {row['EEGid']} | Pleasure={row['Pleasure']:+.1f} -> {valence_str}"
        )

    # Steps 1, 2, 3: Process each recording into epoch features
    all_epochs = []
    print(
        f"\n[2/5] Cleaning & slicing recordings "
        f"(asr_cutoff={HYPERPARAMS['asr_cutoff']}, window_sec={HYPERPARAMS['window_sec']}s, step_sec={HYPERPARAMS['step_sec']}s)..."
    )
    for _, row in df_labels.iterrows():
        eeg_id = str(row["EEGid"]).strip()
        csv_path = find_recording_csv(eeg_id)
        df_railed, stats = load_and_clean_rails(
            csv_path, rail_threshold_uv=HYPERPARAMS["rail_threshold_uv"]
        )
        df_clean = clean_eeg_dataframe(df_railed, asr_cutoff=HYPERPARAMS["asr_cutoff"])
        df_feats = slice_and_extract_recording(
            df_clean,
            eeg_id=eeg_id,
            window_sec=HYPERPARAMS["window_sec"],
            step_sec=HYPERPARAMS["step_sec"],
        )
        print(
            f"  - {stats['file']} ({stats['duration_s']}s) -> "
            f"Repaired {stats['rail_pct']}% rail dropouts -> "
            f"Extracted {len(df_feats)} epochs"
        )
        all_epochs.append(df_feats)

    df_features = pd.concat(all_epochs, ignore_index=True)

    # Merge features with admaster labels
    df_master = df_features.merge(
        df_labels[
            ["EEGid", "adlink", "Pleasure", "P_norm", "A_norm", "D_norm", "Emotion_Valence", "Keywords"]
        ],
        on="EEGid",
        how="left",
    )
    print(f"\n[3/5] Built Master Dataset: {len(df_master)} total epochs x 13 features.")

    # Step 5: Train & evaluate Classifier (Onboarding Notebook Module 6)
    print(
        f"\n[4/5] Training & Evaluating Classifier "
        f"(model={HYPERPARAMS['model_type']}, C={HYPERPARAMS['C']}, lr={HYPERPARAMS['learning_rate']})..."
    )
    _, df_results, metrics = train_and_evaluate_model_a(
        df_master,
        model_type=HYPERPARAMS["model_type"],
        C=HYPERPARAMS["C"],
        learning_rate=HYPERPARAMS["learning_rate"],
        max_iter=HYPERPARAMS["max_iter"],
        n_splits=HYPERPARAMS["n_splits"],
    )
    print(metrics["classification_report"])
    print(f"Overall 5-Fold CV Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"Overall ROC-AUC Score      : {metrics['roc_auc']:.4f}")

    print("\n[5/5] Top Learned Feature Weights (Positive = Predicts Positive/Approach):")
    for _, w_row in metrics["feature_weights"].head(5).iterrows():
        print(f"  {w_row['Feature']:<20}: {w_row['Weight']:+.4f}")

    out_csv = OUTPUT_DIR / "master_epochs_predictions.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved full epoch predictions to: {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
