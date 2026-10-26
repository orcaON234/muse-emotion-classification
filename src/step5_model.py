"""Step 5: Model A Multi-Output PAD Regressor & Discrete Emotion Tokenizer.

Owner: Freshman 5
Responsibilities:
  1. Train a regularized Multi-Output Ridge regression pipeline mapping the
     extracted spectral/FAA features -> [P_norm, A_norm, D_norm].
  2. Apply the DeWave discrete tokenization concept: map continuous PAD coordinates
     to discrete emotion tokens (e.g. 'Excited / Liking', 'Calm / Engaged',
     'Frustrated / Combative', 'Avoidant / Disliking').
  3. Evaluate with cross-validation and export epoch-level predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
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

TARGET_COLS = ["P_norm", "A_norm", "D_norm"]

# 3D PAD Emotion Prototypes (Grounded in Slide 5 & Slide 9)
EMOTION_PROTOTYPES = {
    "Excited / High Liking": np.array([0.70, 0.65, 0.50]),
    "Calm / Pleasant": np.array([0.55, -0.40, 0.35]),
    "Focused / Interested": np.array([0.45, 0.40, 0.30]),
    "Angry / Frustrated (High Dom)": np.array([-0.60, 0.70, 0.60]),
    "Anxious / Overwhelmed (Low Dom)": np.array([-0.60, 0.70, -0.60]),
    "Disliking / Avoidant": np.array([-0.40, 0.40, -0.35]),
    "Bored / Disengaged": np.array([-0.20, -0.60, -0.30]),
}


def pad_to_emotion_token(pad_vec: np.ndarray) -> str:
    """Assign a 3D PAD coordinate [P, A, D] to the nearest discrete emotion token."""
    best_label = "Neutral"
    best_dist = float("inf")
    for label, proto in EMOTION_PROTOTYPES.items():
        dist = float(np.linalg.norm(pad_vec - proto))
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def train_and_evaluate_model_a(df_master: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Fit Multi-Output Ridge model and generate discrete emotion predictions."""
    X = df_master[FEATURE_COLS].to_numpy(dtype=float)
    Y = df_master[TARGET_COLS].to_numpy(dtype=float)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=5.0, random_state=42)),
        ]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    Y_pred = np.clip(cross_val_predict(model, X, Y, cv=kf), -1.0, 1.0)

    # Fit on full dataset for deployment / feature weight inspection
    model.fit(X, Y)

    df_results = df_master.copy()
    df_results["Pred_P"] = np.round(Y_pred[:, 0], 3)
    df_results["Pred_A"] = np.round(Y_pred[:, 1], 3)
    df_results["Pred_D"] = np.round(Y_pred[:, 2], 3)
    df_results["Predicted_Emotion_Token"] = [
        pad_to_emotion_token(row) for row in Y_pred
    ]

    metrics = {
        "MAE_Pleasure": round(float(mean_absolute_error(Y[:, 0], Y_pred[:, 0])), 4),
        "MAE_Arousal": round(float(mean_absolute_error(Y[:, 1], Y_pred[:, 1])), 4),
        "MAE_Dominance": round(float(mean_absolute_error(Y[:, 2], Y_pred[:, 2])), 4),
        "R2_Overall": round(float(r2_score(Y, Y_pred)), 4),
    }
    return model, df_results, metrics
