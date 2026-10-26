"""Step 5: Emotion Valence Classifier (Onboarding Notebook ML Approach).

Owner: Member 5
Responsibilities:
  1. Train the exact Scikit-Learn Pipeline from Module 6 of
     `Neuromarketing_Technical_Onboarding_Fall2026.ipynb`:
     `StandardScaler` + `LogisticRegression` (or `SGDClassifier`, `SVC`, `RandomForestClassifier`).
  2. Evaluate using 5-Fold `StratifiedKFold` cross-validation (`classification_report`,
     `confusion_matrix`, and `roc_auc_score`).
  3. Extract learned feature weights (`coef_`) to verify neuroscience validity
     (e.g. positive weight on `FAA` and `Left_Frontal_Theta`).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def build_classifier_pipeline(
    model_type: str = "logreg",
    C: float = 1.0,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
) -> Pipeline:
    """Create the Scikit-Learn Pipeline matching the Onboarding Notebook.

    Hyperparameters to test:
        model_type: 'logreg' (Logistic Regression), 'sgd' (Gradient Descent with learning_rate),
                    'svm' (Support Vector Classifier), or 'rf' (Random Forest)
        C: Inverse regularization strength (smaller C = stronger penalty against overfitting)
        learning_rate: Step size `eta0` used when `model_type='sgd'`
        max_iter: Maximum training iterations (epochs over the training set)
    """
    if model_type == "logreg":
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    elif model_type == "sgd":
        clf = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=learning_rate,
            alpha=1.0 / max(C * 100.0, 1e-6),
            max_iter=max_iter,
            random_state=42,
        )
    elif model_type == "svm":
        clf = SVC(kernel="rbf", C=C, probability=True, random_state=42)
    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


def train_and_evaluate_model_a(
    df_master: pd.DataFrame,
    model_type: str = "logreg",
    C: float = 1.0,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    n_splits: int = 5,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Train and cross-validate the Emotion Valence classifier (Module 6 approach)."""
    X = df_master[FEATURE_COLS].to_numpy(dtype=float)
    y = df_master["Emotion_Valence"].to_numpy(dtype=int)

    pipeline = build_classifier_pipeline(
        model_type=model_type,
        C=C,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )

    # 5-Fold Stratified Cross-Validation (exact match to Onboarding Notebook Cell 16)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = float(accuracy_score(y, y_pred))
    auc_score = float(roc_auc_score(y, y_prob))
    cm = confusion_matrix(y, y_pred)
    report_str = classification_report(
        y,
        y_pred,
        target_names=["Negative/Withdrawal (0)", "Positive/Approach (1)"],
    )

    # Fit on full dataset to inspect learned feature weights (Cell 18 of Onboarding Notebook)
    pipeline.fit(X, y)
    clf_step = pipeline.named_steps["classifier"]
    if hasattr(clf_step, "coef_"):
        weights = clf_step.coef_[0]
    elif hasattr(clf_step, "feature_importances_"):
        weights = clf_step.feature_importances_
    else:
        weights = np.zeros(len(FEATURE_COLS))

    df_weights = pd.DataFrame(
        {"Feature": FEATURE_COLS, "Weight": np.round(weights, 4)}
    ).sort_values("Weight", ascending=False)

    df_results = df_master.copy()
    df_results["Predicted_Valence"] = y_pred
    df_results["Approach_Probability"] = np.round(y_prob, 4)
    df_results["Predicted_Label"] = np.where(
        y_pred == 1, "Positive/Approach", "Negative/Withdrawal"
    )

    metrics = {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc_score, 4),
        "confusion_matrix": cm,
        "classification_report": report_str,
        "feature_weights": df_weights,
    }
    return pipeline, df_results, metrics
