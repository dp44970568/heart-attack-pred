"""
Model Training, Cross-Validation, and Evaluation Pipeline for Heart Attack Prediction.
Trains both Full Hospital Clinical and Routine Personal Screening Models.
Evaluates Random Forest, XGBoost, HistGradientBoosting, Logistic Regression, and Ensemble.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    brier_score_loss
)
from xgboost import XGBClassifier

from src.config import (
    CLINICAL_FEATURES,
    ROUTINE_FEATURES,
    MODELS_DIR,
    FEATURE_LABELS
)
from src.data_loader import (
    load_clinical_training_data,
    load_routine_training_data
)


def get_model_candidates(features_list: List[str]) -> Dict[str, Any]:
    """Define candidate model pipelines."""
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced"
            ))
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                eval_metric="logloss"
            ))
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=5,
                random_state=42
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.8,
                max_iter=500,
                random_state=42,
                class_weight="balanced"
            ))
        ]),
        "Soft Voting Ensemble": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=2, random_state=42, class_weight="balanced")),
                    ("xgb", XGBClassifier(n_estimators=90, max_depth=3, learning_rate=0.05, subsample=0.85, random_state=42, eval_metric="logloss")),
                    ("lr", Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=0.8, max_iter=500, random_state=42, class_weight="balanced"))]))
                ],
                voting="soft",
                weights=[2, 2, 1]
            ))
        ])
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series, model_name_prefix: str) -> Dict[str, Any]:
    """
    Run 5-Fold Stratified Cross Validation across all model candidates.
    Compute comprehensive clinical performance metrics.
    """
    candidates = get_model_candidates(list(X.columns))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print(f"\n--- Cross-Validation Benchmark: {model_name_prefix} ---")
    print(f"{'Model':<24} | {'Accuracy':<8} | {'ROC-AUC':<8} | {'Recall':<8} | {'Precision':<9} | {'F1':<8} | {'Specificity':<11}")
    print("-" * 92)

    for name, pipeline in candidates.items():
        # Get out-of-fold probability predictions
        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = float(accuracy_score(y, y_pred))
        auc = float(roc_auc_score(y, y_prob))
        rec = float(recall_score(y, y_pred)) # Sensitivity
        prec = float(precision_score(y, y_pred))
        f1 = float(f1_score(y, y_pred))
        brier = float(brier_score_loss(y, y_prob))
        
        # Specificity
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        print(f"{name:<24} | {acc:.3f}    | {auc:.3f}    | {rec:.3f}    | {prec:.3f}     | {f1:.3f}    | {spec:.3f}")

        # Compute ROC curve coordinates for plotting in Streamlit
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        # Sample points to keep JSON compact
        step = max(1, len(fpr) // 50)
        roc_pts = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr[::step], tpr[::step])]

        results[name] = {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4),
            "recall": round(rec, 4),
            "sensitivity": round(rec, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "roc_curve": roc_pts
        }

    return results


def train_and_export_all():
    """
    Train both full hospital clinical model and routine personal model.
    Export serialized pipelines and evaluation analytics.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    feature_importances = {}

    # ==========================================
    # 1. Full Clinical Model (13 Features)
    # ==========================================
    print("Loading Clinical Hospital Training Data...")
    X_clin, y_clin = load_clinical_training_data()
    clin_eval = evaluate_models(X_clin, y_clin, "Full Hospital Clinical (13 Biomarkers)")
    all_metrics["clinical_model"] = clin_eval

    # Fit final winning model: Soft Voting Ensemble (or Random Forest)
    # Both RF and Soft Voting achieve exceptional balance of ROC-AUC and Recall
    best_clin_pipeline = get_model_candidates(CLINICAL_FEATURES)["Soft Voting Ensemble"]
    best_clin_pipeline.fit(X_clin, y_clin)
    clin_model_path = MODELS_DIR / "hospital_clinical_pipeline.joblib"
    joblib.dump(best_clin_pipeline, clin_model_path)
    print(f"Saved Full Clinical Pipeline to {clin_model_path}")

    # Extract feature importance from the Random Forest component
    rf_clin = best_clin_pipeline.named_steps["clf"].named_estimators_["rf"]
    importances_clin = rf_clin.feature_importances_
    sorted_idx_clin = np.argsort(importances_clin)[::-1]
    feature_importances["clinical"] = [
        {
            "feature": CLINICAL_FEATURES[i],
            "label": FEATURE_LABELS.get(CLINICAL_FEATURES[i], CLINICAL_FEATURES[i]),
            "importance": round(float(importances_clin[i]), 4)
        }
        for i in sorted_idx_clin
    ]

    # ==========================================
    # 2. Routine Person Model (9 Features)
    # ==========================================
    print("\nLoading Routine Personal Screening Data...")
    X_rout, y_rout = load_routine_training_data()
    rout_eval = evaluate_models(X_rout, y_rout, "Routine Personal Screening (9 Non-Invasive Vitals)")
    all_metrics["routine_model"] = rout_eval

    # Fit final winning model for routine checkups
    best_rout_pipeline = get_model_candidates(ROUTINE_FEATURES)["Soft Voting Ensemble"]
    best_rout_pipeline.fit(X_rout, y_rout)
    rout_model_path = MODELS_DIR / "routine_person_pipeline.joblib"
    joblib.dump(best_rout_pipeline, rout_model_path)
    print(f"Saved Routine Person Pipeline to {rout_model_path}")

    # Extract feature importance from RF component
    rf_rout = best_rout_pipeline.named_steps["clf"].named_estimators_["rf"]
    importances_rout = rf_rout.feature_importances_
    sorted_idx_rout = np.argsort(importances_rout)[::-1]
    feature_importances["routine"] = [
        {
            "feature": ROUTINE_FEATURES[i],
            "label": FEATURE_LABELS.get(ROUTINE_FEATURES[i], ROUTINE_FEATURES[i]),
            "importance": round(float(importances_rout[i]), 4)
        }
        for i in sorted_idx_rout
    ]

    # Save metrics and feature importances
    metrics_path = MODELS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")

    feat_path = MODELS_DIR / "feature_importance.json"
    with open(feat_path, "w") as f:
        json.dump(feature_importances, f, indent=2)
    print(f"Saved feature importances to {feat_path}")

    print("\n[SUCCESS] Training and Serialization Completed Successfully!")


if __name__ == "__main__":
    train_and_export_all()
