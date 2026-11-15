"""
Production Inference and Clinical Risk Prediction Engine.
Handles automatic model routing (Hospital Clinical vs Routine Person),
calibrated risk score computation, feature attribution, and batch hospital processing.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

from src.config import (
    CLINICAL_FEATURES,
    ROUTINE_FEATURES,
    FEATURE_LABELS,
    MODELS_DIR,
    CHEST_PAIN_MAP,
    RESTECG_MAP,
    SLOPE_MAP,
    THAL_MAP
)
from src.data_loader import (
    clean_hospital_dataframe,
    normalize_value,
    standardize_column_name
)
from src.medical_guidelines import (
    classify_risk,
    generate_clinical_insights,
    generate_recommendations
)


class HeartRiskPredictor:
    """
    Intelligent Heart Attack & Coronary Disease Risk Predictor.
    Automatically detects available clinical markers and routes to either:
    1. Full Hospital Clinical Pipeline (13 Biomarkers)
    2. Routine Personal Checkup Pipeline (9 Non-Invasive Vitals)
    """

    def __init__(self):
        self.clinical_model_path = MODELS_DIR / "hospital_clinical_pipeline.joblib"
        self.routine_model_path = MODELS_DIR / "routine_person_pipeline.joblib"

        if not self.clinical_model_path.exists() or not self.routine_model_path.exists():
            raise FileNotFoundError(
                f"Trained models not found in {MODELS_DIR}. Run 'python src/train.py' first."
            )

        self.clinical_pipeline = joblib.load(self.clinical_model_path)
        self.routine_pipeline = joblib.load(self.routine_model_path)

    def _prepare_patient_features(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize patient dictionary keys and values."""
        normalized = {}
        for k, v in patient_data.items():
            std_k = standardize_column_name(k)
            norm_v = normalize_value(v)
            normalized[std_k] = norm_v
        return normalized

    def _determine_mode(self, clean_data: Dict[str, Any]) -> str:
        """
        Determine if patient has invasive diagnostic markers (fluoroscopy ca, thallium thal, slope).
        If present, route to 'clinical'; otherwise route to 'routine'.
        """
        has_ca = clean_data.get("ca") is not None and not pd.isna(clean_data.get("ca"))
        has_thal = clean_data.get("thal") is not None and not pd.isna(clean_data.get("thal"))
        has_slope = clean_data.get("slope") is not None and not pd.isna(clean_data.get("slope"))

        if has_ca and has_thal and has_slope:
            return "clinical"
        return "routine"

    def predict_single(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate clinical heart attack prediction for a single individual or hospital patient.
        """
        clean_data = self._prepare_patient_features(patient_data)
        mode = self._determine_mode(clean_data)

        if mode == "clinical":
            features = CLINICAL_FEATURES
            pipeline = self.clinical_pipeline
            mode_name = "Full Hospital Clinical Pipeline (13 Biomarkers)"
        else:
            features = ROUTINE_FEATURES
            pipeline = self.routine_pipeline
            mode_name = "Routine Personal Screening Pipeline (9 Non-Invasive Vitals)"

        # Construct single-row DataFrame
        input_row = {feat: clean_data.get(feat, np.nan) for feat in features}
        df_input = pd.DataFrame([input_row])

        # Get calibrated probability
        prob = float(pipeline.predict_proba(df_input)[0, 1])

        # Risk classification
        risk_info = classify_risk(prob)

        # Clinical insights & recommendations
        insights = generate_clinical_insights(clean_data)
        recommendations = generate_recommendations(risk_info, clean_data)

        # Compute feature contributions (deviation from sample baseline)
        feature_contributions = self._compute_feature_attributions(clean_data, mode)

        return {
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100, 1),
            "risk_tier": risk_info["tier"],
            "risk_code": risk_info["code"],
            "risk_color": risk_info["color"],
            "risk_badge": risk_info["badge"],
            "urgency": risk_info["urgency"],
            "summary": risk_info["summary"],
            "model_mode": mode_name,
            "features_used": features,
            "feature_contributions": feature_contributions,
            "clinical_insights": insights,
            "recommendations": recommendations,
            "raw_input": patient_data
        }

    def _compute_feature_attributions(self, clean_data: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
        """
        Calculate relative risk directional impact of patient's biomarkers
        compared to typical healthy reference norms.
        """
        contributions = []

        # 1. Chest Pain
        cp = clean_data.get("cp")
        if cp is not None and not pd.isna(cp):
            if cp == 0:  # Typical angina
                contributions.append({"factor": "Chest Pain (Typical Angina)", "impact": "High Risk Increase", "direction": "risk_up", "weight": 2.2})
            elif cp == 1:
                contributions.append({"factor": "Chest Pain (Atypical Angina)", "impact": "Moderate Risk Increase", "direction": "risk_up", "weight": 1.2})
            elif cp == 3:
                contributions.append({"factor": "Asymptomatic / Silent Ischemia", "impact": "Silent Risk Factor", "direction": "neutral", "weight": 0.5})
            else:
                contributions.append({"factor": "Non-Anginal Discomfort", "impact": "Low Risk Contribution", "direction": "risk_down", "weight": -0.8})

        # 2. ST Depression
        oldpeak = clean_data.get("oldpeak")
        if oldpeak is not None and not pd.isna(oldpeak):
            if oldpeak >= 2.0:
                contributions.append({"factor": f"ST Depression ({oldpeak} mm)", "impact": "Severe Ischemia Risk", "direction": "risk_up", "weight": 2.5})
            elif oldpeak >= 1.0:
                contributions.append({"factor": f"ST Depression ({oldpeak} mm)", "impact": "Moderate Ischemia Risk", "direction": "risk_up", "weight": 1.5})
            elif oldpeak == 0.0:
                contributions.append({"factor": "Normal ST Segment (0.0 mm)", "impact": "Protective Normal Wave", "direction": "risk_down", "weight": -1.2})

        # 3. Exercise Angina
        exang = clean_data.get("exang")
        if exang is not None and not pd.isna(exang):
            if exang == 1:
                contributions.append({"factor": "Exercise-Induced Angina", "impact": "High Risk Trigger", "direction": "risk_up", "weight": 1.8})
            else:
                contributions.append({"factor": "No Angina on Exercise", "impact": "Cardiorespiratory Reserve", "direction": "risk_down", "weight": -1.0})

        # 4. Max Heart Rate
        thalach = clean_data.get("thalach")
        age = clean_data.get("age", 55)
        if thalach is not None and not pd.isna(thalach):
            pred_max = 220 - (age if not pd.isna(age) else 55)
            pct = (thalach / pred_max) * 100 if pred_max > 0 else 100
            if pct < 75:
                contributions.append({"factor": f"Low Peak HR ({thalach} bpm)", "impact": "Chronotropic Impairment", "direction": "risk_up", "weight": 1.4})
            elif pct >= 90:
                contributions.append({"factor": f"High Peak HR ({thalach} bpm)", "impact": "Strong Cardiac Capacity", "direction": "risk_down", "weight": -1.5})

        # 5. Fluoroscopy Vessels (if clinical)
        ca = clean_data.get("ca")
        if ca is not None and not pd.isna(ca) and ca > 0:
            contributions.append({"factor": f"Stenosed Vessels ({int(ca)} blocked)", "impact": "Major Anatomical Occlusion", "direction": "risk_up", "weight": 2.4})

        # 6. Thallium Perfusion (if clinical)
        thal = clean_data.get("thal")
        if thal is not None and not pd.isna(thal):
            if thal == 2 or thal == 7:
                contributions.append({"factor": "Reversible Thallium Defect", "impact": "Active Reversible Ischemia", "direction": "risk_up", "weight": 2.3})
            elif thal == 1 or thal == 6:
                contributions.append({"factor": "Fixed Thallium Defect", "impact": "Myocardial Scar Tissue", "direction": "risk_up", "weight": 1.6})
            elif thal == 0 or thal == 3:
                contributions.append({"factor": "Normal Myocardial Perfusion", "impact": "Intact Coronary Perfusion", "direction": "risk_down", "weight": -1.5})

        # 7. Blood Pressure
        bp = clean_data.get("trestbps")
        if bp is not None and not pd.isna(bp):
            if bp >= 140:
                contributions.append({"factor": f"Hypertensive BP ({bp} mmHg)", "impact": "Arterial Shear Stress", "direction": "risk_up", "weight": 1.1})
            elif bp < 120:
                contributions.append({"factor": f"Optimal BP ({bp} mmHg)", "impact": "Optimal Vascular Health", "direction": "risk_down", "weight": -0.7})

        # 8. Cholesterol
        chol = clean_data.get("chol")
        if chol is not None and not pd.isna(chol) and chol > 0:
            if chol >= 240:
                contributions.append({"factor": f"Hypercholesterolemia ({chol} mg/dl)", "impact": "Atherogenic Plaque Risk", "direction": "risk_up", "weight": 0.8})
            elif chol < 200:
                contributions.append({"factor": f"Desirable Cholesterol ({chol} mg/dl)", "impact": "Low Atherogenic Burden", "direction": "risk_down", "weight": -0.6})

        # Sort by absolute weight
        contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)
        return contributions

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform batch prediction on an entire cohort from any hospital CSV.
        Maintains all original columns and appends clinical risk predictions.
        """
        output_df = df.copy()
        clean_df, _ = clean_hospital_dataframe(df)

        probs = []
        tiers = []
        codes = []
        urgencies = []
        top_factors = []
        modes = []

        for idx, row in clean_df.iterrows():
            row_dict = row.to_dict()
            pred = self.predict_single(row_dict)

            probs.append(pred["probability_pct"])
            tiers.append(pred["risk_tier"])
            codes.append(pred["risk_code"])
            urgencies.append(pred["urgency"])
            modes.append(pred["model_mode"].split(" (")[0])

            # Top 2 primary risk contributors
            up_factors = [c["factor"] for c in pred["feature_contributions"] if c["direction"] == "risk_up"]
            top_factors.append(", ".join(up_factors[:2]) if up_factors else "None flagged")

        output_df["heart_attack_risk_pct"] = probs
        output_df["risk_tier"] = tiers
        output_df["triage_urgency"] = urgencies
        output_df["primary_risk_factors"] = top_factors
        output_df["evaluation_model"] = modes

        return output_df


# Quick singleton instance helper
_predictor_instance = None

def get_predictor() -> HeartRiskPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HeartRiskPredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = get_predictor()
    print("Testing single personal screening:")
    person_sample = {
        "age": 55,
        "sex": 1,
        "chest_pain": 0, # Typical angina
        "blood_pressure": 150,
        "cholesterol": 260,
        "fasting_blood_sugar": 0,
        "ecg": 0,
        "max_heart_rate": 125,
        "exercise_angina": 1,
        "st_depression": 1.8
    }
    res = predictor.predict_single(person_sample)
    print(f"Personal Risk: {res['risk_tier']} ({res['probability_pct']}%)")
    print(f"Mode: {res['model_mode']}")
    print("Top factors:", [f['factor'] for f in res['feature_contributions'][:3]])

    print("\nTesting hospital batch prediction:")
    sample_csv = MODELS_DIR.parent / "data" / "processed" / "sample_hospital_batch.csv"
    if sample_csv.exists():
        batch_input = pd.read_csv(sample_csv)
        batch_out = predictor.predict_batch(batch_input)
        print("Batch predictions output:")
        print(batch_out[["patient_id", "age", "heart_attack_risk_pct", "risk_tier", "triage_urgency"]].head(5))
