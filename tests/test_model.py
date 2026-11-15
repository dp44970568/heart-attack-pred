"""
Unit and Integration Test Suite for CardioPredict.
Verifies data preprocessing, model inference, clinical guidelines, and API endpoints.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import pandas as pd
import numpy as np

from src.data_loader import (
    standardize_column_name,
    normalize_value,
    clean_hospital_dataframe
)
from src.medical_guidelines import (
    classify_risk,
    generate_clinical_insights,
    generate_recommendations
)
from src.predictor import get_predictor
from fastapi.testclient import TestClient
from api import app


class TestHeartAttackPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()
        cls.client = TestClient(app)

    def test_column_standardization(self):
        self.assertEqual(standardize_column_name("Blood Pressure"), "trestbps")
        self.assertEqual(standardize_column_name("resting_bp"), "trestbps")
        self.assertEqual(standardize_column_name("patient_age"), "age")
        self.assertEqual(standardize_column_name("gender"), "sex")
        self.assertEqual(standardize_column_name("serum_cholesterol"), "chol")

    def test_value_normalization(self):
        self.assertEqual(normalize_value("Male"), 1)
        self.assertEqual(normalize_value("Female"), 0)
        self.assertEqual(normalize_value("Yes"), 1)
        self.assertEqual(normalize_value("No"), 0)
        self.assertEqual(normalize_value("typical angina"), 0)
        self.assertEqual(normalize_value("atypical angina"), 1)
        self.assertEqual(normalize_value("reversible defect"), 2)

    def test_risk_classification_tiers(self):
        low = classify_risk(0.15)
        self.assertEqual(low["code"], "LOW")
        self.assertEqual(low["tier"], "Low Risk")

        mod = classify_risk(0.40)
        self.assertEqual(mod["code"], "MODERATE")

        high = classify_risk(0.65)
        self.assertEqual(high["code"], "HIGH")

        crit = classify_risk(0.85)
        self.assertEqual(crit["code"], "CRITICAL")

    def test_person_mode_prediction(self):
        """Test routine screening with 9 non-invasive vitals (no fluoroscopy, no thallium)."""
        healthy_person = {
            "age": 32,
            "sex": 0,
            "cp": 2, # Non-anginal
            "trestbps": 115,
            "chol": 175,
            "fbs": 0,
            "restecg": 0,
            "thalach": 175,
            "exang": 0,
            "oldpeak": 0.0
        }
        res = self.predictor.predict_single(healthy_person)
        self.assertIn("probability", res)
        self.assertIn("risk_tier", res)
        self.assertIn("Routine Personal Screening", res["model_mode"])
        # Healthy 32-year-old should be Low Risk
        self.assertEqual(res["risk_tier"], "Low Risk")
        self.assertLess(res["probability_pct"], 35.0)

    def test_hospital_clinical_mode_prediction(self):
        """Test full hospital diagnostic with 13 markers including catheterization."""
        severe_patient = {
            "age": 67,
            "sex": 1,
            "cp": 0, # Typical angina
            "trestbps": 165,
            "chol": 290,
            "fbs": 1,
            "restecg": 1,
            "thalach": 110,
            "exang": 1,
            "oldpeak": 3.0,
            "slope": 2, # Downsloping
            "ca": 2, # 2 blocked vessels
            "thal": 2 # Reversible defect
        }
        res = self.predictor.predict_single(severe_patient)
        self.assertIn("Full Hospital Clinical", res["model_mode"])
        # Severe multi-vessel patient should be High or Critical Risk
        self.assertIn(res["risk_tier"], ["High Risk", "Critical Risk"])
        self.assertGreater(res["probability_pct"], 70.0)
        self.assertGreaterEqual(len(res["clinical_insights"]), 3)

    def test_batch_hospital_prediction(self):
        """Test processing a batch hospital cohort."""
        raw_hospital_cohort = pd.DataFrame([
            {"patient_id": "P01", "Age": 68, "Gender": "Male", "Blood_Pressure": 160, "Cholesterol": 280, "Chest_Pain": "Typical Angina", "Max_Heart_Rate": 115, "Exercise_Angina": "Yes", "ST_Depression": 2.5, "Major_Vessels": 2, "Thalassemia": "Reversible Defect", "ST_Slope": "Flat", "Fasting_Blood_Sugar": "Yes", "Resting_ECG": "ST Abnormality"},
            {"patient_id": "P02", "Age": 29, "Gender": "Female", "Blood_Pressure": 110, "Cholesterol": 165, "Chest_Pain": "Non-Anginal", "Max_Heart_Rate": 185, "Exercise_Angina": "No", "ST_Depression": 0.0, "Major_Vessels": 0, "Thalassemia": "Normal", "ST_Slope": "Upsloping", "Fasting_Blood_Sugar": "No", "Resting_ECG": "Normal"}
        ])
        out_df = self.predictor.predict_batch(raw_hospital_cohort)
        self.assertEqual(len(out_df), 2)
        self.assertIn("heart_attack_risk_pct", out_df.columns)
        self.assertIn("risk_tier", out_df.columns)
        self.assertIn("triage_urgency", out_df.columns)
        # P01 should be critical/high, P02 should be low
        self.assertGreater(out_df.loc[0, "heart_attack_risk_pct"], out_df.loc[1, "heart_attack_risk_pct"])

    def test_api_health_and_info(self):
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")

        res_info = self.client.get("/model/info")
        self.assertEqual(res_info.status_code, 200)
        data = res_info.json()
        self.assertIn("models", data)
        self.assertIn("full_hospital_clinical", data["models"])

    def test_api_predict_endpoint(self):
        payload = {
            "age": 60,
            "sex": 1,
            "cp": 0,
            "trestbps": 150,
            "chol": 260,
            "thalach": 120,
            "exang": 1,
            "oldpeak": 2.0
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("probability_pct", data)
        self.assertIn("risk_tier", data)
        self.assertIn("recommendations", data)


if __name__ == "__main__":
    unittest.main()
