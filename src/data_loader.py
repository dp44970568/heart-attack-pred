"""
Data Loading, Ingestion, and Normalization Pipeline for Heart Attack Prediction.
Handles raw medical hospital CSV formats, column alias resolution, and type conversions.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

from src.config import (
    CLINICAL_FEATURES,
    ROUTINE_FEATURES,
    COLUMN_ALIASES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR
)


def standardize_column_name(col: str) -> str:
    """Normalize raw hospital column string to standard token."""
    cleaned = col.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")
    return COLUMN_ALIASES.get(cleaned, cleaned)


def normalize_value(val: Any) -> Any:
    """Normalize categorical string representations into integers."""
    if pd.isna(val) or val is None or val == "?" or val == "":
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return val
    
    val_str = str(val).strip().lower()
    
    # Sex
    if val_str in ["m", "male", "man", "1"]:
        return 1
    if val_str in ["f", "female", "woman", "0"]:
        return 0
    
    # Binary Yes/No
    if val_str in ["yes", "y", "true", "t", "positive", "1"]:
        return 1
    if val_str in ["no", "n", "false", "f", "negative", "0"]:
        return 0
        
    # Chest pain string mapping
    if "typical" in val_str and "atypical" not in val_str:
        return 0
    if "atypical" in val_str:
        return 1
    if "non" in val_str or "sharp" in val_str:
        return 2
    if "asymptomatic" in val_str or "silent" in val_str:
        return 3
        
    # Resting ECG string mapping
    if "normal" in val_str:
        return 0
    if "st" in val_str or "wave" in val_str or "abnorm" in val_str:
        return 1
    if "hypertrophy" in val_str or "lvh" in val_str:
        return 2
        
    # Slope mapping
    if "up" in val_str:
        return 0
    if "flat" in val_str:
        return 1
    if "down" in val_str:
        return 2
        
    # Thallium mapping
    if "normal" in val_str:
        return 0
    if "fixed" in val_str:
        return 1
    if "revers" in val_str:
        return 2

    try:
        return float(val_str)
    except ValueError:
        return np.nan


def clean_hospital_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Standardize a raw hospital dataset DataFrame:
    1. Renames columns using COLUMN_ALIASES
    2. Converts values to numerical representations
    3. Retains patient_id if present
    """
    mapped_df = df.copy()
    col_mapping = {}
    
    for orig_col in df.columns:
        std_col = standardize_column_name(orig_col)
        col_mapping[orig_col] = std_col
        
    mapped_df.rename(columns=col_mapping, inplace=True)
    
    # Apply value normalizer to all candidate features
    for col in CLINICAL_FEATURES:
        if col in mapped_df.columns:
            mapped_df[col] = mapped_df[col].apply(normalize_value)
            
    return mapped_df, col_mapping


def load_clinical_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the clean full hospital clinical dataset (13 features + binary target).
    """
    filepath = PROCESSED_DATA_DIR / "hospital_cleveland_clean.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Missing clinical dataset at {filepath}")
        
    df = pd.read_csv(filepath)
    X = df[CLINICAL_FEATURES].copy()
    y = df["target"].copy()
    return X, y


def load_routine_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the routine person screening dataset (9 non-invasive vitals + binary target).
    """
    filepath = PROCESSED_DATA_DIR / "routine_person_train.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Missing routine dataset at {filepath}")
        
    df = pd.read_csv(filepath)
    X = df[ROUTINE_FEATURES].copy()
    y = df["target"].copy()
    return X, y


def generate_sample_hospital_cohort(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Generate a diverse, realistic batch of 25 hospital patients with varying clinical risk levels,
    demographics, vital signs, and diagnostic test results.
    """
    np.random.seed(42)
    cohort_data = [
        # Patient 1: High risk, symptomatic, typical acute presentation
        {"patient_id": "MED-1001", "age": 64, "gender": "Male", "chest_pain": "Typical Angina", "blood_pressure": 162, "cholesterol": 286, "fasting_blood_sugar": "Yes", "ecg": "ST-T Abnormality", "max_heart_rate": 115, "exercise_angina": "Yes", "st_depression": 2.4, "st_slope": "Flat", "major_vessels": 2, "thalassemia": "Reversible Defect", "department": "Emergency"},
        # Patient 2: Young healthy female routine checkup
        {"patient_id": "MED-1002", "age": 34, "gender": "Female", "chest_pain": "Non-Anginal Pain", "blood_pressure": 112, "cholesterol": 178, "fasting_blood_sugar": "No", "ecg": "Normal", "max_heart_rate": 178, "exercise_angina": "No", "st_depression": 0.0, "st_slope": "Upsloping", "major_vessels": 0, "thalassemia": "Normal", "department": "Preventive Care"},
        # Patient 3: Senior male with prior MI (fixed defect)
        {"patient_id": "MED-1003", "age": 71, "gender": "Male", "chest_pain": "Asymptomatic", "blood_pressure": 148, "cholesterol": 245, "fasting_blood_sugar": "No", "ecg": "Left Ventricular Hypertrophy", "max_heart_rate": 122, "exercise_angina": "Yes", "st_depression": 1.8, "st_slope": "Flat", "major_vessels": 3, "thalassemia": "Fixed Defect", "department": "Cardiology"},
        # Patient 4: Middle aged female, mild hypertension
        {"patient_id": "MED-1004", "age": 52, "gender": "Female", "chest_pain": "Atypical Angina", "blood_pressure": 134, "cholesterol": 210, "fasting_blood_sugar": "No", "ecg": "Normal", "max_heart_rate": 164, "exercise_angina": "No", "st_depression": 0.2, "st_slope": "Upsloping", "major_vessels": 0, "thalassemia": "Normal", "department": "Outpatient"},
        # Patient 5: Male with severe ischemia
        {"patient_id": "MED-1005", "age": 59, "gender": "Male", "chest_pain": "Typical Angina", "blood_pressure": 170, "cholesterol": 310, "fasting_blood_sugar": "Yes", "ecg": "ST-T Abnormality", "max_heart_rate": 105, "exercise_angina": "Yes", "st_depression": 3.2, "st_slope": "Downsloping", "major_vessels": 2, "thalassemia": "Reversible Defect", "department": "Cardiology"},
        # Patient 6: Fit athletic male
        {"patient_id": "MED-1006", "age": 42, "gender": "Male", "chest_pain": "Non-Anginal Pain", "blood_pressure": 118, "cholesterol": 185, "fasting_blood_sugar": "No", "ecg": "Normal", "max_heart_rate": 182, "exercise_angina": "No", "st_depression": 0.1, "st_slope": "Upsloping", "major_vessels": 0, "thalassemia": "Normal", "department": "Wellness"},
        # Patient 7: Postmenopausal female with diabetic risk
        {"patient_id": "MED-1007", "age": 63, "gender": "Female", "chest_pain": "Asymptomatic", "blood_pressure": 150, "cholesterol": 268, "fasting_blood_sugar": "Yes", "ecg": "Normal", "max_heart_rate": 138, "exercise_angina": "No", "st_depression": 1.2, "st_slope": "Flat", "major_vessels": 1, "thalassemia": "Normal", "department": "Endocrinology"},
        # Patient 8: Young male with non-cardiac chest discomfort
        {"patient_id": "MED-1008", "age": 28, "gender": "Male", "chest_pain": "Non-Anginal Pain", "blood_pressure": 120, "cholesterol": 165, "fasting_blood_sugar": "No", "ecg": "Normal", "max_heart_rate": 190, "exercise_angina": "No", "st_depression": 0.0, "st_slope": "Upsloping", "major_vessels": 0, "thalassemia": "Normal", "department": "Outpatient"},
        # Patient 9: Elderly female, borderline high risk
        {"patient_id": "MED-1009", "age": 68, "gender": "Female", "chest_pain": "Typical Angina", "blood_pressure": 155, "cholesterol": 290, "fasting_blood_sugar": "No", "ecg": "ST-T Abnormality", "max_heart_rate": 128, "exercise_angina": "Yes", "st_depression": 2.0, "st_slope": "Flat", "major_vessels": 1, "thalassemia": "Reversible Defect", "department": "Cardiology"},
        # Patient 10: Middle-aged executive, stress-induced symptoms
        {"patient_id": "MED-1010", "age": 49, "gender": "Male", "chest_pain": "Atypical Angina", "blood_pressure": 138, "cholesterol": 235, "fasting_blood_sugar": "No", "ecg": "Normal", "max_heart_rate": 158, "exercise_angina": "No", "st_depression": 0.6, "st_slope": "Upsloping", "major_vessels": 0, "thalassemia": "Normal", "department": "Executive Health"},
        # Patient 11: Patient in hypertensive urgency
        {"patient_id": "MED-1011", "age": 62, "gender": "Male", "chest_pain": "Typical Angina", "blood_pressure": 182, "cholesterol": 275, "fasting_blood_sugar": "Yes", "ecg": "Left Ventricular Hypertrophy", "max_heart_rate": 118, "exercise_angina": "Yes", "st_depression": 2.8, "st_slope": "Downsloping", "major_vessels": 3, "thalassemia": "Reversible Defect", "department": "Emergency"},
        # Patient 12: Female with atypical sharp pain
        {"patient_id": "MED-1012", "age": 45, "gender": "Female", "chest_pain": "Atypical Angina", "blood_pressure": 124, "cholesterol": 195, "fasting_blood_sugar": "No", "ecg": "Normal", "max_heart_rate": 168, "exercise_angina": "No", "st_depression": 0.0, "st_slope": "Upsloping", "major_vessels": 0, "thalassemia": "Normal", "department": "Outpatient"},
        # Patient 13: Asymptomatic senior with multivessel disease
        {"patient_id": "MED-1013", "age": 73, "gender": "Male", "chest_pain": "Asymptomatic", "blood_pressure": 142, "cholesterol": 252, "fasting_blood_sugar": "Yes", "ecg": "ST-T Abnormality", "max_heart_rate": 110, "exercise_angina": "Yes", "st_depression": 1.6, "st_slope": "Flat", "major_vessels": 2, "thalassemia": "Fixed Defect", "department": "Cardiology"},
        # Patient 14: Healthy adult female
        {"patient_id": "MED-1014", "age": 39, "gender": "Female", "chest_pain": "Non-Anginal Pain", "blood_pressure": 115, "cholesterol": 182, "fasting_blood_sugar": "No", "ecg": "Normal", "max_heart_rate": 174, "exercise_angina": "No", "st_depression": 0.0, "st_slope": "Upsloping", "major_vessels": 0, "thalassemia": "Normal", "department": "Preventive Care"},
        # Patient 15: Critical acute coronary syndrome presentation
        {"patient_id": "MED-1015", "age": 66, "gender": "Male", "chest_pain": "Typical Angina", "blood_pressure": 175, "cholesterol": 320, "fasting_blood_sugar": "Yes", "ecg": "ST-T Abnormality", "max_heart_rate": 98, "exercise_angina": "Yes", "st_depression": 3.8, "st_slope": "Downsloping", "major_vessels": 3, "thalassemia": "Reversible Defect", "department": "Emergency"}
    ]
    
    df = pd.DataFrame(cohort_data)
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "sample_hospital_batch.csv"
        
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    sample_df = generate_sample_hospital_cohort()
    print(f"Generated sample hospital cohort with {len(sample_df)} patients at {PROCESSED_DATA_DIR / 'sample_hospital_batch.csv'}")
