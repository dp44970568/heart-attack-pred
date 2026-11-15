"""
Configuration and Medical Standards for Heart Attack Risk Prediction.
Defines feature schema, medical reference ranges, and hospital raw column aliases.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Feature definitions for Full Clinical Hospital Model (13 features)
CLINICAL_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

# Feature definitions for Routine / Personal Checkup Model (9 non-invasive vitals)
ROUTINE_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak"
]

# Human-readable labels for features
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Biological Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Result",
    "thalach": "Maximum Heart Rate Achieved (bpm)",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Oldpeak in mm)",
    "slope": "Slope of Peak Exercise ST Segment",
    "ca": "Major Vessels Colored by Fluoroscopy (0-3)",
    "thal": "Thallium Stress Test Scintigraphy"
}

# Categorical mappings
CHEST_PAIN_MAP = {
    0: "Typical Angina (substernal chest pain provoked by exertion)",
    1: "Atypical Angina (chest discomfort not fully typical)",
    2: "Non-Anginal Pain (sharp or localized chest discomfort)",
    3: "Asymptomatic / Silent Ischemia (no chest pain detected)"
}

RESTECG_MAP = {
    0: "Normal",
    1: "ST-T Wave Abnormality (T wave inversions / ST changes)",
    2: "Left Ventricular Hypertrophy (LVH by Estes' criteria)"
}

SLOPE_MAP = {
    0: "Upsloping (normal physiologic exercise response)",
    1: "Flat (ischemia indicator)",
    2: "Downsloping (strong myocardial ischemia marker)"
}

THAL_MAP = {
    0: "Normal Blood Flow (normal myocardial perfusion)",
    1: "Fixed Defect (scar tissue from prior myocardial infarction)",
    2: "Reversible Defect (active inducible ischemia / restricted blood flow)"
}

# Hospital Raw Column Aliases for Flexible CSV Ingestion
COLUMN_ALIASES = {
    # Age
    "age": "age",
    "patient_age": "age",
    "age_years": "age",
    # Sex
    "sex": "sex",
    "gender": "sex",
    "male": "sex",
    # Chest Pain
    "cp": "cp",
    "chest_pain": "cp",
    "chest_pain_type": "cp",
    "chestpain": "cp",
    "angina_type": "cp",
    # Blood Pressure
    "trestbps": "trestbps",
    "blood_pressure": "trestbps",
    "resting_bp": "trestbps",
    "systolic_bp": "trestbps",
    "resting_blood_pressure": "trestbps",
    "bp": "trestbps",
    # Cholesterol
    "chol": "chol",
    "cholesterol": "chol",
    "serum_chol": "chol",
    "serum_cholesterol": "chol",
    # Fasting Blood Sugar
    "fbs": "fbs",
    "blood_sugar": "fbs",
    "fasting_blood_sugar": "fbs",
    "glucose": "fbs",
    "fasting_glucose": "fbs",
    # Resting ECG
    "restecg": "restecg",
    "resting_ecg": "restecg",
    "ecg": "restecg",
    "electrocardiogram": "restecg",
    # Max Heart Rate
    "thalach": "thalach",
    "max_heart_rate": "thalach",
    "max_hr": "thalach",
    "heart_rate": "thalach",
    "peak_hr": "thalach",
    # Exercise Angina
    "exang": "exang",
    "exercise_angina": "exang",
    "exercise_induced_angina": "exang",
    # ST Depression
    "oldpeak": "oldpeak",
    "st_depression": "oldpeak",
    "st_dep": "oldpeak",
    # Slope
    "slope": "slope",
    "st_slope": "slope",
    "slope_st": "slope",
    # Major Vessels (Fluoroscopy)
    "ca": "ca",
    "major_vessels": "ca",
    "vessels": "ca",
    "fluoroscopy": "ca",
    "colored_vessels": "ca",
    # Thallium Scintigraphy
    "thal": "thal",
    "thalassemia": "thal",
    "thallium": "thal",
    "thallium_scan": "thal",
    "perfusion": "thal",
    # Target / Diagnosis
    "target": "target",
    "heart_disease": "target",
    "condition": "target",
    "diagnosis": "target",
    "output": "target",
    "patient_id": "patient_id",
    "id": "patient_id"
}

# American Heart Association / ACC Clinical Benchmark Ranges
MEDICAL_RANGES = {
    "blood_pressure": {
        "normal": (90, 119),
        "elevated": (120, 129),
        "stage1_hypertension": (130, 139),
        "stage2_hypertension": (140, 180),
        "hypertensive_crisis": (180, 260)
    },
    "cholesterol": {
        "desirable": (120, 199),
        "borderline": (200, 239),
        "high": (240, 600)
    },
    "fasting_blood_sugar": {
        "normal": (70, 99),
        "prediabetes": (100, 125),
        "diabetes_threshold": 126
    },
    "st_depression": {
        "normal": (0.0, 0.5),
        "mild_ischemia": (0.6, 1.0),
        "significant_ischemia": (1.1, 2.0),
        "severe_ischemia": (2.1, 8.0)
    }
}
