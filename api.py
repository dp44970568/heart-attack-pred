"""
CardioPredict - FastAPI Production Service for Heart Attack & Coronary Risk Prediction.
Serves real-time REST endpoints for individual patients and bulk hospital batch data.
"""

import sys
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from src.predictor import get_predictor
from src.config import MODELS_DIR, FEATURE_LABELS, MEDICAL_RANGES


app = FastAPI(
    title="CardioPredict Clinical AI API",
    description="Machine Learning API for Heart Attack Risk Prediction based on Hospital Records and Personal Health Vitals.",
    version="1.0.0"
)

# Enable CORS for web and client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    patient_id: Optional[str] = Field(None, example="PATIENT-001")
    age: float = Field(..., ge=1.0, le=120.0, example=58.0)
    sex: Union[int, str] = Field(..., example=1, description="1 for Male, 0 for Female, or 'Male'/'Female'")
    cp: Optional[Union[int, str]] = Field(None, example=0, description="Chest pain type (0-3 or text description)")
    trestbps: Optional[float] = Field(None, ge=50.0, le=300.0, example=140.0, description="Resting blood pressure in mm Hg")
    chol: Optional[float] = Field(None, ge=50.0, le=800.0, example=245.0, description="Serum cholesterol in mg/dl")
    fbs: Optional[Union[int, str]] = Field(None, example=0, description="Fasting blood sugar > 120 mg/dl (1/0, Yes/No)")
    restecg: Optional[Union[int, str]] = Field(None, example=0, description="Resting ECG (0=Normal, 1=ST abnormality, 2=LVH)")
    thalach: Optional[float] = Field(None, ge=40.0, le=250.0, example=135.0, description="Maximum heart rate achieved")
    exang: Optional[Union[int, str]] = Field(None, example=1, description="Exercise induced angina (1/0, Yes/No)")
    oldpeak: Optional[float] = Field(0.0, ge=0.0, le=10.0, example=1.5, description="ST depression induced by exercise (mm)")
    # Hospital Clinical Optional Markers
    slope: Optional[Union[int, str]] = Field(None, example=1, description="Slope of peak exercise ST segment (0-2)")
    ca: Optional[Union[int, str]] = Field(None, example=1, description="Major vessels colored by fluoroscopy (0-3)")
    thal: Optional[Union[int, str]] = Field(None, example=2, description="Thallium scintigraphy defect (0=Normal, 1=Fixed, 2=Reversible)")


class BatchInput(BaseModel):
    patients: List[PatientInput]


@app.get("/")
def root():
    return {
        "service": "CardioPredict - Heart Attack Risk Prediction API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict_single": "POST /predict",
            "predict_batch_json": "POST /predict/batch",
            "predict_batch_csv": "POST /predict/upload-csv"
        }
    }


@app.get("/health")
def health_check():
    predictor = get_predictor()
    return {
        "status": "healthy",
        "clinical_model_loaded": predictor.clinical_pipeline is not None,
        "routine_model_loaded": predictor.routine_pipeline is not None
    }


@app.get("/model/info")
def model_info():
    metrics_path = MODELS_DIR / "model_metrics.json"
    feat_path = MODELS_DIR / "feature_importance.json"

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    features = {}
    if feat_path.exists():
        with open(feat_path, "r", encoding="utf-8") as f:
            features = json.load(f)

    return {
        "models": {
            "full_hospital_clinical": {
                "algorithm": "Soft Voting Ensemble (Random Forest + XGBoost + Calibrated Logistic Regression)",
                "features_count": 13,
                "metrics": metrics.get("clinical_model", {}).get("Soft Voting Ensemble", {})
            },
            "routine_personal_screening": {
                "algorithm": "Soft Voting Ensemble (Random Forest + XGBoost + Calibrated Logistic Regression)",
                "features_count": 9,
                "metrics": metrics.get("routine_model", {}).get("Soft Voting Ensemble", {})
            }
        },
        "top_features": features,
        "clinical_reference_ranges": MEDICAL_RANGES
    }


@app.post("/predict")
def predict_single_patient(patient: PatientInput):
    """Predict risk for a single individual or hospital patient."""
    try:
        predictor = get_predictor()
        patient_dict = patient.model_dump(exclude_none=True) if hasattr(patient, "model_dump") else patient.dict(exclude_none=True)
        result = predictor.predict_single(patient_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
def predict_batch_patients(batch: BatchInput):
    """Predict risk for a list of patients in JSON."""
    try:
        predictor = get_predictor()
        patient_dicts = [p.model_dump(exclude_none=True) if hasattr(p, "model_dump") else p.dict(exclude_none=True) for p in batch.patients]
        df = pd.DataFrame(patient_dicts)
        out_df = predictor.predict_batch(df)
        return {
            "total_records": len(out_df),
            "results": out_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch error: {str(e)}")


@app.post("/predict/upload-csv")
async def predict_upload_csv(file: UploadFile = File(...)):
    """Upload a raw hospital CSV file and receive analyzed risk predictions."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        predictor = get_predictor()
        result_df = predictor.predict_batch(df)

        return {
            "filename": file.filename,
            "total_patients": len(result_df),
            "triage_summary": result_df["risk_tier"].value_counts().to_dict(),
            "urgency_summary": result_df["triage_urgency"].value_counts().to_dict(),
            "patients": result_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
