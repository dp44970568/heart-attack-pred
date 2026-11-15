# ❤️ CardioGuard AI — Heart Attack Risk Prediction & Hospital Triage System

An end-to-end clinical machine learning platform designed to predict acute myocardial infarction (heart attack) and coronary artery disease risk from raw hospital diagnostic records or personal health metrics.

---

## 🌟 Key Features

1. **Dual Clinical & Personal Diagnostic Modes**:
   - **For a Person (Routine Checkup)**: Uses 9 non-invasive vitals (Age, Sex, Chest Discomfort, Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Peak Exercise Heart Rate, Exercise Angina). Ideal for individual screening at home or outpatient triage where invasive imaging is unavailable.
   - **For Medical Hospitals & Emergency Departments**: Uses all 13-14 clinical diagnostic biomarkers, including coronary fluoroscopy vessel narrowing (`ca`), nuclear thallium myocardial perfusion scintigraphy (`thal`), and ST depression / slope (`oldpeak`, `slope`).
2. **Automated Schema & Column Mapping**:
   - Hospital CSVs often use varied column headers (`patient_age`, `gender`, `blood_pressure`, `bp`, `serum_chol`, etc.) and text representations (`Male`/`Female`, `Yes`/`No`, `Typical Angina`). CardioGuard automatically detects, cleans, standardizes, and infers on raw hospital datasets.
3. **Calibrated Machine Learning Ensemble**:
   - Multi-algorithm soft voting ensemble combining **XGBoost**, **Random Forest**, and **Calibrated Logistic Regression**.
   - Achieves **0.909 ROC-AUC**, **84.8% Accuracy**, and **90.2% Specificity** on clinical benchmarks with 5-Fold Stratified Cross-Validation.
4. **Clinical Risk Stratification & Explainability**:
   - Categorizes risk into 4 ACC/AHA tiers: **Low Risk (<25%)**, **Moderate Risk (25-50%)**, **High Risk (50-75%)**, and **Critical Risk (>75%)**.
   - Identifies specific clinical drivers (e.g. ST segment depression, multi-vessel stenosis, stage 2 hypertension).
   - Generates guideline-directed medical therapy recommendations based on ACC/AHA cardiology standards.
5. **Interactive Streamlit Web Dashboard**:
   - Real-time risk gauge, biomarker attribution waterfall, printable patient summary reports.
   - Bulk drag-and-drop CSV upload with cohort triage analytics and exportable diagnosed CSVs.
6. **Production FastAPI REST Service & CLI**:
   - High-throughput asynchronous REST API with OpenAPI documentation (`/docs`).
   - Interactive terminal questionnaire and CLI batch runner.

---

## 📊 Model Performance Benchmarks (5-Fold Stratified CV)

### 🏥 Full Hospital Clinical Pipeline (13 Biomarkers)
*Trained on multi-center clinical hospital databases (Cleveland Clinic Foundation).*

| Algorithm | ROC-AUC | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Soft Voting Ensemble (Winner)** | **0.909** | **83.5%** | **78.4%** | **87.8%** | **0.813** | **0.117** |
| **Random Forest** | 0.908 | 84.8% | 78.4% | 90.2% | 0.826 | 0.119 |
| **Logistic Regression (Calibrated)** | 0.908 | 82.8% | 79.9% | 85.4% | 0.810 | 0.125 |
| **XGBoost** | 0.903 | 82.8% | 79.1% | 86.0% | 0.809 | 0.126 |
| **Hist Gradient Boosting** | 0.896 | 82.8% | 79.9% | 85.4% | 0.810 | 0.129 |

---

### 👤 Routine Personal Screening Pipeline (9 Non-Invasive Vitals)
*Trained on 661 patients across Cleveland Clinic, Hungarian Institute of Cardiology, and VA Medical Center.*

| Algorithm | ROC-AUC | Accuracy | Sensitivity (Recall) | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Soft Voting Ensemble (Winner)** | **0.882** | **80.2%** | **75.2%** | **84.7%** | **0.783** |
| **Logistic Regression (Calibrated)** | 0.884 | 80.2% | 79.0% | 81.3% | 0.791 |
| **Random Forest** | 0.876 | 79.9% | 75.8% | 83.6% | 0.782 |
| **XGBoost** | 0.878 | 79.7% | 74.8% | 84.1% | 0.778 |

---

## 📁 Repository Structure

```text
heart_attack_pred/
├── data/
│   ├── raw/
│   │   └── uci_heart_combined.csv        # 920 records from 4 hospitals
│   └── processed/
│       ├── hospital_cleveland_clean.csv  # 303 gold-standard clinical records
│       ├── routine_person_train.csv      # 661 non-invasive vitals records
│       └── sample_hospital_batch.csv     # Ready-to-use sample batch for hospital testing
├── models/
│   ├── hospital_clinical_pipeline.joblib # Full clinical pipeline
│   ├── routine_person_pipeline.joblib   # Routine screening pipeline
│   ├── model_metrics.json               # Cross-validation performance metrics
│   └── feature_importance.json          # Gini feature importances
├── src/
│   ├── __init__.py
│   ├── config.py                         # Clinical ranges, schema, column aliases
│   ├── data_loader.py                    # Raw CSV cleaner, normalizer, sample generator
│   ├── train.py                          # Cross-validation, model training & export
│   ├── predictor.py                      # Production inference engine & router
│   └── medical_guidelines.py             # ACC/AHA cardiology rules & clinical advice
├── app.py                                # Streamlit Web Application (UI)
├── api.py                                # FastAPI REST API service
├── cli.py                                # Command-Line Interface (Interactive & Batch)
├── tests/
│   └── test_model.py                     # Unit and integration test suite
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Launch the Streamlit Web Application
Run the interactive dashboard in your browser:
```bash
streamlit run app.py
```
- Navigate to **"Personal Health Screening"** to evaluate individual risk with intuitive sliders and real-time guidance.
- Navigate to **"Hospital Diagnostic & Batch Cohort"** to test single patient clinical catheterization values or click **"Load Sample Hospital Cohort (15 Patients)"** to run instant cohort batch triage!

---

### 2. Run the FastAPI REST Server
Start the high-performance REST API:
```bash
python api.py
# Or with uvicorn directly:
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
- Open your browser to `http://127.0.0.1:8000/docs` to explore interactive Swagger OpenAPI documentation.

#### Example API Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 62,
    "sex": 1,
    "cp": 0,
    "trestbps": 160,
    "chol": 280,
    "fbs": 1,
    "restecg": 1,
    "thalach": 115,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 2,
    "ca": 2,
    "thal": 2
  }'
```

---

### 3. Use the Command-Line Interface (CLI)

**Interactive Questionnaire (For an individual person):**
```bash
python cli.py
```

**Batch Hospital CSV Evaluation (For hospitals):**
```bash
python cli.py --batch data/processed/sample_hospital_batch.csv --output hospital_triage_results.csv
```

---

### 4. Run the Unit & Integration Tests
```bash
python -m unittest tests/test_model.py
```
*All 8 tests verify preprocessing, clinical model routing, batch processing, and REST API endpoints.*

---

## 🩺 Clinical Reference Ranges (ACC/AHA Guidelines)

| Clinical Parameter | Normal / Desirable | Borderline / Elevated | High Risk / Urgent |
| :--- | :--- | :--- | :--- |
| **Resting Blood Pressure** | < 120 mm Hg | 120 - 139 mm Hg (Stage 1) | ≥ 140 mm Hg (Stage 2 / Crisis) |
| **Serum Cholesterol** | < 200 mg/dl | 200 - 239 mg/dl | ≥ 240 mg/dl |
| **Fasting Blood Sugar** | 70 - 99 mg/dl | 100 - 125 mg/dl (Prediabetes) | ≥ 126 mg/dl (Diabetic) |
| **ST Depression (Oldpeak)** | 0.0 mm | 0.1 - 0.9 mm | ≥ 1.0 mm (Subendocardial Ischemia) |
| **Coronary Vessels (`ca`)** | 0 vessels narrowed | 1 vessel | ≥ 2 vessels (Multivessel Disease) |
| **Thallium Perfusion (`thal`)**| Normal Blood Flow | Fixed Defect (Prior Scar) | Reversible Defect (Active Ischemia) |
