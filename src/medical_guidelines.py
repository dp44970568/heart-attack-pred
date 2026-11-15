"""
Cardiology Clinical Knowledge Base and Risk Stratification Engine.
Based on ACC/AHA (American College of Cardiology / American Heart Association) guidelines.
Provides risk level classifications, symptom analysis, and actionable clinical recommendations.
"""

from typing import Dict, Any, List

def classify_risk(probability: float) -> Dict[str, Any]:
    """
    Classify predicted heart attack / coronary disease probability into clinical tiers.
    """
    prob_pct = round(probability * 100, 1)
    
    if prob_pct < 25.0:
        return {
            "tier": "Low Risk",
            "code": "LOW",
            "color": "#10B981", # Green
            "probability": prob_pct,
            "badge": "🟢 Low Probability",
            "urgency": "Routine",
            "summary": "Low probability of significant coronary artery disease or acute cardiac event."
        }
    elif prob_pct < 50.0:
        return {
            "tier": "Moderate Risk",
            "code": "MODERATE",
            "color": "#F59E0B", # Amber
            "probability": prob_pct,
            "badge": "🟡 Moderate Probability",
            "urgency": "Elective / Non-Urgent Outpatient",
            "summary": "Intermediate probability. Notable cardiovascular risk factors present requiring clinical lifestyle modification."
        }
    elif prob_pct < 75.0:
        return {
            "tier": "High Risk",
            "code": "HIGH",
            "color": "#EF4444", # Red
            "probability": prob_pct,
            "badge": "🔴 High Probability",
            "urgency": "Urgent Cardiology Consultation",
            "summary": "Substantial evidence of coronary ischemia or elevated risk of acute myocardial infarction."
        }
    else:
        return {
            "tier": "Critical Risk",
            "code": "CRITICAL",
            "color": "#7F1D1D", # Dark Red / Crimson
            "probability": prob_pct,
            "badge": "🚨 Critical Probability",
            "urgency": "Immediate Clinical / Emergency Attention",
            "summary": "Extremely high clinical probability of obstructive coronary artery disease or imminent cardiovascular event."
        }


def generate_clinical_insights(patient_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Analyze raw patient clinical parameters and flag medical risk factors according to clinical thresholds.
    """
    insights = []
    
    # Blood Pressure Analysis (AHA Hypertension guidelines)
    trestbps = patient_data.get("trestbps")
    if trestbps is not None:
        if trestbps >= 180:
            insights.append({
                "category": "Blood Pressure",
                "severity": "CRITICAL",
                "finding": f"Resting Blood Pressure is {trestbps} mm Hg (Hypertensive Crisis range).",
                "implication": "Extreme hemodynamic stress on coronary vasculature and myocardium."
            })
        elif trestbps >= 140:
            insights.append({
                "category": "Blood Pressure",
                "severity": "HIGH",
                "finding": f"Resting Blood Pressure is {trestbps} mm Hg (Stage 2 Hypertension).",
                "implication": "Chronic arterial wall shear stress accelerating atherogenesis."
            })
        elif trestbps >= 130:
            insights.append({
                "category": "Blood Pressure",
                "severity": "MODERATE",
                "finding": f"Resting Blood Pressure is {trestbps} mm Hg (Stage 1 Hypertension).",
                "implication": "Mildly elevated vascular afterload."
            })

    # Serum Cholesterol Analysis
    chol = patient_data.get("chol")
    if chol is not None and chol > 0:
        if chol >= 280:
            insights.append({
                "category": "Cholesterol",
                "severity": "HIGH",
                "finding": f"Serum Cholesterol is {chol} mg/dl (Markedly Hypercholesterolemic).",
                "implication": "High circulating atherogenic lipids promoting coronary plaque formation."
            })
        elif chol >= 240:
            insights.append({
                "category": "Cholesterol",
                "severity": "MODERATE",
                "finding": f"Serum Cholesterol is {chol} mg/dl (Elevated / Borderline High).",
                "implication": "Elevated coronary artery disease risk marker."
            })

    # ST Depression (oldpeak)
    oldpeak = patient_data.get("oldpeak")
    if oldpeak is not None:
        if oldpeak >= 2.0:
            insights.append({
                "category": "Electrocardiogram (ECG)",
                "severity": "CRITICAL",
                "finding": f"Exercise-induced ST Depression of {oldpeak} mm.",
                "implication": "Strong indicator of subendocardial myocardial ischemia under workload."
            })
        elif oldpeak >= 1.0:
            insights.append({
                "category": "Electrocardiogram (ECG)",
                "severity": "HIGH",
                "finding": f"Exercise-induced ST Depression of {oldpeak} mm.",
                "implication": "Clinically meaningful ischemic repolarization abnormality."
            })

    # Exercise Induced Angina
    exang = patient_data.get("exang")
    if exang == 1:
        insights.append({
            "category": "Symptomology",
            "severity": "HIGH",
            "finding": "Exercise-Induced Angina (Chest pain triggered by exertion).",
            "implication": "Classic sign of coronary blood supply failing to meet myocardial oxygen demand."
        })

    # Fluoroscopy Major Vessels (ca)
    ca = patient_data.get("ca")
    if ca is not None and ca > 0:
        insights.append({
            "category": "Coronary Fluoroscopy",
            "severity": "CRITICAL" if ca >= 2 else "HIGH",
            "finding": f"{int(ca)} major coronary vessel(s) showed calcification / stenosis on fluoroscopy.",
            "implication": "Direct anatomical confirmation of obstructive coronary plaque burden."
        })

    # Thallium Nuclear Perfusion (thal)
    thal = patient_data.get("thal")
    if thal is not None:
        if thal == 2 or thal == 7: # Reversible defect
            insights.append({
                "category": "Nuclear Scintigraphy (Thallium)",
                "severity": "CRITICAL",
                "finding": "Reversible Thallium Defect identified.",
                "implication": "Active, viable myocardium undergoing transient ischemia that can benefit from revascularization."
            })
        elif thal == 1 or thal == 6: # Fixed defect
            insights.append({
                "category": "Nuclear Scintigraphy (Thallium)",
                "severity": "HIGH",
                "finding": "Fixed Thallium Defect identified.",
                "implication": "Indicates previous myocardial infarction (scar tissue / non-viable myocardium)."
            })

    # Peak Heart Rate vs Age-Predicted Max
    age = patient_data.get("age")
    thalach = patient_data.get("thalach")
    if age and thalach:
        pred_max_hr = 220 - age
        pct_achieved = (thalach / pred_max_hr) * 100
        if pct_achieved < 75:
            insights.append({
                "category": "Cardiorespiratory Fitness",
                "severity": "MODERATE",
                "finding": f"Peak heart rate reached was {thalach} bpm ({pct_achieved:.0f}% of age-predicted maximum {pred_max_hr} bpm).",
                "implication": "Chronotropic incompetence or early exercise termination due to cardiac fatigue/symptoms."
            })

    # Fasting Blood Sugar
    fbs = patient_data.get("fbs")
    if fbs == 1 or (isinstance(fbs, (int, float)) and fbs > 120):
        insights.append({
            "category": "Metabolic",
            "severity": "MODERATE",
            "finding": "Fasting blood sugar > 120 mg/dl (Diabetic / Impaired fasting glucose).",
            "implication": "Metabolic dysfunction which substantially multiplies coronary atherosclerotic risk."
        })

    return insights


def generate_recommendations(risk_info: Dict[str, Any], patient_data: Dict[str, Any]) -> List[str]:
    """
    Generate actionable medical recommendations tailored to the patient's risk profile.
    """
    code = risk_info["code"]
    recs = []

    if code == "CRITICAL":
        recs.append("🚨 **Immediate Medical Action**: Recommend prompt emergency department or urgent cardiology evaluation for diagnostic angiography / cardiac catheterization.")
        recs.append("📋 **Continuous Telemetry / ECG**: Perform 12-lead ECG, troponin cardiac enzyme series, and continuous cardiac monitoring.")
        recs.append("💊 **Guideline-Directed Medical Therapy**: Evaluate for dual antiplatelet therapy (DAPT), high-intensity statin, and beta-blocker initiation under cardiologist supervision.")
    elif code == "HIGH":
        recs.append("🩺 **Urgent Outpatient Cardiology Referral**: Schedule a comprehensive cardiology evaluation within 1-2 weeks.")
        recs.append("🔬 **Advanced Diagnostics**: Consider Coronary Computed Tomography Angiography (CCTA) or Stress Echocardiogram to evaluate myocardial perfusion and coronary stenosis.")
        recs.append("💊 **Pharmacotherapy Review**: Target blood pressure < 130/80 mm Hg and LDL cholesterol reduction with lipid-lowering therapy.")
    elif code == "MODERATE":
        recs.append("📅 **Primary Care Consultation**: Follow up with primary care physician within 30 days for cardiovascular risk factor optimization.")
        recs.append("🥗 **Therapeutic Lifestyle Changes**: Adopt Mediterranean or DASH dietary pattern, reduce sodium (< 2,300 mg/day), and manage saturated fats.")
        recs.append("🏃 **Supervised Aerobic Activity**: Target at least 150 minutes per week of moderate-intensity aerobic exercise, pending physician clearance.")
        recs.append("📊 **Lipid & Glycemic Profiling**: Repeat lipid panel (ApoB/LDL-C) and HbA1c screening.")
    else: # LOW
        recs.append("✅ **Maintain Cardioprotective Lifestyle**: Continue regular physical activity, balanced whole-food diet, and adequate sleep.")
        recs.append("⏱️ **Routine Periodic Screening**: Maintain annual blood pressure checks and bi-annual lipid/glucose wellness screenings.")
        recs.append("🚭 **Primary Prevention**: Avoid tobacco exposure and maintain healthy body composition (BMI 18.5 - 24.9).")

    return recs
