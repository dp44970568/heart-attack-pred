"""
CardioGuard AI - Heart Attack Risk Prediction & Hospital Triage Platform.
Interactive Streamlit Web Application supporting both individual person assessments
and medical hospital raw batch cohort analysis.
"""

import sys
import io
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.predictor import get_predictor
from src.config import (
    CLINICAL_FEATURES,
    ROUTINE_FEATURES,
    FEATURE_LABELS,
    CHEST_PAIN_MAP,
    RESTECG_MAP,
    SLOPE_MAP,
    THAL_MAP,
    MEDICAL_RANGES,
    MODELS_DIR,
    PROCESSED_DATA_DIR
)
from src.medical_guidelines import classify_risk

# Streamlit Page Config
st.set_page_config(
    page_title="CardioGuard AI - Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-banner {
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Predictor
@st.cache_resource
def load_cached_predictor():
    return get_predictor()

predictor = load_cached_predictor()

# Sidebar Information
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.title("CardioGuard AI")
    st.markdown("**Clinical Decision Support System**")
    st.markdown("Engineered for **Hospitals, Clinicians & Individual Screening**.")
    
    st.divider()
    st.subheader("System Status")
    st.success("Clinical Model: Active (AUC: 0.909)")
    st.success("Routine Model: Active (AUC: 0.884)")
    st.info("Algorithms: Soft Voting Ensemble (XGBoost + Random Forest + Calibrated Logistic Regression)")

    st.divider()
    st.markdown("""
    **Reference Guidelines:**
    - ACC/AHA 2019 Primary Prevention of CVD
    - 2021 AHA/ACC Chest Pain Guidelines
    - Cleveland Clinic & Multi-Center Hospital Study
    """)


# Main Header
st.markdown('<div class="main-header">❤️ CardioGuard Clinical AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Acute Myocardial Infarction & Coronary Artery Disease Risk Stratification</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "👤 Personal Health Screening (For a Person)",
    "🏥 Hospital Diagnostic & Batch Cohort (For Hospitals)",
    "📊 Model Performance & Clinical Analytics"
])


# ==============================================================================
# TAB 1: PERSONAL HEALTH SCREENING
# ==============================================================================
with tab1:
    st.markdown("### Routine Cardiovascular Health Screening")
    st.markdown(
        "Enter non-invasive routine medical vitals to estimate personal heart attack risk. "
        "No invasive laboratory catheterization or nuclear imaging required."
    )

    with st.form("personal_screening_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 1. Demographics & Symptoms")
            age = st.slider("Age (years)", min_value=18, max_value=95, value=55, help="Patient biological age")
            sex_choice = st.radio("Biological Sex", ["Male", "Female"], horizontal=True)
            sex = 1 if sex_choice == "Male" else 0

            cp_options = {
                "Typical Angina (Chest tightness / pressure on exertion)": 0,
                "Atypical Angina (Chest discomfort not typical)": 1,
                "Non-Anginal Pain (Sharp, localized or fleeting ache)": 2,
                "Asymptomatic (No current chest pain)": 3
            }
            cp_label = st.selectbox("Chest Pain Experience", list(cp_options.keys()), index=0)
            cp = cp_options[cp_label]

            exang_choice = st.radio("Exercise-Induced Angina (Chest pain on exertion)", ["No", "Yes"], horizontal=True)
            exang = 1 if exang_choice == "Yes" else 0

        with col2:
            st.markdown("##### 2. Hemodynamics & Metabolic Vitals")
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80,
                max_value=240,
                value=135,
                help="AHA Guidelines: Normal <120, Elevated 120-129, Stage 1: 130-139, Stage 2: >=140"
            )
            # Live BP category badge
            if trestbps < 120:
                st.caption("🟢 AHA Range: Normal (<120 mmHg)")
            elif trestbps < 130:
                st.caption("🟡 AHA Range: Elevated (120-129 mmHg)")
            elif trestbps < 140:
                st.caption("🟠 AHA Range: Stage 1 Hypertension (130-139 mmHg)")
            else:
                st.caption("🔴 AHA Range: Stage 2 Hypertension (≥140 mmHg)")

            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=100,
                max_value=550,
                value=230,
                help="Clinical Range: Desirable <200, Borderline 200-239, High >=240"
            )
            if chol < 200:
                st.caption("🟢 Desirable (<200 mg/dl)")
            elif chol < 240:
                st.caption("🟡 Borderline High (200-239 mg/dl)")
            else:
                st.caption("🔴 High Cholesterol (≥240 mg/dl)")

            fbs_choice = st.radio("Fasting Blood Sugar > 120 mg/dl (Diabetic threshold)", ["No (Normal / Pre-diabetic)", "Yes (> 120 mg/dl)"], horizontal=True)
            fbs = 1 if "Yes" in fbs_choice else 0

        with col3:
            st.markdown("##### 3. Cardiac Function & Stress Response")
            pred_max_hr = 220 - age
            thalach = st.slider(
                f"Maximum Heart Rate Achieved (bpm)",
                min_value=60,
                max_value=220,
                value=min(145, pred_max_hr),
                help=f"Age-predicted maximum for age {age} is approximately {pred_max_hr} bpm."
            )
            pct_achieved = (thalach / pred_max_hr) * 100 if pred_max_hr > 0 else 100
            st.caption(f"Achieved {pct_achieved:.0f}% of age-predicted peak ({pred_max_hr} bpm)")

            restecg_options = {
                "Normal Resting ECG": 0,
                "ST-T Wave Abnormality (T-wave inversion / ST changes)": 1,
                "Left Ventricular Hypertrophy (LVH)": 2
            }
            restecg_label = st.selectbox("Resting Electrocardiogram (ECG)", list(restecg_options.keys()), index=0)
            restecg = restecg_options[restecg_label]

            oldpeak = st.slider(
                "ST Depression (Oldpeak in mm)",
                min_value=0.0,
                max_value=6.0,
                value=0.8,
                step=0.1,
                help="ST segment depression induced by exercise relative to rest. > 1.0 mm is clinically significant for ischemia."
            )
            if oldpeak == 0.0:
                st.caption("🟢 Normal (No ST depression)")
            elif oldpeak < 1.0:
                st.caption("🟡 Mild ST depression (<1.0 mm)")
            else:
                st.caption("🔴 Significant Ischemia Marker (≥1.0 mm)")

        submit_person = st.form_submit_button("⚡ Evaluate Cardiovascular Risk Profile", use_container_width=True)

    if submit_person:
        patient_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak
        }

        with st.spinner("Analyzing cardiovascular biomarkers..."):
            result = predictor.predict_single(patient_data)

        # Risk Banner
        bg_color = result["risk_color"]
        st.markdown(f"""
        <div class="risk-banner" style="background-color: {bg_color};">
            <h2 style="margin: 0; color: white;">{result['risk_badge']} — {result['risk_tier'].upper()} ({result['probability_pct']}%)</h2>
            <p style="font-size: 1.15rem; margin: 8px 0 0 0; color: white;"><b>Clinical Urgency:</b> {result['urgency']}</p>
        </div>
        """, unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("#### 🔬 Key Risk Factor Breakdown")
            st.markdown("Biomarker contribution to individual risk score:")
            
            for item in result["feature_contributions"][:5]:
                if item["direction"] == "risk_up":
                    st.error(f"⬆️ **{item['factor']}**: {item['impact']}")
                elif item["direction"] == "risk_down":
                    st.success(f"⬇️ **{item['factor']}**: {item['impact']}")
                else:
                    st.info(f"➡️ **{item['factor']}**: {item['impact']}")

            if result["clinical_insights"]:
                st.markdown("#### ⚠️ Clinical Findings Flagged")
                for finding in result["clinical_insights"]:
                    st.warning(f"**[{finding['category']}]**: {finding['finding']}\n\n*Physiological implication:* {finding['implication']}")

        with res_col2:
            st.markdown("#### 📋 Actionable Cardiology Recommendations")
            st.markdown(f"*{result['summary']}*")
            for rec in result["recommendations"]:
                st.markdown(f"- {rec}")

            st.divider()
            st.markdown("#### 📄 Export Patient Assessment")
            summary_text = f"""==================================================
CARDIOGUARD CLINICAL RISK ASSESSMENT REPORT
==================================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Patient Demographics: Age {age}, Biological Sex: {sex_choice}

VITAL SIGNS & CLINICAL VALUES:
- Resting Blood Pressure: {trestbps} mm Hg
- Serum Cholesterol: {chol} mg/dl
- Fasting Blood Sugar: {'>120 mg/dl' if fbs == 1 else '<=120 mg/dl'}
- Chest Pain Presentation: {cp_label}
- Exercise-Induced Angina: {'Present' if exang == 1 else 'Absent'}
- Peak Heart Rate: {thalach} bpm ({pct_achieved:.0f}% of max {pred_max_hr} bpm)
- ST Depression: {oldpeak} mm

RISK STRATIFICATION RESULTS:
- Risk Tier: {result['risk_tier']}
- Probability: {result['probability_pct']}%
- Clinical Urgency: {result['urgency']}
- Diagnostic Pipeline: {result['model_mode']}

ACTIONABLE RECOMMENDATIONS:
""" + "\n".join([f"- {r.replace('**', '').replace('🚨 ', '').replace('🩺 ', '').replace('💊 ', '').replace('✅ ', '')}" for r in result["recommendations"]]) + """

==================================================
NOTICE: Generated for clinical decision support.
Confirm with licensed physician / cardiologist.
==================================================
"""
            st.download_button(
                label="📥 Download Clinical Summary Report (.txt)",
                data=summary_text,
                file_name=f"cardiovascular_risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )


# ==============================================================================
# TAB 2: HOSPITAL DIAGNOSTIC & BATCH COHORT
# ==============================================================================
with tab2:
    st.markdown("### Medical Hospital Diagnostic Suite")
    st.markdown(
        "Supports full clinical diagnostics (including coronary fluoroscopy and nuclear perfusion scintigraphy) "
        "as well as batch cohort triage for hospitals and emergency departments."
    )

    hospital_mode = st.radio(
        "Select Hospital Operation Mode:",
        ["📁 Batch Hospital Cohort Analysis (CSV File)", "🩺 Comprehensive Single Patient Diagnostic (Full 13 Biomarkers)"],
        horizontal=True
    )

    if hospital_mode == "📁 Batch Hospital Cohort Analysis (CSV File)":
        st.markdown("#### Hospital Cohort Batch Ingestion & Triage")
        st.markdown(
            "Upload any raw hospital CSV containing patient vitals. CardioGuard automatically maps hospital column variations "
            "(e.g., `blood_pressure`, `bp`, `gender`, `chest_pain`), handles text values, and computes cohort risk triage."
        )

        bcol1, bcol2 = st.columns([2, 1])

        with bcol1:
            uploaded_file = st.file_uploader("Upload Raw Hospital CSV File", type=["csv"])

        with bcol2:
            st.markdown("**Test with Hospital Sample:**")
            load_sample = st.button("📥 Load Sample Hospital Cohort (15 Patients)", use_container_width=True)

        batch_df = None
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded hospital dataset with {len(batch_df)} patient records.")
        elif load_sample:
            sample_path = PROCESSED_DATA_DIR / "sample_hospital_batch.csv"
            if sample_path.exists():
                batch_df = pd.read_csv(sample_path)
                st.info("Loaded sample hospital cohort (15 patient admissions across Emergency, Cardiology, and Wellness).")

        if batch_df is not None:
            with st.spinner("Processing hospital cohort and performing clinical inference..."):
                analyzed_df = predictor.predict_batch(batch_df)

            # High-level Hospital Triage Metrics
            total_pts = len(analyzed_df)
            crit_count = (analyzed_df["risk_tier"] == "Critical Risk").sum()
            high_count = (analyzed_df["risk_tier"] == "High Risk").sum()
            mod_count = (analyzed_df["risk_tier"] == "Moderate Risk").sum()
            low_count = (analyzed_df["risk_tier"] == "Low Risk").sum()

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Cohort Patients", total_pts)
            kpi2.metric("🚨 Critical Risk (Immediate)", f"{crit_count} ({crit_count/total_pts*100:.0f}%)")
            kpi3.metric("🔴 High Risk (Urgent)", f"{high_count} ({high_count/total_pts*100:.0f}%)")
            kpi4.metric("🟢 Low / Moderate Risk", f"{low_count + mod_count} ({(low_count+mod_count)/total_pts*100:.0f}%)")

            # Cohort Charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("##### Risk Tier Distribution")
                tier_counts = analyzed_df["risk_tier"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 3.5))
                colors = {"Critical Risk": "#7F1D1D", "High Risk": "#EF4444", "Moderate Risk": "#F59E0B", "Low Risk": "#10B981"}
                bar_colors = [colors.get(t, "#6B7280") for t in tier_counts.index]
                ax.bar(tier_counts.index, tier_counts.values, color=bar_colors, edgecolor="black", alpha=0.85)
                ax.set_ylabel("Patient Count")
                ax.set_title("Cohort Risk Stratification")
                plt.xticks(rotation=15)
                st.pyplot(fig)

            with chart_col2:
                st.markdown("##### Heart Attack Probability Spread (%)")
                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                ax2.hist(analyzed_df["heart_attack_risk_pct"], bins=10, color="#2563EB", edgecolor="white", alpha=0.8)
                ax2.set_xlabel("Predicted Probability (%)")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Distribution of Predicted Risk Scores")
                st.pyplot(fig2)

            # Filterable Table
            st.markdown("#### 📋 Cohort Patient Roster & Triage Decisions")
            filter_tier = st.multiselect("Filter by Risk Tier:", ["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"], default=["Critical Risk", "High Risk", "Moderate Risk", "Low Risk"])
            
            filtered_df = analyzed_df[analyzed_df["risk_tier"].isin(filter_tier)]
            
            # Display columns nicely
            st.dataframe(filtered_df, use_container_width=True)

            # Download Enriched Hospital CSV
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export Diagnosed Hospital Cohort CSV",
                data=csv_buffer.getvalue(),
                file_name="cardiovascular_hospital_cohort_diagnosed.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.markdown("#### Comprehensive Single Patient Diagnostic (Full 13 Biomarkers)")
        st.markdown("Includes fluoroscopy catheterization and thallium stress scintigraphy for complete hospital assessment.")

        with st.form("hospital_full_form"):
            hcol1, hcol2, hcol3 = st.columns(3)

            with hcol1:
                st.markdown("##### Demographics & Angina")
                h_pid = st.text_input("Patient ID / Medical Record #", "HOSP-8891")
                h_age = st.slider("Patient Age", 18, 95, 62)
                h_sex = st.radio("Biological Sex", ["Male", "Female"], horizontal=True, key="h_sex")
                h_cp = st.selectbox(
                    "Chest Pain Type",
                    ["0: Typical Angina", "1: Atypical Angina", "2: Non-Anginal Pain", "3: Asymptomatic"],
                    index=0
                )
                h_exang = st.radio("Exercise-Induced Angina", ["No", "Yes"], horizontal=True, key="h_exang")

            with hcol2:
                st.markdown("##### Hemodynamics & Labs")
                h_trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 240, 155)
                h_chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 550, 275)
                h_fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], horizontal=True, key="h_fbs")
                h_thalach = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 115)
                h_oldpeak = st.slider("ST Depression (mm)", 0.0, 6.0, 2.2, step=0.1)

            with hcol3:
                st.markdown("##### Invasive & Nuclear Imaging")
                h_restecg = st.selectbox(
                    "Resting ECG Result",
                    ["0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"],
                    index=1
                )
                h_slope = st.selectbox(
                    "Peak Exercise ST Slope",
                    ["0: Upsloping (normal)", "1: Flat (ischemia)", "2: Downsloping (severe ischemia)"],
                    index=1
                )
                h_ca = st.selectbox(
                    "Major Vessels Colored by Fluoroscopy (0-3)",
                    [0, 1, 2, 3],
                    index=2,
                    help="Number of major coronary vessels with >50% stenosis"
                )
                h_thal = st.selectbox(
                    "Thallium Scintigraphy Perfusion Test",
                    ["0: Normal Blood Flow", "1: Fixed Defect (Previous Infarct Scar)", "2: Reversible Defect (Active Ischemia)"],
                    index=2
                )

            submit_hospital = st.form_submit_button("⚡ Run Full Hospital Diagnostic Evaluation", use_container_width=True)

        if submit_hospital:
            h_data = {
                "patient_id": h_pid,
                "age": h_age,
                "sex": 1 if h_sex == "Male" else 0,
                "cp": int(h_cp.split(":")[0]),
                "trestbps": h_trestbps,
                "chol": h_chol,
                "fbs": 1 if h_fbs == "Yes" else 0,
                "restecg": int(h_restecg.split(":")[0]),
                "thalach": h_thalach,
                "exang": 1 if h_exang == "Yes" else 0,
                "oldpeak": h_oldpeak,
                "slope": int(h_slope.split(":")[0]),
                "ca": h_ca,
                "thal": int(h_thal.split(":")[0])
            }

            with st.spinner("Running full clinical diagnostic pipeline..."):
                h_res = predictor.predict_single(h_data)

            st.markdown(f"""
            <div class="risk-banner" style="background-color: {h_res['risk_color']};">
                <h2 style="margin: 0; color: white;">Patient {h_pid}: {h_res['risk_badge']} — {h_res['risk_tier'].upper()} ({h_res['probability_pct']}%)</h2>
                <p style="font-size: 1.15rem; margin: 8px 0 0 0; color: white;"><b>Triage Recommendation:</b> {h_res['urgency']}</p>
                <p style="font-size: 0.95rem; margin: 4px 0 0 0; color: white;">Pipeline: {h_res['model_mode']}</p>
            </div>
            """, unsafe_allow_html=True)

            hc1, hc2 = st.columns(2)
            with hc1:
                st.markdown("##### 🔍 Identified Clinical Drivers")
                for c in h_res["feature_contributions"][:6]:
                    if c["direction"] == "risk_up":
                        st.error(f"⬆️ **{c['factor']}**: {c['impact']}")
                    else:
                        st.success(f"⬇️ **{c['factor']}**: {c['impact']}")

            with hc2:
                st.markdown("##### 🏥 Cardiologist Action Plan")
                for r in h_res["recommendations"]:
                    st.markdown(f"- {r}")


# ==============================================================================
# TAB 3: MODEL PERFORMANCE & CLINICAL ANALYTICS
# ==============================================================================
with tab3:
    st.markdown("### Model Architecture, Clinical Benchmarks & Validation")
    st.markdown(
        "CardioGuard was evaluated with 5-Fold Stratified Cross-Validation on authentic hospital patient datasets "
        "incorporating the Cleveland Clinic Foundation, Hungarian Institute of Cardiology, and VA Medical Center."
    )

    metrics_path = MODELS_DIR / "model_metrics.json"
    feat_path = MODELS_DIR / "feature_importance.json"

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_data = json.load(f)

        mcol1, mcol2 = st.columns(2)

        with mcol1:
            st.markdown("#### Full Hospital Clinical Model (13 Biomarkers)")
            clin_dict = metrics_data.get("clinical_model", {})
            clin_summary = []
            for name, stats in clin_dict.items():
                clin_summary.append({
                    "Algorithm": name,
                    "Accuracy": f"{stats['accuracy']*100:.1f}%",
                    "ROC-AUC": f"{stats['roc_auc']:.3f}",
                    "Recall / Sensitivity": f"{stats['recall']*100:.1f}%",
                    "Specificity": f"{stats['specificity']*100:.1f}%",
                    "F1 Score": f"{stats['f1_score']:.3f}"
                })
            st.dataframe(pd.DataFrame(clin_summary), use_container_width=True)

        with mcol2:
            st.markdown("#### Routine Personal Screening Model (9 Vitals)")
            rout_dict = metrics_data.get("routine_model", {})
            rout_summary = []
            for name, stats in rout_dict.items():
                rout_summary.append({
                    "Algorithm": name,
                    "Accuracy": f"{stats['accuracy']*100:.1f}%",
                    "ROC-AUC": f"{stats['roc_auc']:.3f}",
                    "Recall / Sensitivity": f"{stats['recall']*100:.1f}%",
                    "Specificity": f"{stats['specificity']*100:.1f}%",
                    "F1 Score": f"{stats['f1_score']:.3f}"
                })
            st.dataframe(pd.DataFrame(rout_summary), use_container_width=True)

    # Global Feature Importance Chart
    if feat_path.exists():
        with open(feat_path) as f:
            feat_data = json.load(f)

        st.divider()
        st.markdown("#### 🎯 Global Feature Importance in Clinical Diagnostics")
        st.markdown("Identified using Random Forest Gini impurity reduction across all training validation folds:")

        clin_feats = feat_data.get("clinical", [])
        if clin_feats:
            f_df = pd.DataFrame(clin_feats).sort_values("importance", ascending=True)
            fig_fi, ax_fi = plt.subplots(figsize=(10, 4.5))
            ax_fi.barh(f_df["label"], f_df["importance"], color="#3B82F6", edgecolor="black", alpha=0.85)
            ax_fi.set_xlabel("Relative Feature Importance Weight")
            ax_fi.set_title("Cardiology Feature Importance Ranking (Full Clinical Model)")
            st.pyplot(fig_fi)

    st.divider()
    st.markdown("#### 📚 Medical Biomarker Clinical Reference Guide")
    st.markdown("""
    | Marker | Medical Name | Normal Range | Clinical Significance |
    | :--- | :--- | :--- | :--- |
    | **trestbps** | Resting Blood Pressure | < 120 mm Hg | Elevated BP increases myocardial wall stress and promotes coronary atherosclerosis |
    | **chol** | Serum Cholesterol | < 200 mg/dl | Atherogenic low-density lipoproteins cause coronary plaque accumulation |
    | **fbs** | Fasting Blood Sugar | ≤ 120 mg/dl | Hyperglycemia induces endothelial dysfunction and accelerates cardiovascular events |
    | **thalach** | Peak Heart Rate | ~ 220 - Age | Submaximal peak HR indicates chronotropic incompetence or ischemic angina limits |
    | **oldpeak** | ST Depression | 0.0 mm | ST depression > 1.0 mm is a hallmark of exertional subendocardial myocardial ischemia |
    | **ca** | Fluoroscopy Vessels | 0 vessels | Number of major coronary arteries with >50% diameter stenosis |
    | **thal** | Thallium Scintigraphy | Normal | Fixed defect denotes prior myocardial infarction scar; Reversible defect denotes active ischemia |
    """)

st.caption("CardioGuard AI — Research & Clinical Decision Support System. Developed for hospital data pipelines and patient risk triage.")
