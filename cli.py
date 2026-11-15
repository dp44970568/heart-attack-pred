"""
Command-Line Interface (CLI) for Heart Attack Risk Prediction.
Supports both interactive terminal questionnaire and batch CSV prediction.
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.predictor import get_predictor


def run_interactive_person_mode():
    """Terminal questionnaire for an individual person."""
    print("=" * 65)
    print("      HEART ATTACK & CARDIOVASCULAR RISK PREDICTOR")
    print("              Routine Personal Screening Mode")
    print("=" * 65)
    print("Please answer the following health metrics (press Enter for defaults):\n")

    try:
        age = float(input("1. Age in years [e.g. 55]: ") or 55)
        
        sex_input = input("2. Biological Sex (M/F or 1/0) [M]: ") or "M"
        sex = 1 if sex_input.strip().upper().startswith("M") or sex_input == "1" else 0

        print("\n3. Chest Pain Experience:")
        print("   [0] Typical Angina (chest pressure/squeezing provoked by physical effort)")
        print("   [1] Atypical Angina (discomfort not completely typical)")
        print("   [2] Non-Anginal Pain (sharp, fleeting, or positional chest pain)")
        print("   [3] Asymptomatic (no chest discomfort)")
        cp_in = input("   Select (0-3) [0]: ") or "0"
        cp = int(cp_in)

        trestbps = float(input("\n4. Resting Blood Pressure in mm Hg [e.g. 135]: ") or 135)
        chol = float(input("5. Serum Cholesterol in mg/dl [e.g. 230]: ") or 230)
        
        fbs_in = input("6. Fasting Blood Sugar > 120 mg/dl? (Y/N) [N]: ") or "N"
        fbs = 1 if fbs_in.strip().upper().startswith("Y") else 0

        print("\n7. Resting Electrocardiogram (ECG):")
        print("   [0] Normal")
        print("   [1] ST-T Wave Abnormality")
        print("   [2] Left Ventricular Hypertrophy")
        restecg_in = input("   Select (0-2) [0]: ") or "0"
        restecg = int(restecg_in)

        thalach = float(input("\n8. Maximum Heart Rate Achieved under exercise (bpm) [e.g. 140]: ") or 140)

        exang_in = input("9. Do you experience angina/chest tightness when exercising? (Y/N) [N]: ") or "N"
        exang = 1 if exang_in.strip().upper().startswith("Y") else 0

        oldpeak = float(input("10. ST Depression on stress test (Oldpeak in mm, 0 if unknown) [0.0]: ") or 0.0)

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

        print("\nCalculating clinical risk profile...\n")
        predictor = get_predictor()
        res = predictor.predict_single(patient_data)

        print("=" * 65)
        print(f" RESULT: {res['risk_tier'].upper()} ({res['probability_pct']}%)")
        print(f" URGENCY LEVEL: {res['urgency']}")
        print(f" EVALUATION ENGINE: {res['model_mode']}")
        print("=" * 65)
        print(f"\nClinical Summary:\n{res['summary']}\n")

        print("Primary Factors Influencing Your Risk Score:")
        for c in res["feature_contributions"][:4]:
            arrow = "[+] HIGHER RISK" if c["direction"] == "risk_up" else "[-] LOWER RISK" if c["direction"] == "risk_down" else "[*] NEUTRAL"
            print(f"  {arrow:<16} : {c['factor']} ({c['impact']})")

        if res["clinical_insights"]:
            print("\nFlagged Clinical Findings:")
            for item in res["clinical_insights"]:
                print(f"  - [{item['severity']}] {item['finding']}")
                print(f"    Clinical context: {item['implication']}")

        print("\nActionable Medical Recommendations:")
        for rec in res["recommendations"]:
            # Strip markdown formatting for terminal
            clean_rec = rec.replace("**", "").replace("🚨 ", "").replace("🩺 ", "").replace("💊 ", "").replace("✅ ", "").replace("📅 ", "").replace("🥗 ", "").replace("🏃 ", "").replace("📊 ", "").replace("🚭 ", "")
            print(f"  * {clean_rec}")

        print("=" * 65)

    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")


def run_batch_file_mode(input_path: str, output_path: str):
    """Run batch prediction on hospital CSV."""
    print(f"Loading hospital records from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Read {len(df)} patient records. Running clinical inference...")
    
    predictor = get_predictor()
    output_df = predictor.predict_batch(df)

    output_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Batch predictions written to {output_path}")
    print("\nTriage Summary:")
    print(output_df["risk_tier"].value_counts())
    print("\nUrgency Summary:")
    print(output_df["triage_urgency"].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Heart Attack Risk Predictor CLI")
    parser.add_argument("--batch", type=str, help="Path to input hospital CSV file for batch diagnosis")
    parser.add_argument("--output", type=str, default="hospital_predictions_output.csv", help="Path to save output CSV")
    args = parser.parse_args()

    if args.batch:
        run_batch_file_mode(args.batch, args.output)
    else:
        run_interactive_person_mode()


if __name__ == "__main__":
    main()
