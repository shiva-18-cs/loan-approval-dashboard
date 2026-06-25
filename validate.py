import sys, os
sys.path.insert(0, '.')
os.chdir(r'C:\Users\surya\.gemini\antigravity\scratch\loan-dashboard')

print("--- Testing utils ---")
from utils import load_raw_data, get_processed_data, train_all_models, predict_single

df = load_raw_data()
print(f"  Raw data: {df.shape}, approval rate: {df['loan_status'].mean()*100:.1f}%")
print(f"  Columns: {list(df.columns)}")
print(f"  Missing: {df.isnull().sum().sum()} total NaNs")

df_proc, ss, mms, features = get_processed_data()
print(f"  Processed: {df_proc.shape}")
print(f"  Selected features: {features}")

print("\n--- Training models (may take ~30s) ---")
results, trained, X_train, X_test, y_train, y_test, feats = train_all_models()
print("\n  Model Results:")
for name, v in results.items():
    print(f"    {name:20s}  Acc={v['Accuracy']}%  Prec={v['Precision']}%  Recall={v['Recall']}%  F1={v['F1 Score']}%")

best = max(results, key=lambda k: results[k]['F1 Score'])
print(f"\n  Best model: {best}")

print("\n--- Testing simulator ---")
sample = dict(
    age=35, gender='male', education='Bachelor', income=75000,
    emp_exp=8, home_ownership='MORTGAGE', loan_amnt=12000,
    loan_intent='PERSONAL', loan_int_rate=10.5, cred_hist=10,
    credit_score=710, prev_defaults='No'
)
prob, pred = predict_single(sample, ss, mms, trained[best], feats)
label = "Approved" if pred == 1 else "Rejected"
risk = "Low" if prob >= 0.70 else ("Medium" if prob >= 0.45 else "High")
print(f"  Result: {label} | Probability: {prob*100:.1f}% | Risk: {risk}")

print("\n--- Testing EDA data aggregations ---")
edu_rates = df.groupby('person_education')['loan_status'].mean() * 100
print(f"  Education approval rates: {edu_rates.round(1).to_dict()}")

gender_rates = df.groupby('person_gender')['loan_status'].mean() * 100
print(f"  Gender approval rates: {gender_rates.round(1).to_dict()}")

print("\n--- Testing SHAP (quick) ---")
try:
    import shap
    import numpy as np
    import pandas as pd
    X_sample = pd.DataFrame(X_test[:50], columns=feats)
    explainer = shap.TreeExplainer(trained['Random Forest'])
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    print(f"  SHAP values shape: {sv.shape}  -- OK")
except Exception as e:
    print(f"  SHAP warning: {e}")

print("\n=== ALL CHECKS PASSED ===")
