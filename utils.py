"""
utils.py  –  Shared data loading, preprocessing, and model training helpers.
All heavy computations are cached with @st.cache_data / @st.cache_resource.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────── constants ───────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "loan_data_v2.csv")

SKEWED_COLS = [
    'person_age', 'person_income', 'person_emp_exp',
    'loan_amnt', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score'
]
NORM_COLS = ['loan_int_rate']

GENDER_MAP       = {'male': 0, 'female': 1}
HOME_MAP         = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
INTENT_MAP       = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2,
                    'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
DEFAULTS_MAP     = {'No': 0, 'Yes': 1}
EDUCATION_MAP    = {'High School': 0, 'Associate': 1, 'Bachelor': 2,
                    'Master': 3, 'Doctorate': 4}

FEATURE_DESCRIPTIONS = {
    'person_age':                    'Age of the loan applicant (years)',
    'person_gender':                 'Gender of the applicant',
    'person_education':              'Highest education level attained',
    'person_income':                 'Annual income in USD',
    'person_emp_exp':                'Years of employment experience',
    'person_home_ownership':         'Home ownership status (Rent / Own / Mortgage)',
    'loan_amnt':                     'Requested loan amount in USD',
    'loan_intent':                   'Purpose of the loan',
    'loan_int_rate':                 'Annual interest rate (%)',
    'loan_percent_income':           'Loan amount as a fraction of income',
    'cb_person_cred_hist_length':    'Credit history length (years)',
    'credit_score':                  'FICO-style credit score (300–850)',
    'previous_loan_defaults_on_file':'Whether the applicant has prior defaults',
    'loan_status':                   'Target: 1 = Approved, 0 = Rejected',
}

# ─────────────────────────── raw data ────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load (or generate) the raw dataset."""
    if not os.path.exists(DATA_PATH):
        _generate_and_save()
    return pd.read_csv(DATA_PATH)


def _generate_and_save():
    """Create synthetic data if the CSV is missing."""
    from generate_data import generate_loan_data
    df = generate_loan_data()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)


# ─────────────────────────── preprocessing ───────────────────────
@st.cache_data(show_spinner=False)
def get_processed_data():
    """
    Returns (df_raw, df_processed, ss, mms, high_corr_features).
    Mirrors the notebook pipeline exactly.
    """
    df = load_raw_data().copy()

    # --- type coercion ---
    df['person_age'] = df['person_age'].astype(int)

    # --- impute ---
    df['person_income'] = df['person_income'].fillna(df['person_income'].median())
    df['person_emp_exp'] = df['person_emp_exp'].fillna(df['person_emp_exp'].median())

    # --- encoding ---
    df['person_gender']                   = df['person_gender'].map(GENDER_MAP)
    df['person_home_ownership']           = df['person_home_ownership'].map(HOME_MAP)
    df['loan_intent']                     = df['loan_intent'].map(INTENT_MAP)
    df['previous_loan_defaults_on_file']  = df['previous_loan_defaults_on_file'].map(DEFAULTS_MAP)
    df['person_education']                = df['person_education'].map(EDUCATION_MAP)

    # --- scaling ---
    ss  = StandardScaler()
    mms = MinMaxScaler()
    df[SKEWED_COLS] = ss.fit_transform(df[SKEWED_COLS])
    df[NORM_COLS]   = mms.fit_transform(df[NORM_COLS])

    # --- outlier trimming (IQR, right tail) only on numerical ---
    numeric_cols = SKEWED_COLS + NORM_COLS
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        df = df[df[col] <= upper]

    df.reset_index(drop=True, inplace=True)

    # --- feature selection (correlation > 0.05) ---
    corr = df.corr()
    high_corr = corr.index[abs(corr['loan_status']) > 0.05].tolist()
    high_corr = [c for c in high_corr if c != 'loan_status']

    return df, ss, mms, high_corr


# ─────────────────────────── model training ──────────────────────
@st.cache_resource(show_spinner=False)
def train_all_models():
    """
    Train LR, SVM, KNN, RF and return results dict + trained RF for SHAP.
    """
    df_proc, ss, mms, high_corr = get_processed_data()

    X = df_proc[high_corr]
    y = df_proc['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM':                 SVC(probability=True, random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=3),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    trained = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        results[name] = {
            'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            'Recall':    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            'F1 Score':  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            'Confusion': confusion_matrix(y_test, y_pred),
        }
        trained[name] = mdl

    return results, trained, X_train, X_test, y_train, y_test, high_corr


# ─────────────────────────── simulator helper ────────────────────
def predict_single(input_dict: dict, ss: StandardScaler,
                   mms: MinMaxScaler, model, feature_cols: list):
    """
    Encode + scale a single applicant dict and return (probability, prediction).
    """
    row = {
        'person_age':                    input_dict['age'],
        'person_gender':                 GENDER_MAP[input_dict['gender']],
        'person_education':              EDUCATION_MAP[input_dict['education']],
        'person_income':                 input_dict['income'],
        'person_emp_exp':                input_dict['emp_exp'],
        'person_home_ownership':         HOME_MAP[input_dict['home_ownership']],
        'loan_amnt':                     input_dict['loan_amnt'],
        'loan_intent':                   INTENT_MAP[input_dict['loan_intent']],
        'loan_int_rate':                 input_dict['loan_int_rate'],
        'loan_percent_income':           input_dict['loan_amnt'] / max(input_dict['income'], 1),
        'cb_person_cred_hist_length':    input_dict['cred_hist'],
        'credit_score':                  input_dict['credit_score'],
        'previous_loan_defaults_on_file': DEFAULTS_MAP[input_dict['prev_defaults']],
    }

    df_row = pd.DataFrame([row])
    df_row[SKEWED_COLS] = ss.transform(df_row[SKEWED_COLS])
    df_row[NORM_COLS]   = mms.transform(df_row[NORM_COLS])

    X_input = df_row[feature_cols]

    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_input)[0][1]
    else:
        prob = float(model.predict(X_input)[0])

    pred = int(model.predict(X_input)[0])
    return prob, pred


# ─────────────────────────── theme colours ───────────────────────
COLORS = {
    'primary':   '#1B4F72',
    'secondary': '#2980B9',
    'accent':    '#27AE60',
    'danger':    '#E74C3C',
    'warning':   '#F39C12',
    'light_bg':  '#F0F4F8',
    'card_bg':   '#FFFFFF',
    'text':      '#2C3E50',
}
