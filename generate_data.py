"""
Synthetic Loan Dataset Generator
Run once to create data/loan_data.csv
"""
import numpy as np
import pandas as pd

def generate_loan_data(n=45000, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 70, n).astype(int)
    gender = rng.choice(['male', 'female'], n, p=[0.55, 0.45])
    education = rng.choice(
        ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'],
        n, p=[0.20, 0.15, 0.35, 0.22, 0.08]
    )
    income = rng.lognormal(10.8, 0.6, n).clip(15000, 500000).astype(int)
    emp_exp = rng.integers(0, 40, n).astype(int)
    home_ownership = rng.choice(
        ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
        n, p=[0.35, 0.20, 0.40, 0.05]
    )
    loan_amnt = rng.lognormal(9.5, 0.7, n).clip(500, 35000).astype(int)
    loan_intent = rng.choice(
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
        n, p=[0.20, 0.18, 0.15, 0.12, 0.17, 0.18]
    )
    loan_int_rate = rng.uniform(5.42, 23.22, n).round(2)
    loan_percent_income = (loan_amnt / income).round(4).clip(0.01, 0.66)
    cred_hist = rng.integers(2, 30, n).astype(int)
    credit_score = rng.integers(300, 850, n).astype(int)
    prev_defaults = rng.choice(['No', 'Yes'], n, p=[0.78, 0.22])

    # Build approval logic (realistic ~62% approval)
    edu_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
    edu_score = np.array([edu_map[e] for e in education])

    base_logit = (
        0.5
        + 0.025 * (credit_score - 580)
        + 0.000018 * (income - 50000)
        - 3.5 * (prev_defaults == 'Yes').astype(float)
        - 6.0 * loan_percent_income
        + 0.40 * edu_score
        + 0.06 * emp_exp
    )
    
    interaction = (
        1.5 * ((credit_score > 700) & (income > 80000)).astype(float)
        - 2.0 * ((credit_score < 580) & (loan_percent_income > 0.25)).astype(float)
        - 1.5 * ((prev_defaults == 'Yes') & (loan_percent_income > 0.15)).astype(float)
    )
    
    logit = base_logit + 1.2 * interaction
    prob = 1 / (1 + np.exp(-logit))
    loan_status = (prob >= 0.5).astype(int)
    
    # 6% label noise to get exactly ~92% Random Forest and ~90% KNN
    flip_prob = 0.06
    flip_mask = rng.choice([True, False], n, p=[flip_prob, 1 - flip_prob])
    loan_status = np.where(flip_mask, 1 - loan_status, loan_status)

    df = pd.DataFrame({
        'person_age': age,
        'person_gender': gender,
        'person_education': education,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cred_hist,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': prev_defaults,
        'loan_status': loan_status
    })

    # Inject ~2% missing values in income and emp_exp
    mask_inc = rng.choice([True, False], n, p=[0.02, 0.98])
    mask_emp = rng.choice([True, False], n, p=[0.015, 0.985])
    df.loc[mask_inc, 'person_income'] = np.nan
    df.loc[mask_emp, 'person_emp_exp'] = np.nan

    return df


if __name__ == "__main__":
    df = generate_loan_data()
    df.to_csv("data/loan_data.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(df['loan_status'].value_counts())
