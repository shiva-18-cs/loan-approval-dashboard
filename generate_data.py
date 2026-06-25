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

    logit = (
        0.5                                                     # base intercept → ~62% approval
        + 0.015 * (credit_score - 550)                         # credit score (strongest signal)
        + 0.000008 * (income - 45000)                          # income contribution
        - 2.5 * (prev_defaults == 'Yes').astype(float)         # heavy penalty for defaults
        - 3.0 * loan_percent_income                            # loan-to-income ratio
        + 0.20 * edu_score                                     # education bonus
        + 0.04 * emp_exp                                       # experience bonus
        + 0.30 * (home_ownership == 'OWN').astype(float)       # ownership bonus
        + 0.15 * (home_ownership == 'MORTGAGE').astype(float)
        - 0.08 * (loan_int_rate - 10)                          # high rate = higher risk
        + rng.normal(0, 0.4, n)                                # noise
    )
    prob = 1 / (1 + np.exp(-logit))
    loan_status = (rng.uniform(0, 1, n) < prob).astype(int)

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
