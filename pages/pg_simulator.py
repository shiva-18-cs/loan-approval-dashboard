"""Page 6 – Loan Approval Simulator"""
import streamlit as st
import pandas as pd
import numpy as np
from utils import (train_all_models, get_processed_data, predict_single,
                   GENDER_MAP, HOME_MAP, INTENT_MAP, DEFAULTS_MAP, EDUCATION_MAP)


def render():
    st.markdown("""
    <div class='section-header'>
        <h2>🎯 Loan Approval Simulator</h2>
        <p style='color:#AED6F1;margin:4px 0 0;font-size:0.88rem;'>
            Enter applicant details and get an instant AI-powered lending decision
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load models silently
    results, trained, X_train, X_test, y_train, y_test, features = train_all_models()
    df_proc, ss, mms, high_corr = get_processed_data()

    # Best model = highest F1
    best_name = max(results, key=lambda k: results[k]['F1 Score'])
    best_model = trained[best_name]

    # ── Form ──────────────────────────────────────────────────────
    st.markdown("#### 📝 Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Details**")
        age        = st.slider("Age",         18, 75, 32, key="sim_age")
        gender     = st.selectbox("Gender",   list(GENDER_MAP.keys()), key="sim_gender")
        education  = st.selectbox("Education",list(EDUCATION_MAP.keys()),
                                  index=2, key="sim_edu")

    with col2:
        st.markdown("**💼 Financial Profile**")
        income     = st.number_input("Annual Income ($)", 10000, 500000, 65000,
                                     step=1000, key="sim_income")
        emp_exp    = st.slider("Employment Experience (years)", 0, 40, 5, key="sim_exp")
        home_own   = st.selectbox("Home Ownership", list(HOME_MAP.keys()), key="sim_home")

    with col3:
        st.markdown("**💳 Loan Details**")
        loan_amnt  = st.number_input("Loan Amount ($)", 500, 35000, 10000,
                                     step=500, key="sim_loan")
        intent     = st.selectbox("Loan Intent",  list(INTENT_MAP.keys()), key="sim_intent")
        int_rate   = st.slider("Interest Rate (%)", 5.4, 23.2, 12.5,
                               step=0.1, key="sim_rate")

    col4, col5 = st.columns(2)
    with col4:
        credit_score = st.slider("Credit Score", 300, 850, 650, key="sim_cs")
        cred_hist    = st.slider("Credit History Length (years)", 2, 30, 8,
                                 key="sim_hist")
    with col5:
        prev_default = st.selectbox("Previous Loan Defaults",
                                    list(DEFAULTS_MAP.keys()), key="sim_def")
        model_choice = st.selectbox("Model to use for prediction",
                                    list(trained.keys()),
                                    index=list(trained.keys()).index(best_name),
                                    key="sim_model")

    st.markdown("---")

    # ── Predict ───────────────────────────────────────────────────
    if st.button("🔮  Run Loan Assessment", type="primary", use_container_width=True):
        input_dict = {
            'age': age, 'gender': gender, 'education': education,
            'income': income, 'emp_exp': emp_exp, 'home_ownership': home_own,
            'loan_amnt': loan_amnt, 'loan_intent': intent, 'loan_int_rate': int_rate,
            'cred_hist': cred_hist, 'credit_score': credit_score,
            'prev_defaults': prev_default,
        }

        model = trained[model_choice]
        prob, pred = predict_single(input_dict, ss, mms, model, features)

        # Risk category
        if prob >= 0.70:
            risk_label, risk_color, risk_icon = "Low Risk",    "#27AE60", "🟢"
        elif prob >= 0.45:
            risk_label, risk_color, risk_icon = "Medium Risk", "#F39C12", "🟡"
        else:
            risk_label, risk_color, risk_icon = "High Risk",   "#E74C3C", "🔴"

        if pred == 1:
            decision_html = f"""
            <div style='background:linear-gradient(135deg,#D5F5E3,#A9DFBF);
                        border-radius:20px;padding:32px;text-align:center;
                        border:3px solid #27AE60;margin-bottom:20px;
                        box-shadow:0 6px 30px rgba(39,174,96,0.25);'>
                <div style='font-size:4rem;'>✅</div>
                <div style='font-size:2rem;font-weight:800;color:#1E8449;margin:8px 0;'>
                    LOAN APPROVED
                </div>
                <div style='color:#27AE60;font-size:0.95rem;'>
                    The applicant meets the credit criteria for loan approval.
                </div>
            </div>"""
        else:
            decision_html = f"""
            <div style='background:linear-gradient(135deg,#FADBD8,#F5B7B1);
                        border-radius:20px;padding:32px;text-align:center;
                        border:3px solid #E74C3C;margin-bottom:20px;
                        box-shadow:0 6px 30px rgba(231,76,60,0.25);'>
                <div style='font-size:4rem;'>❌</div>
                <div style='font-size:2rem;font-weight:800;color:#C0392B;margin:8px 0;'>
                    LOAN REJECTED
                </div>
                <div style='color:#E74C3C;font-size:0.95rem;'>
                    Insufficient creditworthiness based on current profile.
                </div>
            </div>"""

        st.markdown(decision_html, unsafe_allow_html=True)

        # KPI cards
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class='kpi-card' style='border-top-color:#2980B9;'>
            <div class='kpi-icon'>📊</div>
            <div class='kpi-value' style='color:#2980B9;'>{prob*100:.1f}%</div>
            <div class='kpi-label'>Approval Probability</div>
        </div>""", unsafe_allow_html=True)

        c2.markdown(f"""
        <div class='kpi-card' style='border-top-color:{"#27AE60" if pred==1 else "#E74C3C"};'>
            <div class='kpi-icon'>{'✅' if pred==1 else '❌'}</div>
            <div class='kpi-value' style='color:{"#27AE60" if pred==1 else "#E74C3C"};'>
                {'Approved' if pred==1 else 'Rejected'}
            </div>
            <div class='kpi-label'>Decision</div>
        </div>""", unsafe_allow_html=True)

        c3.markdown(f"""
        <div class='kpi-card' style='border-top-color:{risk_color};'>
            <div class='kpi-icon'>{risk_icon}</div>
            <div class='kpi-value' style='color:{risk_color};'>{risk_label}</div>
            <div class='kpi-label'>Risk Category</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability gauge
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            delta={'reference': 50, 'suffix': '%'},
            number={'suffix': '%', 'font': {'size': 40}},
            title={'text': 'Approval Probability Score'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0,  45], 'color': '#FADBD8'},
                    {'range': [45, 70], 'color': '#FDEBD0'},
                    {'range': [70, 100],'color': '#D5F5E3'},
                ],
                'threshold': {
                    'line': {'color': '#1B4F72', 'width': 4},
                    'thickness': 0.8,
                    'value': 50,
                }
            }
        ))
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Profile summary
        pct_inc = loan_amnt / income * 100
        st.markdown("#### 📋 Applicant Profile Summary")
        summary_data = {
            'Attribute': ['Credit Score','Annual Income','Loan Amount','Loan-to-Income Ratio',
                          'Employment Experience','Prior Defaults','Education','Home Ownership'],
            'Value': [f"{credit_score}",f"${income:,}",f"${loan_amnt:,}",
                      f"{pct_inc:.1f}%",f"{emp_exp} years",prev_default,
                      education, home_own],
            'Assessment': [
                '⭐ Excellent' if credit_score>=750 else ('✅ Good' if credit_score>=650 else '⚠ Fair'),
                '✅ Strong' if income>=75000 else ('⚠ Moderate' if income>=40000 else '❌ Low'),
                '✅ Moderate' if loan_amnt<=15000 else ('⚠ High' if loan_amnt<=25000 else '❌ Very High'),
                '✅ Safe' if pct_inc<=15 else ('⚠ Elevated' if pct_inc<=30 else '❌ Risky'),
                '✅ Stable' if emp_exp>=5 else '⚠ New',
                '✅ None' if prev_default=='No' else '❌ Has Defaults',
                '✅ Advanced' if education in ['Master','Doctorate'] else '⚠ Standard',
                '✅ Owner' if home_own in ['OWN','MORTGAGE'] else '⚠ Renter',
            ]
        }
        st.dataframe(pd.DataFrame(summary_data),
                     use_container_width=True, hide_index=True)
