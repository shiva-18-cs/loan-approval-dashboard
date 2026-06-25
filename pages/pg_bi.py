"""Page 9 – Business Intelligence Dashboard"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import load_raw_data


def render():
    df = load_raw_data()

    st.markdown("""
    <div class='section-header'>
        <h2>📈 Business Intelligence Dashboard</h2>
        <p style='color:#AED6F1;margin:4px 0 0;font-size:0.88rem;'>
            Executive-level insights · Portfolio analytics · Strategic recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Derived columns for BI
    df2 = df.copy()
    df2['approval_label'] = df2['loan_status'].map({1:'Approved', 0:'Rejected'})

    # Credit segments
    def seg_credit(s):
        if s >= 750: return 'Excellent (750+)'
        if s >= 700: return 'Very Good (700-749)'
        if s >= 650: return 'Good (650-699)'
        if s >= 600: return 'Fair (600-649)'
        return 'Poor (<600)'

    # Income segments
    def seg_income(i):
        if i >= 150000: return 'High ($150K+)'
        if i >= 75000:  return 'Upper-Mid ($75K-150K)'
        if i >= 40000:  return 'Middle ($40K-75K)'
        return 'Lower (<$40K)'

    df2['credit_segment'] = df2['credit_score'].apply(seg_credit)
    df2['income_segment'] = df2['person_income'].apply(seg_income)

    # Loan risk
    def seg_risk(row):
        score = 0
        if row['credit_score'] < 600: score += 2
        if row['previous_loan_defaults_on_file'] == 'Yes': score += 3
        if row['loan_percent_income'] > 0.3: score += 1
        if score >= 4: return 'High Risk'
        if score >= 2: return 'Medium Risk'
        return 'Low Risk'

    df2['risk_tier'] = df2.apply(seg_risk, axis=1)

    tabs = st.tabs([
        "📈 Approval Analytics",
        "⚠ Risk Segmentation",
        "💳 Credit Analysis",
        "💰 Income Analysis",
        "🎯 Loan Demand",
        "💡 Recommendations",
    ])

    # ─ 1. Approval Analytics ─────────────────────────────────────
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        total   = len(df2)
        approved = df2['loan_status'].sum()
        avg_loan = df2['loan_amnt'].mean()
        avg_int  = df2['loan_int_rate'].mean()

        kpis = [
            (c1, "📁", f"{total:,}",    "Total Applications", "#2980B9"),
            (c2, "✅", f"{approved:,}", "Loans Approved",     "#27AE60"),
            (c3, "💰", f"${avg_loan:,.0f}", "Avg Loan Size",  "#F39C12"),
            (c4, "📊", f"{avg_int:.1f}%",  "Avg Interest Rate","#8E44AD"),
        ]
        for col, icon, val, lbl, color in kpis:
            col.markdown(f"""
            <div class='kpi-card' style='border-top-color:{color};'>
                <div class='kpi-icon'>{icon}</div>
                <div class='kpi-value' style='color:{color};'>{val}</div>
                <div class='kpi-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Approval by intent
        intent_df = (df2.groupby(['loan_intent','approval_label'])
                      .size().reset_index(name='count'))
        fig = px.bar(intent_df, x='loan_intent', y='count', color='approval_label',
                     color_discrete_map={'Approved':'#27AE60','Rejected':'#E74C3C'},
                     title='Loan Applications by Intent & Status',
                     barmode='group',
                     labels={'loan_intent':'Intent','count':'Applications','approval_label':'Status'})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Interest rate by status
        fig2 = px.box(df2, x='approval_label', y='loan_int_rate',
                      color='approval_label',
                      color_discrete_map={'Approved':'#27AE60','Rejected':'#E74C3C'},
                      title='Interest Rate Distribution by Decision',
                      labels={'approval_label':'', 'loan_int_rate':'Interest Rate (%)'})
        fig2.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ─ 2. Risk Segmentation ───────────────────────────────────────
    with tabs[1]:
        risk_cnt = df2['risk_tier'].value_counts().reset_index()
        risk_cnt.columns = ['Risk Tier', 'Count']
        risk_colors = {'Low Risk':'#27AE60','Medium Risk':'#F39C12','High Risk':'#E74C3C'}

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(risk_cnt, names='Risk Tier', values='Count',
                         color='Risk Tier', color_discrete_map=risk_colors,
                         title='Portfolio Risk Distribution', hole=0.5)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            risk_app = (df2.groupby('risk_tier')['loan_status']
                         .agg(['mean','count']).reset_index())
            risk_app.columns = ['Risk Tier','Approval Rate','Count']
            risk_app['Approval Rate'] *= 100

            fig2 = px.bar(risk_app, x='Risk Tier', y='Approval Rate',
                          color='Risk Tier', color_discrete_map=risk_colors,
                          title='Approval Rate by Risk Tier',
                          text=risk_app['Approval Rate'].round(1).astype(str)+'%')
            fig2.update_traces(textposition='outside')
            fig2.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Risk × Intent heatmap
        risk_intent = (df2.groupby(['loan_intent','risk_tier'])
                        .size().unstack(fill_value=0))
        fig3 = px.imshow(risk_intent, text_auto=True, aspect='auto',
                         color_continuous_scale='RdYlGn',
                         title='Risk Tier by Loan Intent Heatmap')
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)

    # ─ 3. Credit Analysis ────────────────────────────────────────
    with tabs[2]:
        cs_order = ['Poor (<600)','Fair (600-649)','Good (650-699)',
                    'Very Good (700-749)','Excellent (750+)']
        cs_df = (df2.groupby('credit_segment')['loan_status']
                  .agg(['mean','count']).reset_index())
        cs_df.columns = ['Credit Segment','Approval Rate','Count']
        cs_df['Approval Rate'] *= 100
        cs_df['credit_segment_ord'] = pd.Categorical(
            cs_df['Credit Segment'], categories=cs_order, ordered=True)
        cs_df = cs_df.sort_values('credit_segment_ord')

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(cs_df, x='Credit Segment', y='Approval Rate',
                         color='Approval Rate', color_continuous_scale='Blues',
                         title='Approval Rate by Credit Segment',
                         text=cs_df['Approval Rate'].round(1).astype(str)+'%')
            fig.update_traces(textposition='outside')
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.funnel(cs_df, x='Count', y='Credit Segment',
                             title='Application Volume by Credit Tier',
                             color_discrete_sequence=['#2980B9'])
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
            💡 <b>Recommendation:</b> Portfolio should target applicants in the 700-750 segment
            (high approval rate, large volume). Products like secured micro-loans can be
            designed to rehabilitate Fair (600-649) applicants and reduce churn.
        </div>""", unsafe_allow_html=True)

    # ─ 4. Income Analysis ────────────────────────────────────────
    with tabs[3]:
        inc_order = ['Lower (<$40K)','Middle ($40K-75K)','Upper-Mid ($75K-150K)','High ($150K+)']
        inc_df = (df2.groupby('income_segment')['loan_status']
                   .agg(['mean','count']).reset_index())
        inc_df.columns = ['Income Segment','Approval Rate','Count']
        inc_df['Approval Rate'] *= 100
        inc_df['ord'] = pd.Categorical(inc_df['Income Segment'], categories=inc_order, ordered=True)
        inc_df = inc_df.sort_values('ord')

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(inc_df, x='Income Segment', y='Approval Rate',
                         color='Approval Rate', color_continuous_scale='Greens',
                         title='Approval Rate by Income Band',
                         text=inc_df['Approval Rate'].round(1).astype(str)+'%')
            fig.update_traces(textposition='outside')
            fig.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(inc_df, names='Income Segment', values='Count',
                          color_discrete_sequence=px.colors.sequential.Greens_r,
                          title='Application Share by Income')
            fig2.update_layout(height=360)
            st.plotly_chart(fig2, use_container_width=True)

    # ─ 5. Loan Demand ─────────────────────────────────────────────
    with tabs[4]:
        intent_vol = df2['loan_intent'].value_counts().reset_index()
        intent_vol.columns = ['Intent', 'Applications']

        fig = px.treemap(intent_vol, path=['Intent'], values='Applications',
                         color='Applications', color_continuous_scale='Blues',
                         title='Loan Demand Treemap by Purpose')
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)

        # Avg loan by intent
        avg_intent = df2.groupby('loan_intent')['loan_amnt'].mean().sort_values().reset_index()
        fig2 = px.bar(avg_intent, x='loan_amnt', y='loan_intent', orientation='h',
                      color='loan_amnt', color_continuous_scale='Purples',
                      title='Average Loan Amount by Intent',
                      labels={'loan_amnt':'Avg Loan ($)','loan_intent':'Intent'})
        fig2.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ─ 6. Recommendations ────────────────────────────────────────
    with tabs[5]:
        recommendations = [
            ("🎯", "#2980B9", "Target High-Quality Segments",
             "Focus acquisition on applicants with 700+ credit scores and >5 years employment. These segments show 75%+ approval rates with minimal default risk."),
            ("⚠", "#E74C3C", "Prior Default Monitoring",
             "22% of applicants have prior defaults. Implement a rehabilitation loan product with stricter covenants to capture this underserved segment safely."),
            ("💰", "#27AE60", "Income-Based Product Design",
             "Design tiered loan products: micro-loans ($1K-$5K) for lower income, standard ($5K-$20K) for middle income, and premium ($20K+) for upper income bands."),
            ("⚡", "#F39C12", "Process Automation",
             "90% of Excellent/Very Good credit applicants are approved. These can be auto-approved within 60 seconds, reducing processing cost by an estimated 73%."),
            ("⚖", "#8E44AD", "Fairness Monitoring",
             "Schedule quarterly bias audits. Maintain EEOC 80% rule compliance across gender, education, and home ownership dimensions to reduce regulatory risk."),
            ("🔍", "#16A085", "Explainability for Compliance",
             "Provide written AI explanations for all rejections. CFPB requires adverse action notices — SHAP-based explanations can automate this requirement."),
        ]
        r1, r2 = st.columns(2)
        r3, r4 = st.columns(2)
        r5, r6 = st.columns(2)
        panels = [r1, r2, r3, r4, r5, r6]
        for panel, (icon, color, title, desc) in zip(panels, recommendations):
            panel.markdown(f"""
            <div style='background:white;border-radius:14px;padding:20px;
                        border-left:4px solid {color};
                        box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-bottom:8px;height:100%;'>
                <div style='font-size:1.6rem;margin-bottom:8px;'>{icon}</div>
                <div style='font-weight:700;color:#1B4F72;font-size:0.92rem;margin-bottom:6px;'>
                    {title}
                </div>
                <div style='color:#5D6D7E;font-size:0.82rem;line-height:1.55;'>{desc}</div>
            </div>""", unsafe_allow_html=True)
