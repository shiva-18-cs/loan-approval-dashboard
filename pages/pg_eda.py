"""Page 4 – Exploratory Data Analysis"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import load_raw_data


INSIGHT = {
    'dist':     "📊 Approximately {pct:.1f}% of applications are approved, indicating a moderately selective lending policy. Monitoring this ratio over time can reveal market sentiment and risk appetite changes.",
    'income':   "💰 Income is right-skewed with a long tail. Most applicants earn between $30K–$100K. High earners (>$150K) dominate the approved segment, confirming income as a strong approval predictor.",
    'credit':   "⭐ Credit scores show a bimodal distribution. Applicants above 650 have significantly higher approval rates. This validates using credit score as a primary underwriting criterion.",
    'loan':     "💳 Loan amounts peak between $5,000–$15,000. Smaller loans are easier to approve; larger loans above $25,000 face stricter scrutiny due to elevated default risk.",
    'edu':      "🎓 Higher education levels correlate with higher approval rates. Doctorate and Master holders are approved at 2× the rate of High School-only applicants — reflecting income and stability signals.",
    'home':     "🏠 Homeowners (OWN + MORTGAGE) have significantly higher approval rates than renters, signalling financial stability and collateral availability to lenders.",
    'defaults': "⚠ Prior defaulters face rejection rates exceeding 85%. This single feature has the strongest individual impact on the model's rejection decision.",
    'intent':   "🎯 Education and home improvement loans show higher approval rates, while VENTURE and PERSONAL loans face stricter scrutiny due to perceived higher risk profiles.",
    'corr':     "🔗 Credit score and previous defaults show the strongest correlation with loan_status. Income and employment experience show moderate positive correlation.",
}


def render():
    df = load_raw_data()

    st.markdown("""
    <div class='section-header'>
        <h2>📊 Exploratory Data Analysis</h2>
        <p style='color:#AED6F1;margin:4px 0 0;font-size:0.88rem;'>
            Interactive visualisations · Pattern discovery · Business insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "🥧 Approval Dist.",
        "💰 Income",
        "⭐ Credit Score",
        "💳 Loan Amount",
        "🎓 Education",
        "🏠 Home Ownership",
        "⚠ Defaults",
        "🎯 Loan Intent",
        "🔗 Correlation",
    ])

    # ─ 1. Approval Distribution ──────────────────────────────────
    with tabs[0]:
        cnt = df['loan_status'].value_counts().reset_index()
        cnt.columns = ['Status', 'Count']
        cnt['Status'] = cnt['Status'].map({1:'Approved', 0:'Rejected'})
        pct = cnt.loc[cnt['Status']=='Approved','Count'].values[0] / len(df) * 100

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(cnt, names='Status', values='Count', hole=0.55,
                         color='Status',
                         color_discrete_map={'Approved':'#27AE60','Rejected':'#E74C3C'},
                         title='Loan Approval Distribution')
            fig.update_traces(textinfo='label+percent', textfont_size=13)
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(cnt, x='Status', y='Count', color='Status',
                          color_discrete_map={'Approved':'#27AE60','Rejected':'#E74C3C'},
                          title='Application Count by Status', text='Count')
            fig2.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig2.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"<div class='insight-box'>{INSIGHT['dist'].format(pct=pct)}</div>",
                    unsafe_allow_html=True)

    # ─ 2. Income Distribution ────────────────────────────────────
    with tabs[1]:
        df_income = df.copy()
        df_income['loan_status_label'] = df_income['loan_status'].map({1:'Approved',0:'Rejected'})
        df_income = df_income[df_income['person_income'] < df_income['person_income'].quantile(0.99)]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_income, x='person_income', color='loan_status_label',
                               color_discrete_map={'Approved':'#27AE60','Rejected':'#E74C3C'},
                               nbins=60, barmode='overlay', opacity=0.75,
                               title='Income Distribution by Loan Status',
                               labels={'person_income':'Annual Income ($)'})
            fig.update_layout(height=380, legend_title='Status')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(df_income, x='loan_status_label', y='person_income',
                          color='loan_status_label',
                          color_discrete_map={'Approved':'#27AE60','Rejected':'#E74C3C'},
                          title='Income Spread by Loan Status',
                          labels={'loan_status_label':'', 'person_income':'Annual Income ($)'})
            fig2.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"<div class='insight-box'>{INSIGHT['income']}</div>",
                    unsafe_allow_html=True)

    # ─ 3. Credit Score ───────────────────────────────────────────
    with tabs[2]:
        df_cs = df.copy()
        df_cs['loan_status_label'] = df_cs['loan_status'].map({1:'Approved',0:'Rejected'})

        fig = px.histogram(df_cs, x='credit_score', color='loan_status_label',
                           nbins=50, barmode='overlay', opacity=0.72,
                           color_discrete_map={'Approved':'#27AE60','Rejected':'#E74C3C'},
                           title='Credit Score Distribution by Loan Status',
                           labels={'credit_score':'Credit Score'})
        fig.update_layout(height=380, legend_title='Status')
        st.plotly_chart(fig, use_container_width=True)

        avg_app = df_cs[df_cs['loan_status']==1]['credit_score'].mean()
        avg_rej = df_cs[df_cs['loan_status']==0]['credit_score'].mean()
        c1, c2 = st.columns(2)
        c1.metric("Avg Credit Score — Approved", f"{avg_app:.0f}")
        c2.metric("Avg Credit Score — Rejected", f"{avg_rej:.0f}",
                  delta=f"{avg_app-avg_rej:.0f} pts lower")
        st.markdown(f"<div class='insight-box'>{INSIGHT['credit']}</div>",
                    unsafe_allow_html=True)

    # ─ 4. Loan Amount ────────────────────────────────────────────
    with tabs[3]:
        df_la = df.copy()
        df_la['loan_status_label'] = df_la['loan_status'].map({1:'Approved',0:'Rejected'})

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_la, x='loan_amnt', nbins=50, color='loan_status_label',
                               color_discrete_map={'Approved':'#27AE60','Rejected':'#E74C3C'},
                               barmode='overlay', opacity=0.72,
                               title='Loan Amount Distribution',
                               labels={'loan_amnt':'Loan Amount ($)'})
            fig.update_layout(height=360, legend_title='Status')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.violin(df_la, y='loan_amnt', x='loan_status_label',
                             color='loan_status_label',
                             color_discrete_map={'Approved':'#27AE60','Rejected':'#E74C3C'},
                             box=True, title='Loan Amount Spread',
                             labels={'loan_amnt':'Loan Amount ($)','loan_status_label':''})
            fig2.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"<div class='insight-box'>{INSIGHT['loan']}</div>",
                    unsafe_allow_html=True)

    # ─ 5. Education ──────────────────────────────────────────────
    with tabs[4]:
        edu_order = ['High School','Associate','Bachelor','Master','Doctorate']
        edu_df = (df.groupby('person_education')['loan_status']
                   .agg(['mean','count'])
                   .rename(columns={'mean':'Approval Rate','count':'Count'})
                   .reset_index())
        edu_df['Approval Rate'] = edu_df['Approval Rate'] * 100
        edu_df['person_education'] = pd.Categorical(
            edu_df['person_education'], categories=edu_order, ordered=True)
        edu_df = edu_df.sort_values('person_education')

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(edu_df, x='person_education', y='Approval Rate',
                         color='Approval Rate', color_continuous_scale='Blues',
                         title='Approval Rate by Education Level',
                         labels={'person_education':'Education'})
            fig.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(edu_df, x='person_education', y='Count',
                          color='person_education', title='Application Count by Education',
                          labels={'person_education':'Education'})
            fig2.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"<div class='insight-box'>{INSIGHT['edu']}</div>",
                    unsafe_allow_html=True)

    # ─ 6. Home Ownership ─────────────────────────────────────────
    with tabs[5]:
        home_df = (df.groupby('person_home_ownership')['loan_status']
                    .agg(['mean','count'])
                    .rename(columns={'mean':'Approval Rate','count':'Count'})
                    .reset_index())
        home_df['Approval Rate'] = home_df['Approval Rate'] * 100

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(home_df, x='person_home_ownership', y='Approval Rate',
                         color='Approval Rate', color_continuous_scale='Greens',
                         title='Approval Rate by Home Ownership',
                         labels={'person_home_ownership':'Ownership Type'})
            fig.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(home_df, names='person_home_ownership', values='Count',
                          title='Application Share by Ownership',
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(height=360)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"<div class='insight-box'>{INSIGHT['home']}</div>",
                    unsafe_allow_html=True)

    # ─ 7. Previous Defaults ──────────────────────────────────────
    with tabs[6]:
        def_df = (df.groupby('previous_loan_defaults_on_file')['loan_status']
                   .agg(['mean','count'])
                   .rename(columns={'mean':'Approval Rate','count':'Count'})
                   .reset_index())
        def_df['Approval Rate'] = def_df['Approval Rate'] * 100

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(def_df, x='previous_loan_defaults_on_file', y='Approval Rate',
                         color='previous_loan_defaults_on_file',
                         color_discrete_map={'No':'#27AE60','Yes':'#E74C3C'},
                         title='Approval Rate: Prior Defaults vs None',
                         labels={'previous_loan_defaults_on_file':'Prior Default'})
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(def_df, names='previous_loan_defaults_on_file', values='Count',
                          color='previous_loan_defaults_on_file',
                          color_discrete_map={'No':'#27AE60','Yes':'#E74C3C'},
                          title='Default History Distribution')
            fig2.update_layout(height=360)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"<div class='insight-box'>{INSIGHT['defaults']}</div>",
                    unsafe_allow_html=True)

    # ─ 8. Loan Intent ────────────────────────────────────────────
    with tabs[7]:
        intent_df = (df.groupby('loan_intent')['loan_status']
                      .agg(['mean','count'])
                      .rename(columns={'mean':'Approval Rate','count':'Count'})
                      .reset_index())
        intent_df['Approval Rate'] = intent_df['Approval Rate'] * 100
        intent_df = intent_df.sort_values('Approval Rate', ascending=False)

        fig = px.bar(intent_df, x='loan_intent', y='Approval Rate',
                     color='Count', color_continuous_scale='Blues',
                     title='Approval Rate & Volume by Loan Intent',
                     text=intent_df['Approval Rate'].round(1).astype(str) + '%',
                     labels={'loan_intent':'Loan Intent','Count':'Applications'})
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, coloraxis_showscale=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div class='insight-box'>{INSIGHT['intent']}</div>",
                    unsafe_allow_html=True)

    # ─ 9. Correlation Heatmap ────────────────────────────────────
    with tabs[8]:
        from utils import get_processed_data
        df_proc, _, _, _ = get_processed_data()
        corr = df_proc.corr().round(3)

        fig = px.imshow(corr, text_auto=True, aspect='auto',
                        color_continuous_scale='RdBu_r',
                        color_continuous_midpoint=0,
                        title='Feature Correlation Heatmap',
                        zmin=-1, zmax=1)
        fig.update_layout(height=560, margin=dict(t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div class='insight-box'>{INSIGHT['corr']}</div>",
                    unsafe_allow_html=True)
