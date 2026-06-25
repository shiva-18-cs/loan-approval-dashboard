"""Page 3 – Data Preprocessing"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils import (load_raw_data, SKEWED_COLS, NORM_COLS,
                   GENDER_MAP, HOME_MAP, INTENT_MAP, DEFAULTS_MAP, EDUCATION_MAP)


def render():
    df_raw = load_raw_data().copy()

    st.markdown("""
    <div class='section-header'>
        <h2>🧹 Data Preprocessing Dashboard</h2>
        <p style='color:#AED6F1;margin:4px 0 0;font-size:0.88rem;'>
            Step-by-step visual walkthrough of every transformation applied to the raw data
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Step progress ─────────────────────────────────────────────
    steps = ["Feature ID", "Scaling", "Encoding", "Outlier Treatment", "Feature Selection"]
    s_cols = st.columns(len(steps))
    colors = ["#2980B9","#8E44AD","#16A085","#D35400","#27AE60"]
    for col, step, color in zip(s_cols, steps, colors):
        col.markdown(f"""
        <div style='background:{color};border-radius:10px;padding:12px 8px;
                    text-align:center;color:white;font-size:0.78rem;font-weight:600;'>
            {step}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Step 1: Feature Identification ───────────────────────────
    with st.expander("📌 Step 1 — Feature Identification", expanded=True):
        cat_cols = [c for c in df_raw.columns if df_raw[c].dtype == 'object']
        num_cols = [c for c in df_raw.columns if df_raw[c].dtype != 'object']

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style='background:#EBF5FB;border-radius:12px;padding:18px;'>
                <div style='font-weight:700;color:#1B4F72;font-size:0.95rem;margin-bottom:12px;'>
                    🔢 Numerical Features ({len(num_cols)})
                </div>
            """, unsafe_allow_html=True)
            for col_name in num_cols:
                st.markdown(f"• `{col_name}`")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div style='background:#F9EBEA;border-radius:12px;padding:18px;'>
                <div style='font-weight:700;color:#922B21;font-size:0.95rem;margin-bottom:12px;'>
                    🔤 Categorical Features ({len(cat_cols)})
                </div>
            """, unsafe_allow_html=True)
            for col_name in cat_cols:
                st.markdown(f"• `{col_name}`")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 2: Scaling ───────────────────────────────────────────
    with st.expander("📏 Step 2 — Feature Scaling", expanded=False):
        st.markdown("""
        <div style='display:flex;gap:16px;flex-wrap:wrap;'>
            <div style='flex:1;min-width:220px;background:#EBF5FB;border-radius:12px;
                        padding:16px;border-left:4px solid #2980B9;'>
                <b style='color:#1B4F72;'>StandardScaler</b>
                <p style='font-size:0.82rem;color:#5D6D7E;margin:6px 0 0;'>
                    Applied to high-variance, skewed features.<br>
                    z = (x − μ) / σ → mean=0, std=1
                </p>
            </div>
            <div style='flex:1;min-width:220px;background:#EBF9F1;border-radius:12px;
                        padding:16px;border-left:4px solid #27AE60;'>
                <b style='color:#1B4F72;'>MinMaxScaler</b>
                <p style='font-size:0.82rem;color:#5D6D7E;margin:6px 0 0;'>
                    Applied to bounded features (interest rate).<br>
                    x' = (x − min) / (max − min) → [0, 1]
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>**StandardScaler columns:**")
        st.code(", ".join(SKEWED_COLS), language="python")
        st.markdown("**MinMaxScaler columns:**")
        st.code(", ".join(NORM_COLS), language="python")

        # Visual: before vs after on income
        col_viz = 'person_income'
        df_raw[col_viz] = df_raw[col_viz].fillna(df_raw[col_viz].median())
        before = df_raw[col_viz].clip(upper=df_raw[col_viz].quantile(0.99))
        from sklearn.preprocessing import StandardScaler
        after = StandardScaler().fit_transform(before.values.reshape(-1,1)).flatten()

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=before, name='Before Scaling', marker_color='#E74C3C',
                                   opacity=0.7, nbinsx=60))
        fig.add_trace(go.Histogram(x=after,  name='After StandardScaler', marker_color='#2980B9',
                                   opacity=0.7, nbinsx=60))
        fig.update_layout(
            barmode='overlay', height=320, title='Income: Before vs After StandardScaler',
            xaxis_title='Value', yaxis_title='Count',
            margin=dict(t=40, b=20), legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Step 3: Encoding ──────────────────────────────────────────
    with st.expander("🔡 Step 3 — Label Encoding", expanded=False):
        encodings = [
            ("Gender",                 GENDER_MAP,    "#2980B9"),
            ("Home Ownership",         HOME_MAP,      "#8E44AD"),
            ("Education",              EDUCATION_MAP, "#16A085"),
            ("Loan Intent",            INTENT_MAP,    "#D35400"),
            ("Previous Loan Defaults", DEFAULTS_MAP,  "#E74C3C"),
        ]
        enc_cols = st.columns(len(encodings))
        for col, (title, mapping, color) in zip(enc_cols, encodings):
            rows_html = "".join(
                f"<tr><td style='padding:3px 8px;color:#5D6D7E;font-size:0.78rem;'>{k}</td>"
                f"<td style='padding:3px 8px;font-weight:700;color:{color};font-size:0.78rem;'>→ {v}</td></tr>"
                for k, v in mapping.items()
            )
            col.markdown(f"""
            <div style='background:white;border-radius:12px;padding:14px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.06);border-top:3px solid {color};'>
                <b style='color:#1B4F72;font-size:0.85rem;'>{title}</b><br>
                <table style='margin-top:8px;width:100%;border-collapse:collapse;'>
                    {rows_html}
                </table>
            </div>""", unsafe_allow_html=True)

    # ── Step 4: Outlier Treatment ─────────────────────────────────
    with st.expander("📦 Step 4 — Outlier Treatment (IQR Method)", expanded=False):
        st.markdown("""
        <div class='insight-box'>
            🔬 <b>Method:</b> IQR-based right-tail trimming. 
            Any value above Q3 + 1.5×IQR is removed from the training set.
            This prevents high-income outliers from distorting model weights.
        </div>""", unsafe_allow_html=True)

        feat_sel = st.selectbox("Choose feature to inspect:", SKEWED_COLS, key="out_feat")

        col_data = df_raw[feat_sel].dropna()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        upper = Q3 + 1.5 * (Q3 - Q1)
        before_data = col_data
        after_data  = col_data[col_data <= upper]

        c1, c2 = st.columns(2)
        with c1:
            fig_b = px.box(before_data, title=f"{feat_sel} — Before",
                           color_discrete_sequence=['#E74C3C'])
            fig_b.update_layout(height=260, showlegend=False,
                                 margin=dict(t=40, b=10))
            st.plotly_chart(fig_b, use_container_width=True)
            st.metric("Outliers detected",
                      f"{(col_data > upper).sum():,} rows removed")

        with c2:
            fig_a = px.box(after_data, title=f"{feat_sel} — After",
                           color_discrete_sequence=['#27AE60'])
            fig_a.update_layout(height=260, showlegend=False,
                                  margin=dict(t=40, b=10))
            st.plotly_chart(fig_a, use_container_width=True)
            st.metric("Remaining rows", f"{len(after_data):,}")

    # ── Step 5: Feature Selection ─────────────────────────────────
    with st.expander("🎯 Step 5 — Feature Selection (Correlation Threshold)", expanded=False):
        st.markdown("""
        <div class='insight-box'>
            📐 <b>Strategy:</b> After preprocessing, features with |correlation| > 0.05 with
            <code>loan_status</code> are selected. This removes low-signal noise
            and improves model generalisation.
        </div>""", unsafe_allow_html=True)

        from utils import get_processed_data
        df_proc, _, _, high_corr = get_processed_data()
        corr_val = df_proc.corr()['loan_status'].drop('loan_status')
        corr_df = corr_val.abs().sort_values(ascending=False).reset_index()
        corr_df.columns = ['Feature', '|Correlation with loan_status|']
        corr_df['Selected'] = corr_df['|Correlation with loan_status|'] > 0.05
        corr_df['Selected'] = corr_df['Selected'].map({True: '✅ Yes', False: '❌ No'})

        fig = px.bar(corr_df, x='Feature', y='|Correlation with loan_status|',
                     color='Selected',
                     color_discrete_map={'✅ Yes':'#27AE60','❌ No':'#E74C3C'},
                     title='Feature Correlation with Target (|r|)')
        fig.add_hline(y=0.05, line_dash='dash', line_color='#F39C12',
                      annotation_text='Threshold 0.05')
        fig.update_layout(height=380, margin=dict(t=40, b=10),
                           xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
