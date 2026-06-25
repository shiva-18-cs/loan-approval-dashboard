"""Page 2 – Dataset Explorer"""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_raw_data, FEATURE_DESCRIPTIONS


def render():
    df = load_raw_data()

    st.markdown("""
    <div class='section-header'>
        <h2>📋 Dataset Explorer</h2>
        <p style='color:#AED6F1;margin:4px 0 0;font-size:0.88rem;'>
            Raw data inspection · Feature audit · Statistical profiling
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Shape & overview ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
    <div class='kpi-card' style='border-top-color:#2980B9;'>
        <div class='kpi-icon'>📊</div>
        <div class='kpi-value' style='color:#2980B9;'>{len(df):,}</div>
        <div class='kpi-label'>Total Rows</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class='kpi-card' style='border-top-color:#8E44AD;'>
        <div class='kpi-icon'>🗂</div>
        <div class='kpi-value' style='color:#8E44AD;'>{df.shape[1]}</div>
        <div class='kpi-label'>Features</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div class='kpi-card' style='border-top-color:#27AE60;'>
        <div class='kpi-icon'>🔢</div>
        <div class='kpi-value' style='color:#27AE60;'>
            {len([c for c in df.columns if df[c].dtype != 'object'])}
        </div>
        <div class='kpi-label'>Numerical Features</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""
    <div class='kpi-card' style='border-top-color:#F39C12;'>
        <div class='kpi-icon'>🔤</div>
        <div class='kpi-value' style='color:#F39C12;'>
            {len([c for c in df.columns if df[c].dtype == 'object' and c != 'loan_status'])}
        </div>
        <div class='kpi-label'>Categorical Features</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔎 Data Preview",
        "📐 Data Types",
        "❓ Missing Values",
        "📈 Statistics",
        "📚 Feature Guide",
    ])

    with tab1:
        n = st.slider("Rows to display", 5, 100, 20, key="preview_rows")
        st.dataframe(df.head(n), use_container_width=True, height=420)
        st.markdown(f"""
        <div class='insight-box'>
            💡 <b>Dataset shape:</b> {df.shape[0]:,} rows × {df.shape[1]} columns.
            The target variable <code>loan_status</code> is binary: 1 = Approved, 0 = Rejected.
        </div>""", unsafe_allow_html=True)

    with tab2:
        dtype_df = pd.DataFrame({
            'Feature': df.columns,
            'Data Type': df.dtypes.astype(str).values,
            'Sample Value': [str(df[c].iloc[0]) for c in df.columns],
            'Unique Values': [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True, height=480)

    with tab3:
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Feature', 'Missing Count']
        missing['Missing %'] = (missing['Missing Count'] / len(df) * 100).round(2)
        missing['Status'] = missing['Missing Count'].apply(
            lambda x: '✅ Complete' if x == 0 else f'⚠ {x} missing'
        )
        st.dataframe(missing, use_container_width=True, hide_index=True, height=480)

        if missing['Missing Count'].sum() > 0:
            fig = px.bar(
                missing[missing['Missing Count'] > 0],
                x='Feature', y='Missing %',
                color='Missing %',
                color_continuous_scale='Reds',
                title='Missing Value Percentage by Feature',
            )
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div class='insight-box'>
                ⚠ <b>Strategy:</b> Numerical missing values are imputed with the <b>column median</b>
                to prevent skewing from outliers — a best practice in financial data preprocessing.
            </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Dataset is complete — no missing values detected.")

    with tab4:
        num_cols = [c for c in df.columns if df[c].dtype != 'object']
        st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True, height=420)
        st.markdown("""
        <div class='insight-box'>
            💡 <b>Observation:</b> Several features (income, loan amount) exhibit high skewness
            requiring StandardScaler normalization before model training. Interest rate is
            bounded and will use MinMaxScaler.
        </div>""", unsafe_allow_html=True)

    with tab5:
        rows = []
        for col, desc in FEATURE_DESCRIPTIONS.items():
            dtype = str(df[col].dtype) if col in df.columns else 'N/A'
            rows.append({'Feature': col, 'Description': desc, 'Type': dtype})
        guide_df = pd.DataFrame(rows)
        st.dataframe(guide_df, use_container_width=True, hide_index=True, height=500)
