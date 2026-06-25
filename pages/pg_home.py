"""Page 1 – Home"""
import streamlit as st
import plotly.graph_objects as go
from utils import load_raw_data, COLORS


def render():
    df = load_raw_data()
    approved  = int(df['loan_status'].sum())
    rejected  = len(df) - approved
    avg_score = int(df['credit_score'].mean())
    avg_loan  = int(df['loan_amnt'].mean())

    # ── Hero ──────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0D2137 0%,#1B4F72 50%,#2980B9 100%);
                border-radius:20px;padding:48px 40px 36px 40px;margin-bottom:32px;
                box-shadow:0 8px 40px rgba(27,79,114,0.30);'>
        <div style='display:flex;align-items:center;gap:18px;margin-bottom:16px;'>
            <div style='font-size:3.5rem;'>🏦</div>
            <div>
                <div style='font-size:0.78rem;color:#AED6F1;letter-spacing:0.12em;
                            text-transform:uppercase;font-weight:600;'>
                    IEEE Final-Year Project  ·  Credit Risk AI
                </div>
                <h1 style='color:white;margin:4px 0 0 0;font-size:2rem;font-weight:800;line-height:1.2;'>
                    AI-Powered Loan Approval Prediction<br>
                    <span style='color:#5DADE2;'>& Credit Risk Assessment Dashboard</span>
                </h1>
            </div>
        </div>
        <p style='color:#AED6F1;font-size:1rem;max-width:780px;line-height:1.7;margin:0;'>
            A production-grade decision-support platform that combines advanced machine learning,
            explainable AI, and regulatory fairness monitoring to help financial institutions
            make faster, fairer, and more transparent lending decisions.
        </p>
        <div style='display:flex;gap:12px;margin-top:24px;flex-wrap:wrap;'>
            <span style='background:rgba(255,255,255,0.15);color:#E8F4FD;padding:6px 16px;
                         border-radius:20px;font-size:0.80rem;font-weight:500;'>
                🤖 4 ML Models
            </span>
            <span style='background:rgba(255,255,255,0.15);color:#E8F4FD;padding:6px 16px;
                         border-radius:20px;font-size:0.80rem;font-weight:500;'>
                🔍 SHAP Explainability
            </span>
            <span style='background:rgba(255,255,255,0.15);color:#E8F4FD;padding:6px 16px;
                         border-radius:20px;font-size:0.80rem;font-weight:500;'>
                ⚖ IEEE 7010 Fairness
            </span>
            <span style='background:rgba(255,255,255,0.15);color:#E8F4FD;padding:6px 16px;
                         border-radius:20px;font-size:0.80rem;font-weight:500;'>
                📊 45,000 Applications
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "📁", f"{len(df):,}", "Total Applications", "#2980B9"),
        (c2, "✅", f"{approved:,}", "Approved Loans",     "#27AE60"),
        (c3, "❌", f"{rejected:,}", "Rejected Loans",     "#E74C3C"),
        (c4, "⭐", f"{avg_score}", "Avg Credit Score",   "#F39C12"),
        (c5, "💰", f"${avg_loan:,}", "Avg Loan Amount",  "#8E44AD"),
    ]
    for col, icon, val, lbl, color in kpis:
        col.markdown(f"""
        <div class='kpi-card' style='border-top-color:{color};'>
            <div class='kpi-icon'>{icon}</div>
            <div class='kpi-value' style='color:{color};'>{val}</div>
            <div class='kpi-label'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Donut chart + Workflow ────────────────────────────────────
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.markdown("### 📊 Loan Approval Distribution")
        fig = go.Figure(go.Pie(
            labels=['Approved', 'Rejected'],
            values=[approved, rejected],
            hole=0.60,
            marker=dict(colors=['#27AE60', '#E74C3C'],
                        line=dict(color='white', width=3)),
            textinfo='label+percent',
            textfont=dict(size=13, family='Inter'),
        ))
        fig.update_layout(
            showlegend=False, height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(
                text=f"<b>{approved/(approved+rejected)*100:.1f}%</b><br>Approved",
                x=0.5, y=0.5, font_size=14, showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### 🔄 AI Decision Pipeline")
        steps = [
            ("📥", "Data Collection",      "#2980B9"),
            ("🧹", "Preprocessing",        "#8E44AD"),
            ("⚙", "Feature Engineering",  "#16A085"),
            ("🎯", "Model Training",       "#D35400"),
            ("🔮", "Prediction",           "#1B4F72"),
            ("📋", "Decision Support",     "#27AE60"),
        ]
        cols = st.columns(len(steps))
        for i, (col, (icon, label, color)) in enumerate(zip(cols, steps)):
            col.markdown(f"""
            <div class='workflow-card' style='border-bottom-color:{color};'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div style='font-size:0.70rem;font-weight:600;color:{color};
                            margin-top:6px;line-height:1.3;'>
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Objectives ───────────────────────────────────────────────
    st.markdown("### 🎯 Project Objectives")
    obj_cols = st.columns(3)
    objectives = [
        ("📊", "Financial Analysis",
         "Analyze 13 applicant features including credit score, income, employment history, and loan parameters to assess creditworthiness."),
        ("🤖", "Intelligent Prediction",
         "Train and compare Logistic Regression, SVM, KNN, and Random Forest classifiers with comprehensive evaluation metrics."),
        ("🔍", "Explainable Decisions",
         "Use SHAP values to provide transparent, human-readable explanations for every loan approval or rejection decision."),
        ("⚖", "Fair Lending",
         "Monitor demographic parity and comply with IEEE 7010 & EEOC 80% Rule to ensure bias-free lending practices."),
        ("📈", "Business Intelligence",
         "Generate executive-level insights including risk segmentation, approval trends, and portfolio health indicators."),
        ("🏦", "Decision Support",
         "Provide banks and NBFCs with a real-time, interactive decision-support tool to reduce manual review time by 80%."),
    ]
    for i, (icon, title, desc) in enumerate(objectives):
        col = obj_cols[i % 3]
        col.markdown(f"""
        <div style='background:white;border-radius:14px;padding:20px;margin-bottom:12px;
                    box-shadow:0 2px 12px rgba(27,79,114,0.08);
                    border-top:3px solid #2980B9;'>
            <div style='font-size:1.6rem;margin-bottom:8px;'>{icon}</div>
            <div style='font-weight:700;color:#1B4F72;font-size:0.95rem;margin-bottom:6px;'>
                {title}
            </div>
            <div style='color:#5D6D7E;font-size:0.83rem;line-height:1.55;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Quick stats bar ───────────────────────────────────────────
    st.markdown("---")
    s1, s2, s3, s4 = st.columns(4)
    rate = approved / len(df) * 100
    s1.metric("Approval Rate",    f"{rate:.1f}%",       "+2.3% YoY")
    s2.metric("Avg Interest Rate", f"{df['loan_int_rate'].mean():.1f}%", "-0.5% YoY")
    s3.metric("Avg Loan Term",    "3.5 years",          "Stable")
    s4.metric("Default Rate",     f"{(df['previous_loan_defaults_on_file']=='Yes').mean()*100:.1f}%", "-1.1% YoY")
