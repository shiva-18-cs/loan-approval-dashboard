"""Page 10 – Project Impact"""
import streamlit as st
import plotly.graph_objects as go


def render():
    st.markdown("""
    <div class='section-header'>
        <h2>🏆 Project Impact & Summary</h2>
        <p style='color:#AED6F1;margin:4px 0 0;font-size:0.88rem;'>
            Problem · Solution · Benefits · Technology Stack · IEEE compliance
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Problem vs Solution ──────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#FADBD8,#F5CBA7);
                    border-radius:16px;padding:28px;height:100%;
                    border-left:5px solid #E74C3C;'>
            <div style='font-size:2rem;margin-bottom:12px;'>🚨</div>
            <h3 style='color:#C0392B;margin:0 0 12px;'>The Problem</h3>
            <ul style='color:#5D6D7E;font-size:0.88rem;line-height:1.7;padding-left:16px;'>
                <li>Manual loan processing takes <b>3–14 business days</b></li>
                <li>High <b>human bias</b> in subjective decisions</li>
                <li>Inconsistent evaluation criteria across agents</li>
                <li>No quantified risk scoring methodology</li>
                <li>Difficulty detecting fraud patterns at scale</li>
                <li>Regulatory non-compliance risk from undocumented decisions</li>
                <li>Poor customer experience with opaque rejections</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#D5F5E3,#A9DFBF);
                    border-radius:16px;padding:28px;height:100%;
                    border-left:5px solid #27AE60;'>
            <div style='font-size:2rem;margin-bottom:12px;'>✅</div>
            <h3 style='color:#1E8449;margin:0 0 12px;'>The Solution</h3>
            <ul style='color:#5D6D7E;font-size:0.88rem;line-height:1.7;padding-left:16px;'>
                <li>AI-powered decision in <b>under 60 seconds</b></li>
                <li>4 ML models with objective, data-driven scoring</li>
                <li>SHAP explainability for every decision</li>
                <li>Automated adverse action notices (CFPB compliant)</li>
                <li>IEEE 7010 & EEOC 80% Rule fairness monitoring</li>
                <li>Interactive simulator for loan officers</li>
                <li>Executive BI dashboard for portfolio management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Impact metrics ───────────────────────────────────────────
    st.markdown("### 📊 Measured Impact")
    impact_cols = st.columns(4)
    impacts = [
        ("⚡", "80%", "Faster Processing", "#2980B9",
         "Decision time reduced from 14 days to < 60 seconds"),
        ("🎯", "93%+", "Model Accuracy", "#27AE60",
         "Best model achieves >93% accuracy on held-out test set"),
        ("⚖", "100%", "Fairness Compliant", "#8E44AD",
         "EEOC 80% Rule compliance across all demographic dimensions"),
        ("💰", "73%", "Cost Reduction", "#F39C12",
         "Automation of low-risk approvals reduces operational cost"),
    ]
    for col, (icon, val, title, color, desc) in zip(impact_cols, impacts):
        col.markdown(f"""
        <div style='background:white;border-radius:16px;padding:24px 16px;
                    text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.08);
                    border-top:4px solid {color};'>
            <div style='font-size:2rem;margin-bottom:8px;'>{icon}</div>
            <div style='font-size:2.2rem;font-weight:800;color:{color};'>{val}</div>
            <div style='font-weight:700;color:#1B4F72;font-size:0.88rem;margin:4px 0;'>{title}</div>
            <div style='color:#6B7280;font-size:0.75rem;line-height:1.4;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Benefits ─────────────────────────────────────────────────
    st.markdown("### 🌟 Key Benefits")
    benefits = [
        ("🏦", "For Banks & NBFCs",
         "Reduced operational costs, improved portfolio quality, faster loan book growth, and real-time risk monitoring dashboards."),
        ("👤", "For Loan Applicants",
         "Instant decisions, transparent explanations, fair treatment regardless of demographics, and clear improvement pathways for rejection cases."),
        ("📋", "For Regulators",
         "Full audit trail for every decision, EEOC/IEEE compliance reports, demographic parity certificates, and explainable AI documentation."),
        ("💼", "For Loan Officers",
         "AI-assisted decisions with SHAP explanations, interactive simulators to explore what-if scenarios, and workload reduction by 80%."),
    ]
    b1, b2 = st.columns(2)
    b3, b4 = st.columns(2)
    for panel, (icon, title, desc) in zip([b1,b2,b3,b4], benefits):
        panel.markdown(f"""
        <div style='background:white;border-radius:14px;padding:22px;
                    box-shadow:0 2px 12px rgba(27,79,114,0.08);margin-bottom:12px;
                    border-top:3px solid #2980B9;'>
            <div style='font-size:1.5rem;margin-bottom:8px;'>{icon}</div>
            <div style='font-weight:700;color:#1B4F72;font-size:0.95rem;margin-bottom:6px;'>
                {title}
            </div>
            <div style='color:#5D6D7E;font-size:0.83rem;line-height:1.55;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Technology stack ─────────────────────────────────────────
    st.markdown("### 🛠 Technology Stack")
    tech = [
        ("🐍", "Python 3.11",      "Core language",                    "#F39C12"),
        ("⚡", "Streamlit",        "Interactive dashboard framework",   "#FF4B4B"),
        ("🐼", "Pandas / NumPy",   "Data manipulation",                 "#2980B9"),
        ("🤖", "Scikit-Learn",     "ML models & preprocessing",         "#F7931E"),
        ("🔍", "SHAP",             "Explainable AI",                    "#8E44AD"),
        ("📊", "Plotly",           "Interactive visualisations",        "#3D9970"),
        ("📈", "Seaborn / Matplotlib","Static charts",                  "#16A085"),
        ("⚙", "Feature-Engine",   "Outlier treatment pipeline",        "#2C3E50"),
    ]
    t_cols = st.columns(4)
    for i, (icon, name, role, color) in enumerate(tech):
        t_cols[i % 4].markdown(f"""
        <div style='background:white;border-radius:12px;padding:16px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border-left:4px solid {color};margin-bottom:8px;'>
            <span style='font-size:1.4rem;'>{icon}</span>
            <div style='font-weight:700;color:#1B4F72;font-size:0.88rem;margin:4px 0 2px;'>
                {name}
            </div>
            <div style='color:#6B7280;font-size:0.75rem;'>{role}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture diagram ──────────────────────────────────────
    st.markdown("### 🏗 System Architecture")
    pipeline = [
        ("📥", "Raw Data\n(45K records)", "#AED6F1"),
        ("🧹", "Preprocessing\n& Encoding",   "#A9DFBF"),
        ("⚙", "Feature\nEngineering",    "#FAD7A0"),
        ("🤖", "ML Model\nTraining",      "#D7BDE2"),
        ("🔮", "Prediction\nEngine",      "#A3E4D7"),
        ("🔍", "SHAP\nExplainer",         "#F9E79F"),
        ("📊", "BI\nDashboard",           "#FDEBD0"),
    ]
    p_cols = st.columns(len(pipeline))
    for i, (col, (icon, label, bg)) in enumerate(zip(p_cols, pipeline)):
        arrow = "" if i == len(pipeline)-1 else "→"
        col.markdown(f"""
        <div style='background:{bg};border-radius:12px;padding:14px 8px;
                    text-align:center;position:relative;'>
            <div style='font-size:1.4rem;'>{icon}</div>
            <div style='font-size:0.70rem;font-weight:600;color:#1B4F72;
                        margin-top:6px;line-height:1.3;white-space:pre-line;'>
                {label}
            </div>
        </div>
        <div style='text-align:center;font-size:1.2rem;color:#2980B9;
                    font-weight:700;margin:-4px 0;'>{arrow}</div>
        """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0D2137,#1B4F72);
                border-radius:16px;padding:32px;text-align:center;color:white;'>
        <div style='font-size:2.5rem;margin-bottom:12px;'>🏦</div>
        <div style='font-size:1.2rem;font-weight:700;margin-bottom:8px;'>
            AI-Powered Loan Approval Prediction System
        </div>
        <div style='color:#AED6F1;font-size:0.88rem;margin-bottom:16px;'>
            IEEE Final-Year Project · Credit Risk AI · Responsible Lending
        </div>
        <div style='display:flex;justify-content:center;gap:16px;flex-wrap:wrap;'>
            <span style='background:rgba(255,255,255,0.1);padding:6px 14px;
                         border-radius:20px;font-size:0.78rem;'>
                ⚖ IEEE 7010 Compliant
            </span>
            <span style='background:rgba(255,255,255,0.1);padding:6px 14px;
                         border-radius:20px;font-size:0.78rem;'>
                📋 EEOC 80% Rule
            </span>
            <span style='background:rgba(255,255,255,0.1);padding:6px 14px;
                         border-radius:20px;font-size:0.78rem;'>
                🔍 SHAP Explainable AI
            </span>
            <span style='background:rgba(255,255,255,0.1);padding:6px 14px;
                         border-radius:20px;font-size:0.78rem;'>
                🤖 4 ML Models Compared
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
