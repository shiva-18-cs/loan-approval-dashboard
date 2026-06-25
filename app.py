"""
app.py  –  Entry point for the Loan Approval Prediction Dashboard.
Run with:  streamlit run app.py
"""
import streamlit as st

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="AI Loan Approval Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──
st.markdown("""
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D2137 0%, #1B4F72 60%, #2980B9 100%);
    color: white;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #E8F4FD !important;
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}

/* Radio buttons styled as nav pills */
div[data-testid="stSidebarNav"] { display: none; }

/* Main bg */
.main { background-color: #F0F4F8; }

/* KPI card */
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(27,79,114,0.10);
    border-top: 4px solid;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(27,79,114,0.18);
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.85rem;
    color: #6B7280;
    font-weight: 500;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.kpi-icon { font-size: 1.8rem; margin-bottom: 6px; }

/* Section headers */
.section-header {
    background: linear-gradient(135deg, #1B4F72 0%, #2980B9 100%);
    color: white !important;
    padding: 16px 24px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 15px rgba(27,79,114,0.2);
}
.section-header h2 { color: white !important; margin: 0; }

/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, #EBF5FB 0%, #D6EAF8 100%);
    border-left: 4px solid #2980B9;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 12px 0;
    font-size: 0.93rem;
    color: #1B4F72;
}

/* Success / danger badges */
.badge-success {
    background: #D5F5E3; color: #1E8449;
    padding: 4px 12px; border-radius: 20px;
    font-weight: 600; font-size: 0.82rem;
}
.badge-danger {
    background: #FADBD8; color: #C0392B;
    padding: 4px 12px; border-radius: 20px;
    font-weight: 600; font-size: 0.82rem;
}

/* Workflow step cards */
.workflow-card {
    background: white;
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(27,79,114,0.08);
    border-bottom: 3px solid #2980B9;
}

/* Dividers */
hr { border-color: #D5E8F5 !important; }

/* Streamlit tables */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Hide hamburger / footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:3rem;'>🏦</div>
        <div style='font-size:1.05rem; font-weight:700; color:#E8F4FD; line-height:1.3;'>
            AI Loan Approval<br>Dashboard
        </div>
        <div style='font-size:0.72rem; color:#AED6F1; margin-top:4px;'>
            IEEE Project · Credit Risk AI
        </div>
    </div>
    <hr style='border-color:#2980B9; margin: 0 0 16px 0;'>
    """, unsafe_allow_html=True)

    pages = {
        "🏠  Home"                      : "Home",
        "📋  Dataset Explorer"          : "Dataset",
        "🧹  Data Preprocessing"        : "Preprocessing",
        "📊  Exploratory Data Analysis" : "EDA",
        "🤖  ML Model Center"           : "Models",
        "🎯  Loan Approval Simulator"   : "Simulator",
        "🔍  Explainable AI"            : "XAI",
        "⚖  Fair AI & Bias Monitor"    : "FairAI",
        "📈  Business Intelligence"     : "BI",
        "🏆  Project Impact"            : "Impact",
    }

    selection = st.radio(
        "Navigate",
        list(pages.keys()),
        label_visibility="collapsed",
    )
    page = pages[selection]

    st.markdown("""
    <hr style='border-color:#2980B9; margin:20px 0 12px 0;'>
    <div style='font-size:0.72rem; color:#AED6F1; text-align:center;'>
        Built with ❤ · Streamlit · Scikit-Learn<br>
        IEEE 7010 Compliant · EEOC 80% Rule
    </div>
    """, unsafe_allow_html=True)


# ── Page Router ──
if page == "Home":
    from pages import pg_home as pg
elif page == "Dataset":
    from pages import pg_dataset as pg
elif page == "Preprocessing":
    from pages import pg_preprocessing as pg
elif page == "EDA":
    from pages import pg_eda as pg
elif page == "Models":
    from pages import pg_models as pg
elif page == "Simulator":
    from pages import pg_simulator as pg
elif page == "XAI":
    from pages import pg_xai as pg
elif page == "FairAI":
    from pages import pg_fairai as pg
elif page == "BI":
    from pages import pg_bi as pg
elif page == "Impact":
    from pages import pg_impact as pg

pg.render()
