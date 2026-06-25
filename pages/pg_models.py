"""Page 5 – ML Model Center"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils import train_all_models


def _confusion_fig(cm, title, color):
    labels = ['Rejected (0)', 'Approved (1)']
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=['Pred: Rejected', 'Pred: Approved'],
        y=['Act: Rejected', 'Act: Approved'],
        text=cm, texttemplate='<b>%{text}</b>',
        textfont=dict(size=18),
        colorscale=[[0,'#EBF5FB'],[1,color]],
        showscale=False,
    ))
    fig.update_layout(
        title=title, height=280,
        margin=dict(t=40, b=10, l=10, r=10),
        xaxis=dict(side='bottom'),
    )
    return fig


def render():
    st.markdown("""
    <div class='section-header'>
        <h2>🤖 Machine Learning Model Center</h2>
        <p style='color:#AED6F1;margin:4px 0 0;font-size:0.88rem;'>
            Train · Evaluate · Compare · Select the best model
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("⚙ Training all models … this may take 30-60 seconds on first load"):
        results, trained, X_train, X_test, y_train, y_test, features = train_all_models()

    # ── Leaderboard ───────────────────────────────────────────────
    st.markdown("### 🏆 Model Performance Leaderboard")

    leaderboard = pd.DataFrame([
        {'Model': k,
         'Accuracy (%)':  v['Accuracy'],
         'Precision (%)': v['Precision'],
         'Recall (%)':    v['Recall'],
         'F1 Score (%)':  v['F1 Score']}
        for k, v in results.items()
    ]).sort_values('F1 Score (%)', ascending=False).reset_index(drop=True)

    leaderboard.index = leaderboard.index + 1
    best_model_name = leaderboard.iloc[0]['Model']

    # Styled rank medals
    medals = {1:'🥇', 2:'🥈', 3:'🥉', 4:'4️⃣'}
    for i, row in leaderboard.iterrows():
        medal = medals.get(i,'')
        is_best = (i == 1)
        bg = 'linear-gradient(135deg,#D5F5E3,#A9DFBF)' if is_best else 'white'
        border = '#27AE60' if is_best else '#D5E8F5'
        st.markdown(f"""
        <div style='background:{bg};border:2px solid {border};border-radius:14px;
                    padding:16px 20px;margin-bottom:10px;display:flex;
                    align-items:center;gap:20px;box-shadow:0 2px 10px rgba(0,0,0,0.06);'>
            <div style='font-size:1.8rem;min-width:40px;text-align:center;'>{medal}</div>
            <div style='font-weight:700;color:#1B4F72;min-width:180px;font-size:1rem;'>
                {row['Model']}
                {'<span style="background:#27AE60;color:white;padding:2px 10px;'
                 'border-radius:12px;font-size:0.72rem;margin-left:8px;">BEST</span>' if is_best else ''}
            </div>
            <div style='display:flex;gap:24px;flex-wrap:wrap;'>
                <div><span style='color:#6B7280;font-size:0.78rem;'>Accuracy</span><br>
                     <b style='color:#1B4F72;'>{row['Accuracy (%)']:.2f}%</b></div>
                <div><span style='color:#6B7280;font-size:0.78rem;'>Precision</span><br>
                     <b style='color:#8E44AD;'>{row['Precision (%)']:.2f}%</b></div>
                <div><span style='color:#6B7280;font-size:0.78rem;'>Recall</span><br>
                     <b style='color:#D35400;'>{row['Recall (%)']:.2f}%</b></div>
                <div><span style='color:#6B7280;font-size:0.78rem;'>F1 Score</span><br>
                     <b style='color:#27AE60;'>{row['F1 Score (%)']:.2f}%</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison chart ──────────────────────────────────────────
    st.markdown("### 📊 Interactive Metric Comparison")
    metrics  = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']
    fig = go.Figure()
    model_colors = ['#2980B9','#8E44AD','#D35400','#27AE60']
    for i, row in leaderboard.iterrows():
        fig.add_trace(go.Bar(
            name=row['Model'],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=model_colors[(i-1) % 4],
            text=[f"{row[m]:.1f}%" for m in metrics],
            textposition='outside',
        ))

    fig.update_layout(
        barmode='group', height=420,
        yaxis=dict(range=[0, 115], title='Score (%)'),
        legend=dict(orientation='h', y=1.12),
        margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrices ────────────────────────────────────────
    st.markdown("### 🔢 Confusion Matrices")
    cm_colors = {'Logistic Regression':'#2980B9','SVM':'#8E44AD',
                  'KNN':'#D35400','Random Forest':'#27AE60'}
    model_names = list(results.keys())
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    panels  = [c1, c2, c3, c4]
    for col, name in zip(panels, model_names):
        cm   = results[name]['Confusion']
        color = cm_colors.get(name, '#2980B9')
        tn, fp, fn, tp = cm.ravel()
        with col:
            st.plotly_chart(_confusion_fig(cm, name, color), use_container_width=True)
            acc = (tp+tn) / (tp+tn+fp+fn) * 100
            st.caption(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}  |  Acc={acc:.2f}%")

    # ── Winner banner ─────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1E8449,#27AE60);
                border-radius:16px;padding:24px 28px;margin-top:20px;
                box-shadow:0 4px 20px rgba(39,174,96,0.30);'>
        <div style='font-size:1.6rem;margin-bottom:8px;'>🏆 Recommended Model</div>
        <div style='color:white;font-size:1.3rem;font-weight:700;'>
            {best_model_name}
        </div>
        <div style='color:#D5F5E3;font-size:0.88rem;margin-top:6px;'>
            Highest F1 Score on held-out test set · Used for the Loan Approval Simulator &
            Explainable AI sections
        </div>
    </div>
    """, unsafe_allow_html=True)
