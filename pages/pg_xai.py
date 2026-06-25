"""Page 7 – Explainable AI (SHAP)"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils import train_all_models, get_processed_data


def render():
    st.markdown("""
    <div class='section-header'>
        <h2>🔍 Explainable AI Center</h2>
        <p style='color:#AED6F1;margin:4px 0 0;font-size:0.88rem;'>
            SHAP values · Feature importance · Plain-English decision explanations
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading models and computing SHAP values …"):
        results, trained, X_train, X_test, y_train, y_test, features = train_all_models()
        df_proc, ss, mms, high_corr = get_processed_data()

    best_name = max(results, key=lambda k: results[k]['F1 Score'])
    rf_model  = trained['Random Forest']   # RF for SHAP TreeExplainer

    X_test_df = pd.DataFrame(X_test, columns=features)

    tab1, tab2, tab3 = st.tabs([
        "📊 Feature Importance",
        "🌊 SHAP Beeswarm",
        "📝 Single Prediction Explanation",
    ])

    # ─ Tab 1: Feature Importance ─────────────────────────────────
    with tab1:
        st.markdown("### 🏅 Global Feature Importance (Random Forest)")
        st.markdown("""
        <div class='insight-box'>
            📐 <b>Mean Decrease in Impurity (MDI)</b> measures how much each feature
            reduces uncertainty across all decision trees. Higher = more influential.
        </div>""", unsafe_allow_html=True)

        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=True)

        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     title='Feature Importance (Random Forest)',
                     labels={'Importance':'Importance Score'})
        fig.update_layout(height=500, coloraxis_showscale=False,
                          yaxis=dict(tickfont=dict(size=11)),
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Top factors
        top = fi_df.sort_values('Importance', ascending=False).head(5)
        st.markdown("#### 🔑 Top 5 Decision Drivers")
        emoji_map = {0:'🥇',1:'🥈',2:'🥉',3:'4️⃣',4:'5️⃣'}
        for i, (_, row) in enumerate(top.iterrows()):
            pct = row['Importance'] / fi_df['Importance'].sum() * 100
            st.markdown(f"""
            <div style='background:white;border-radius:10px;padding:14px 18px;
                        margin-bottom:8px;box-shadow:0 2px 8px rgba(0,0,0,0.06);
                        display:flex;align-items:center;gap:16px;'>
                <div style='font-size:1.6rem;min-width:36px;'>{emoji_map[i]}</div>
                <div style='font-weight:600;color:#1B4F72;flex:1;'>
                    {row['Feature']}
                </div>
                <div style='color:#2980B9;font-weight:700;'>{row['Importance']:.4f}</div>
                <div style='width:120px;background:#EBF5FB;border-radius:8px;height:10px;'>
                    <div style='background:#2980B9;height:10px;border-radius:8px;
                                width:{pct*3:.0f}px;max-width:120px;'></div>
                </div>
                <div style='color:#6B7280;font-size:0.82rem;min-width:40px;'>{pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    # ─ Tab 2: SHAP Beeswarm (approximated) ───────────────────────
    with tab2:
        st.markdown("### 🌊 SHAP Value Distribution (Sample)")
        st.markdown("""
        <div class='insight-box'>
            🔬 <b>SHAP (SHapley Additive exPlanations)</b> assigns each feature a value
            representing its contribution to the prediction for each individual.
            Points to the right = push toward approval; left = push toward rejection.
        </div>""", unsafe_allow_html=True)

        try:
            import shap
            sample_idx = np.random.choice(len(X_test_df), size=min(300, len(X_test_df)), replace=False)
            X_sample   = X_test_df.iloc[sample_idx]

            explainer   = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_sample)

            # For binary classification shap_values is list [class0, class1]
            if isinstance(shap_values, list):
                sv = shap_values[1]   # class 1 (approved)
            else:
                sv = shap_values

            shap_df = pd.DataFrame(sv, columns=features)
            mean_abs = shap_df.abs().mean().sort_values(ascending=False)

            # Plot mean |SHAP|
            fig = px.bar(
                x=mean_abs.values,
                y=mean_abs.index,
                orientation='h',
                color=mean_abs.values,
                color_continuous_scale='RdBu',
                title='Mean |SHAP Value| — Global Impact',
                labels={'x': 'Mean |SHAP|', 'y': 'Feature'},
            )
            fig.update_layout(height=480, coloraxis_showscale=False,
                               margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # SHAP scatter for top feature
            top_feat = mean_abs.index[0]
            scatter_df = pd.DataFrame({
                'Feature Value':  X_sample[top_feat].values,
                'SHAP Value':     shap_df[top_feat].values,
            })
            fig2 = px.scatter(scatter_df, x='Feature Value', y='SHAP Value',
                              color='SHAP Value', color_continuous_scale='RdBu',
                              title=f'SHAP Dependency Plot — {top_feat}',
                              opacity=0.6)
            fig2.add_hline(y=0, line_dash='dash', line_color='black')
            fig2.update_layout(height=360)
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.warning(f"SHAP computation skipped (library issue): {e}")
            st.info("Falling back to feature importance chart.")

    # ─ Tab 3: Single prediction explanation ──────────────────────
    with tab3:
        st.markdown("### 📝 Explain a Single Decision")
        idx = st.slider("Select test record index", 0, len(X_test_df)-1, 0, key="xai_idx")

        row = X_test_df.iloc[[idx]]
        true_label = int(y_test.iloc[idx]) if hasattr(y_test, 'iloc') else int(y_test[idx])
        prob_val   = rf_model.predict_proba(row)[0][1]

        decision = "✅ Approved" if prob_val >= 0.5 else "❌ Rejected"
        truth    = "✅ Approved" if true_label == 1 else "❌ Rejected"
        correct  = (prob_val >= 0.5) == (true_label == 1)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class='kpi-card' style='border-top-color:#2980B9;'>
            <div class='kpi-value' style='color:#2980B9;'>{prob_val*100:.1f}%</div>
            <div class='kpi-label'>Model Probability</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""
        <div class='kpi-card' style='border-top-color:{"#27AE60" if prob_val>=0.5 else "#E74C3C"};'>
            <div class='kpi-value' style='font-size:1rem;
                color:{"#27AE60" if prob_val>=0.5 else "#E74C3C"};'>{decision}</div>
            <div class='kpi-label'>Predicted</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""
        <div class='kpi-card' style='border-top-color:{"#27AE60" if correct else "#E74C3C"};'>
            <div class='kpi-value' style='font-size:1rem;
                color:{"#27AE60" if correct else "#E74C3C"};'>
                {"✅ Correct" if correct else "❌ Incorrect"}
            </div>
            <div class='kpi-label'>vs Ground Truth ({truth})</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Waterfall-style explanation using feature importances as proxy
        importances = rf_model.feature_importances_
        feat_vals   = row.values.flatten()
        contributions = importances * feat_vals
        contrib_df = pd.DataFrame({
            'Feature':      features,
            'Contribution': contributions,
            'Direction':    ['Positive' if c > 0 else 'Negative' for c in contributions]
        }).sort_values('Contribution', key=abs, ascending=False)

        colors = ['#27AE60' if c > 0 else '#E74C3C'
                  for c in contrib_df['Contribution']]

        fig = go.Figure(go.Bar(
            x=contrib_df['Contribution'],
            y=contrib_df['Feature'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.4f}" for v in contrib_df['Contribution']],
            textposition='outside',
        ))
        fig.add_vline(x=0, line_color='black', line_width=1)
        fig.update_layout(
            title='Feature Contribution Waterfall (Proxy)',
            height=460, margin=dict(t=40, b=10),
            xaxis_title='Contribution to Approval Score',
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plain English
        pos_top = contrib_df[contrib_df['Contribution']>0].head(3)['Feature'].tolist()
        neg_top = contrib_df[contrib_df['Contribution']<0].head(3)['Feature'].tolist()

        st.markdown("#### 🗣 Plain-English Explanation")
        if pos_top:
            positives = ", ".join(pos_top)
            st.success(f"✅ **Supporting approval:** {positives} — these features indicate lower credit risk and stronger repayment capacity.")
        if neg_top:
            negatives = ", ".join(neg_top)
            st.error(f"❌ **Opposing approval:** {negatives} — these features signal elevated default risk or insufficient creditworthiness.")

        st.markdown(f"""
        <div class='insight-box'>
            🤖 <b>Model Confidence:</b> {prob_val*100:.1f}% approval probability.
            {"The applicant comfortably clears the 50% decision threshold." if prob_val>=0.5
             else "The applicant does not clear the 50% approval threshold."}
            To improve approval odds: boost credit score, reduce loan-to-income ratio,
            and eliminate any outstanding defaults.
        </div>""", unsafe_allow_html=True)
