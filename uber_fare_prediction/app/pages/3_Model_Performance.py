"""
Model Performance Dashboard
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px

from src.dashboard_utils import (
    load_model_comparison,
    load_feature_importance,
    load_random_search_results,
)

st.title("📈 Model Performance")

st.markdown(
    """
Evaluate the performance of the trained Random Forest model and
review the hyperparameter optimization results.
"""
)

st.divider()

# ==========================================================
# Load Reports
# ==========================================================

comparison_df = load_model_comparison()
importance_df = load_feature_importance()
random_df = load_random_search_results()

# ==========================================================
# KPI Cards
# ==========================================================

st.subheader("Performance Metrics")

best_model = comparison_df.iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", f"{best_model['MAE']:.3f}")

with col2:
    st.metric("RMSE", f"{best_model['RMSE']:.3f}")

with col3:
    st.metric("R² Score", f"{best_model['R²']:.3f}")

st.divider()

# ==========================================================
# Model Comparison
# ==========================================================

st.subheader("Model Comparison")

fig = px.bar(
    comparison_df,
    x="Model",
    y="R²",
    text="R²",
    title="Model Performance Comparison",
)

fig.update_traces(texttemplate="%{text:.3f}")

st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    comparison_df,
    use_container_width=True,
)

st.divider()

# ==========================================================
# Feature Importance
# ==========================================================

st.subheader("Feature Importance")

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True,
)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance",
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    importance_df.sort_values(
        by="Importance",
        ascending=False,
    ),
    use_container_width=True,
)

st.divider()

# ==========================================================
# Hyperparameter Optimization
# ==========================================================

st.subheader("Random Search Results")

best_trial = random_df.loc[
    random_df["mean_test_score"].idxmax()
]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Best CV Score",
        f"{best_trial['mean_test_score']:.4f}",
    )

with col2:
    st.metric(
        "Best Trees",
        int(best_trial["param_n_estimators"]),
    )

fig = px.scatter(
    random_df,
    x="param_n_estimators",
    y="mean_test_score",
    color="param_max_depth",
    size="mean_fit_time",
    hover_data=[
        "param_min_samples_split",
        "param_min_samples_leaf",
    ],
    title="Random Search Performance",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

with st.expander("View Random Search Results"):
    st.dataframe(
        random_df,
        use_container_width=True,
    )

st.divider()

st.success(
    f"""
Best Model: **{best_model['Model']}**

R² Score: **{best_model['R²']:.3f}**

The Random Forest model provides strong predictive performance
for Uber fare estimation.
"""
)