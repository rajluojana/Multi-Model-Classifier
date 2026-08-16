"""
Multi-Model Classification Dashboard
Streamlit app satisfying FR-6 through FR-9 of REQUIREMENTS.md.

Upload a CSV (schema matching test_data.csv), pick one of the 5 trained models,
and view the 6 evaluation metrics plus a confusion matrix / classification report.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Multi-Model Classification Dashboard", layout="wide")

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "kNN": "kNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest (Ensemble)": "Random_Forest_Ensemble.pkl",
}

TARGET_COL = "target"


@st.cache_resource
def load_scaler_and_features():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return scaler, feature_names


@st.cache_resource
def load_model(model_key):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[model_key]))


@st.cache_data
def load_training_metrics():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


st.title("🔬 Multi-Model Classification Dashboard")
st.caption(
    "Breast Cancer Wisconsin (Diagnostic) dataset — 5 classical ML models, "
    "6 evaluation metrics, same train/test split and preprocessing for all models."
)

# ---- Sidebar: reference metrics from training (metrics.json is the shared source
# of truth per IMPLEMENTATION_GUIDE.md step 6 — not recomputed here) ----
training_metrics = load_training_metrics()
if training_metrics:
    with st.sidebar:
        st.header("📊 Training-time metrics")
        st.caption("From `model/metrics.json` (held-out split used during training)")
        ref_df = pd.DataFrame(training_metrics).T
        st.dataframe(ref_df.style.format("{:.4f}"), use_container_width=True)

# ---- Main controls ----
col_upload, col_select = st.columns(2)

with col_upload:
    st.subheader("1. Upload test data")
    uploaded_file = st.file_uploader(
        "Upload a CSV matching the schema of test_data.csv (features + 'target' column)",
        type="csv",
    )

with col_select:
    st.subheader("2. Select model")
    model_choice = st.selectbox("Choose which trained model to evaluate", list(MODEL_FILES.keys()))

st.divider()

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if TARGET_COL not in data.columns:
            st.error(
                f"Uploaded CSV is missing the required '{TARGET_COL}' column. "
                f"Found columns: {list(data.columns)}"
            )
        else:
            scaler, feature_names = load_scaler_and_features()

            missing = set(feature_names) - set(data.columns)
            if missing:
                st.error(
                    "Uploaded CSV is missing expected feature columns: "
                    f"{sorted(missing)}"
                )
            else:
                X = data.reindex(columns=feature_names, fill_value=0)
                y_true = data[TARGET_COL]

                X_scaled = scaler.transform(X)

                model = load_model(model_choice)
                y_pred = model.predict(X_scaled)
                y_proba = model.predict_proba(X_scaled)

                is_multiclass = y_true.nunique() > 2
                avg = "weighted" if is_multiclass else "binary"
                auc = (
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                    if is_multiclass
                    else roc_auc_score(y_true, y_proba[:, 1])
                )

                st.subheader(f"3. Results — {model_choice}")

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                m2.metric("AUC", f"{auc:.4f}")
                m3.metric(
                    "Precision",
                    f"{precision_score(y_true, y_pred, average=avg, zero_division=0):.4f}",
                )
                m4.metric(
                    "Recall",
                    f"{recall_score(y_true, y_pred, average=avg, zero_division=0):.4f}",
                )
                m5.metric(
                    "F1 Score",
                    f"{f1_score(y_true, y_pred, average=avg, zero_division=0):.4f}",
                )
                m6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

                cm_col, report_col = st.columns(2)

                with cm_col:
                    st.markdown("**Confusion Matrix**")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ConfusionMatrixDisplay(
                        confusion_matrix(y_true, y_pred)
                    ).plot(ax=ax, colorbar=False)
                    st.pyplot(fig)

                with report_col:
                    st.markdown("**Classification Report**")
                    st.text(classification_report(y_true, y_pred, zero_division=0))

                with st.expander("Preview uploaded data"):
                    st.dataframe(data.head(20), use_container_width=True)

    except Exception as e:
        st.error(f"Could not process the uploaded file: {e}")
else:
    st.info("⬆️ Upload a CSV file to see metrics, confusion matrix, and classification report.")
