import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import confusion_matrix

# Title for App
st.set_page_config(page_title="ML Model Evaluator", layout="wide")
st.title("Classification Model Performance Evaluator")
st.write("Upload your dataset to compare 6 different ML models.")

# Model selection
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Provide model path for execution
model_paths = {
    "Logistic Regression":"model/logistic.pkl",
    "Decision Tree":"model/tree.pkl",
    "KNN":"model/knn.pkl",
    "Naive Bayes":"model/nb.pkl",
    "Random Forest":"model/rf.pkl",
    "XGBoost":"model/xgb.pkl"
}

# Display Metric Evaluation



# Show Confusion Matrix
cm = confusion_matrix(y, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.subheader("Confusion Matrix")
st.pyplot(fig)
st.write(cm)