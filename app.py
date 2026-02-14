import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from model.dataPreProcFeatureEng import dataProcScaling_TestData
from model.evaluateMetrics import getMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Title for App
st.set_page_config(page_title="ML Model Evaluator", layout="wide")
st.title("Classification Models Performance Evaluator")
st.write("Upload your dataset to compare 6 different Classification ML models.")

# Upload Test Data
columns =[
"age",
"workclass",
"fnlwgt",
"education",
"education-num",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"capital-gain",
"capital-loss",
"hours-per-week",
"native-country",
"class"
]

uploaded_file = st.file_uploader("Choose file")
if uploaded_file is not None:
    
    uploaded_file.seek(0)
    try:
        df_test = pd.read_csv(
        uploaded_file,
        header=None,
        names=columns,
        sep=",",
        skipinitialspace=True
        )
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop() # Stops execution if file is bad



    # Test data info
    st.write("Test data :", df_test.shape)

    # Data processing on test data
    X_testData, X_testData_scaled, y_testData=dataProcScaling_TestData(df_test)


    # Model selection
    st.subheader("Select Models to Evaluate")
    model_dict = {
        "Logistic Regression": "logisticRegression",
        "Decision Tree": "decisionTree",
        "KNN": "knnClassifier",
        "Naive Bayes": "naiveBayes",
        "Random Forest": "randomForest",
        "XGBoost": "xgBoost"
    }

    scaledModels =["Logistic Regression", "KNN", "Naive Bayes"]


    # Allow multiple model selection
    model_names = st.multiselect("Select Model(s)", list(model_dict.keys()), default=list(model_dict.keys())[:1])
    if len(model_names)>0:
        metrics_list = []
        confusion_matrices = {}
        predictions = {}

        # Provide model path for execution
        for model_name in model_names:
            model_file_key = model_dict[model_name]
            model_path = os.path.join("model/pkl", f"{model_file_key}.pkl")
            print("------------------------------------------------------")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
            else:
                st.error(f"Model file not found: {model_path}")
                continue

        # Display Metric Evaluation
            
            if model_name not in scaledModels:
            #metrics_list.append(getMetrics(model,X_testData,y_testData))
                metrics,y_pred,y_prob=getMetrics(model,X_testData,y_testData)
                metrics['Model'] = model_name
                metrics_list.append(metrics)
                cm = confusion_matrix(y_testData, y_pred)
                confusion_matrices[model_name] = cm
            else:
            #metrics_list.append(getMetrics(model,X_test_scaledData,y_testData)) 
                metrics,y_pred,y_prob=getMetrics(model,X_testData_scaled,y_testData)
                metrics['Model'] = model_name
                metrics_list.append(metrics)
                cm = confusion_matrix(y_testData, y_pred)
                confusion_matrices[model_name] = cm
            
        st.subheader("Evaluation Metrics Comparison")
        metrics_df = pd.DataFrame(metrics_list).set_index('Model')
        st.dataframe(metrics_df)


        # Show Confusion Matrix
        st.subheader("Confusion Matrix")
        for model_name in model_names:
            st.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
            ax.set_title(f"Confusion Matrix - {model_name}")
            st.pyplot(fig)