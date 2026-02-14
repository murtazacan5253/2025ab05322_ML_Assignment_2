# Import Dependencies
from fairlearn.datasets import fetch_adult
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def dataProcScaling_TrainData():
    adult_data = fetch_adult(as_frame=True)
    df=adult_data.frame

    # Information about dataset
    df.info()
    
    #Share before data pre-processing
    print("Train shape before processing:", df.shape)

    # Remove irrelavant columns
    df = df.drop(columns=['fnlwgt'])
    df = df.drop(columns=['education-num'])

    # Handle missing values, if any
    # Convert "?" → NaN
    df.replace((["?", " ?"]), np.nan, inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Separate Features & Target
    target_col = 'class' # Standard target name in OpenML version
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode Target Variable
    encode_target = LabelEncoder()
    y = encode_target.fit_transform(y)

    # One-Hot Encode Categorical Features
    X = pd.get_dummies(X, drop_first=True)
    joblib.dump(X.columns, "model/trainFeatureColumns.pkl")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler,"model/scaler.pkl")
    X_test_scaled = scaler.transform(X_test)


    # Verification
    print("Train shape after processing:", X_train_scaled.shape)
    print("Test shape after processing:", X_test_scaled.shape)

    return(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)

def dataProcScaling_TestData():
    
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

    df_test = pd.read_csv(
    "adult.test",
    header=None,
    names=columns,
    sep=",",
    skipinitialspace=True
    )

    # Information about dataset
    df_test.info()

    #Share before data pre-processing
    print("Train shape before processing:", df_test.shape)

    # Remove irrelavant columns
    df_test = df_test.drop(columns=['fnlwgt'])
    df_test = df_test.drop(columns=['education-num'])

    # Handle missing values, if any
    # Convert "?" → NaN
    df_test.replace((["?", " ?"]), np.nan, inplace=True)

    # Drop rows with missing values
    df_test.dropna(inplace=True)

    # Separate Features & Target
    target_col = 'class' # Standard target name in OpenML version
    X_testData = df_test.drop(columns=[target_col])
    y_testData = df_test[target_col]

    # Encode Target Variable
    encode_target = LabelEncoder()
    y_testData = encode_target.fit_transform(y_testData)

    # One-Hot Encode Categorical Features
    X_testData = pd.get_dummies(X_testData, drop_first=True)
    training_cols = joblib.load("model/trainFeatureColumns.pkl")
    X_testData = X_testData.reindex(columns=training_cols, fill_value=0)

    # Feature Scaling
    scaler = joblib.load("model/scaler.pkl")

    X_test_scaledData = scaler.transform(X_testData)


    # Verification
    print("Test shape after processing:", X_test_scaledData.shape)

    return (X_testData, X_test_scaledData, y_testData)
