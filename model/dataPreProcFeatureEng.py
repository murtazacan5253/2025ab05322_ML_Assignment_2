# Import Dependencies
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def dataPreProcFeatureEng_TrainData():

    adultIncome_data = fetch_openml('adult', version=2, as_frame=True)
    df=adultIncome_data.frame

    # Information about dataset
    df.info()


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

    #Share before data pre-processing
    print("Train shape:", X.shape)


    # Encode Target Variable
    encode_target_train = LabelEncoder()
    y = encode_target_train.fit_transform(y)

    # One-Hot Encode Categorical Features
    X = pd.get_dummies(X, drop_first=True)

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
    X_test_scaled = scaler.transform(X_test)


    # Verification
    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)

    return(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)