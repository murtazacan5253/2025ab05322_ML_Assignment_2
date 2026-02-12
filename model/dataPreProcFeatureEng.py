# Import Dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def dataPreProcFeatureEng_MainData():

    # Add column names as not present in the UCI repo data
    columns = [
        "age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week",
        "native_country","income"
    ]

    df = pd.read_csv(
        "adult.data",
        header=None,
        names=columns,
        sep=",",
        skipinitialspace=True
    )

    # Handle missing values, if any
    print((df.isin(["?", " ?"])).sum())

    # Convert "?" → NaN
    df.replace((["?", " ?"]), np.nan, inplace=True) 

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Separate Features & Target
    X = df.drop("income", axis=1)
    y = df["income"]

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

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

def dataPreProcFeatureEng_TestData (df,true):
    df.replace((["?", " ?"]), np.nan, inplace=True)
    df.dropna(inplace=True)

    # Separate target
    y = df["income"]
    X = df.drop("income", axis=1)

    # Encode target
    encode_target_test = LabelEncoder()
    y = encode_target_test.fit_transform(y)

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Align with training columns
    training_cols = joblib.load("model/feature_columns.pkl")
    X = X.reindex(columns=training_cols, fill_value=0)

    # Scaling if required
    if scale_required:
        sclare=scaler = StandardScaler()
        //scaler = joblib.load("model/scaler.pkl")
        X_scaled = scaler.transform(X)

    return X_scaled, y
