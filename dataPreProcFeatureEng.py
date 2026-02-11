# Import Dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
encode_target = LabelEncoder()
y = encode_target.fit_transform(y)

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

