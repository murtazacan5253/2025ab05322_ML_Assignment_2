# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from xgboost import XGBClassifier
import joblib

# XG Boost
def trainXGBoost(X_train, y_train):
    X_train, X_test, _, _, y_train, y_test = dataPreProcFeatureEng()

    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train,y_train)

    joblib.dump(model,"model/xgb.pkl")
    return model