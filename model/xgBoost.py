# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from xgboost import XGBClassifier
import joblib

# XG Boost
def xgBoost():
    X_train, X_test, _, _, y_train, y_test = dataPreProcFeatureEng()

    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    joblib.dump(model,"model/xgb.pkl")
    return (y_test, pred)