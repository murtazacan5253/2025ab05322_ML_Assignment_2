# Import Dependencies
from xgboost import XGBClassifier
import joblib

# XG Boost - train on non-scaled data
def trainXGBoost(X_train, y_train):
    #X_train, X_test, _, _, y_train, y_test = dataPreProcFeatureEng()

    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train,y_train)

    joblib.dump(model,"model/pkl/xgBoost.pkl")
    return model