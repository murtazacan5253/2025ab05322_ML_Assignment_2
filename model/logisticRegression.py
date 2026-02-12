# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from sklearn.linear_model import LogisticRegression
import joblib

# Logistic Regression
def logisticRegression ():
    _, _, Xs_train, Xs_test, y_train, y_test = dataPreProcFeatureEng()

    model = LogisticRegression(max_iter=200)
    model.fit(Xs_train, y_train)

    pred = model.predict(Xs_test)
    prob = model.predict_proba(Xs_test)[:,1]

    joblib.dump(model,"model/logistic.pkl")
    return (y_test, pred)