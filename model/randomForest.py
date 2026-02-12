# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from sklearn.ensemble import RandomForestClassifier
import joblib

# Random Forest
def randomForest():
    X_train, X_test, _, _, y_train, y_test = dataPreProcFeatureEng()

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    joblib.dump(model,"model/rf.pkl")
    return (y_test, pred)