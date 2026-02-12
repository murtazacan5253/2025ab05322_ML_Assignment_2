# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from sklearn.tree import DecisionTreeClassifier
import joblib

# Decision Tree
def decisionTree():
    X_train, X_test, _, _, y_train, y_test = dataPreProcFeatureEng()

    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    joblib.dump(model,"model/tree.pkl")
    return (y_test, pred)