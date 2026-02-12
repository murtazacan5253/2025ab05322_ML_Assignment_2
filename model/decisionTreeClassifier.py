# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from sklearn.tree import DecisionTreeClassifier
import joblib

# Decision Tree
def trainDecisionTree(X_train, y_train):
    #X_train, X_test, _, _, y_train, y_test = dataPreProcFeatureEng()

    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)

    joblib.dump(model,"model/tree.pkl")
    return model