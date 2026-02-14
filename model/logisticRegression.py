# Import Dependencies
from sklearn.linear_model import LogisticRegression
import joblib

# Logistic Regression - train on scaled data
def trainLogisticRegression (Xs_train,y_train):
    #_, _, Xs_train, Xs_test, y_train, y_test = dataPreProcFeatureEng()

    model = LogisticRegression(max_iter=200)
    model.fit(Xs_train, y_train)

    joblib.dump(model,"model/logisticRegression.pkl")
    return model