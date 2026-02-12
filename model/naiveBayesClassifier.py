# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from sklearn.naive_bayes import GaussianNB
import joblib

# Navie Bayes Classifier  - Gaussian
def trainNaiveBayesClassifier (Xs_train,y_train):
    #_, _, Xs_train, Xs_test, y_train, y_test = dataPreProcFeatureEng()

    model = GaussianNB()
    model.fit(Xs_train,y_train)

    joblib.dump(model,"model/nb.pkl")
    return model