# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from sklearn.naive_bayes import GaussianNB
import joblib

# Navie Bayes Classifier  - Gaussian
def naiveBayesClassifier ():
    _, _, Xs_train, Xs_test, y_train, y_test = dataPreProcFeatureEng()

    model = GaussianNB()
    model.fit(Xs_train,y_train)

    pred = model.predict(Xs_test)
    prob = model.predict_proba(Xs_test)[:,1]

    joblib.dump(model,"model/nb.pkl")
    return (y_test, pred)