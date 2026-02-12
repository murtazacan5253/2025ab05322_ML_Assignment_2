# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from sklearn.neighbors import KNeighborsClassifier
import joblib

def knnClassifier ():
    _, _, Xs_train, Xs_test, y_train, y_test = dataPreProcFeatureEng()

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(Xs_train,y_train)

    pred = model.predict(Xs_test)
    prob = model.predict_proba(Xs_test)[:,1]

    joblib.dump(model,"model/knn.pkl")
    return (y_test, pred)