# Import Dependencies
from sklearn.neighbors import KNeighborsClassifier
import joblib

# K-Nearest Neighbour Classifier - train on scaled data
def trainKnnClassifier (Xs_train, y_train):
    #_, _, Xs_train, Xs_test, y_train, y_test = dataPreProcFeatureEng()

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(Xs_train,y_train)

    joblib.dump(model,"model/pkl/knnClassifier.pkl")
    return model