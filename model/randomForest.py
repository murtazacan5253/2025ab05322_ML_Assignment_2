# Import Dependencies
from sklearn.ensemble import RandomForestClassifier
import joblib

# Random Forest - train on non-scaled data
def trainRandomForest(X_train, y_train):
    #X_train, X_test, _, _, y_train, y_test = dataPreProcFeatureEng()

    model = RandomForestClassifier()
    model.fit(X_train,y_train)


    joblib.dump(model,"model/randomForest.pkl")
    return model