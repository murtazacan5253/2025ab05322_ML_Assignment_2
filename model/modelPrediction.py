# Import Dependencies
import joblib


def  predLogisticRegression(X_test, y_test):
    model = joblib.load("model/logistic.pkl")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return (y_pred, y_prob, y_test)

def  predKnnClassifier(X_test, y_test):
    model = joblib.load("model/knn.pkl")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return (y_pred, y_prob, y_test)

def  predDecisionTree(X_test, y_test):
    model = joblib.load("model/tree.pkl")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return (y_pred, y_prob, y_test)

def  predNaiveBayesClassifier(X_test, y_test):
    model = joblib.load("model/nb.pkl")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return (y_pred, y_prob, y_test)

def  predRandomForest(X_test, y_test):
    model = joblib.load("model/rf.pkl")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return (y_pred, y_prob, y_test)

def  predXGBoost(X_test, y_test):
    model = joblib.load("model/xgb.pkl")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return (y_pred, y_prob, y_test)