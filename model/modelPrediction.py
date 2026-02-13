# Import Dependencies


def  modelPredict(model, X_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return (y_pred, y_prob)