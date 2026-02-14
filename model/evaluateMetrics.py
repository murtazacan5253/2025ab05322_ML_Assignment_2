# Import Dependencies
from sklearn.metrics import *
#from modelPrediction import modelPredicts

def getMetrics(model,X_test,y_test):
    # Model Prediction
    #pred, prob= modelPredicts(model, X_test)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    #return (y_pred, y_prob)
    # Model Metric Score
    metrics = {
        "Accuracy:": accuracy_score(y_test,y_pred),
        "AUC:": roc_auc_score(y_test,y_prob),
        "Precision:": precision_score(y_test,y_pred),
        "Recall:": recall_score(y_test,y_pred),
        "F1:": f1_score(y_test,y_pred),
        "MCC:": matthews_corrcoef(y_test,y_pred)
    }

    return metrics,y_pred,y_prob

    

