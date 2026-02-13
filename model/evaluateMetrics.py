# Import Dependencies
from sklearn.metrics import *
from modelPrediction import modelPredict

def getMetrics (model,X_test,y_test):
    # Model Prediction
    pred, prob= modelPredict(model, X_test)

    # Model Metric Score
    metrics = {
        print("===========================================")
        print("Accuracy:", accuracy_score(y_test,pred))
        print("AUC:", roc_auc_score(y_test,prob))
        print("Precision:", precision_score(y_test,pred))
        print("Recall:", recall_score(y_test,pred))
        print("F1:", f1_score(y_test,pred))
        print("MCC:", matthews_corrcoef(y_test,pred))
    }

    return metrics

    

