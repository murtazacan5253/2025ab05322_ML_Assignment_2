# Import Dependencies
from sklearn.metrics import *
from modelPrediction import *


# Logicstic Regression Metrics Evaluation
pred, prob, y_test = predLogisticRegression()
print("Metrics for Logistic Regression")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,prob))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# Decision Tree Metrics Evaluation
y_test, pred, y_test = predDecisionTree()
print("Metrics for Decision Tree Classifier")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,prob))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# KNN Classifier Metrics Evaluation
y_test, pred, y_test = predKnnClassifier()
print("Metrics for KNN Classifier")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,prob))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# Naive Bayes Classifier Metrics Evaluation
y_test, pred, y_test = predNaiveBayesClassifier()
print("Metrics for Naive Bayes Classifier")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,prob))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# Random Forest Metrics Evaluation
y_test, pred, y_test = predRandomForest()
print("Metrics for random Forest")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,prob))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# XG Boost Metrics Evaluation
y_test, pred, y_test = predXGBoost()
print("Metrics for XG Boost")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,prob))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

