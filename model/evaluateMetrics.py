# Import Dependencies
from sklearn.metrics import *
from logisticRegression import logisticRegression
from decisionTreeClassifier import decisionTree
from knnClassifier import knnClassifier
from naiveBayesClassifier import naiveBayesClassifier
from randomForest import randomForest
from xgBoost import xgBoost

# Logicstic Regression Metrics Evaluation
y_test, pred = logisticRegression()
print("Metrics for Logistic Regression")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,pred))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# Decision Tree Metrics Evaluation
y_test, pred = decisionTree()
print("Metrics for Decision Tree Classifier")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,pred))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# KNN Classifier Metrics Evaluation
y_test, pred = knnClassifier()
print("Metrics for KNN Classifier")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,pred))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# Naive Bayes Classifier Metrics Evaluation
y_test, pred = naiveBayesClassifier()
print("Metrics for Naive Bayes Classifier")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,pred))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# Random Forest Metrics Evaluation
y_test, pred = randomForest()
print("Metrics for random Forest")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,pred))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

# XG Boost Metrics Evaluation
y_test, pred = xgBoost()
print("Metrics for XG Boost")
print("===========================================")
print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,pred))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

