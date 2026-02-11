# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import joblib

# Random Forest
X_train, X_test, _, _, y_train, y_test = dataPreProcFeatureEng()

model = RandomForestClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,prob))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

joblib.dump(model,"model/rf.pkl")