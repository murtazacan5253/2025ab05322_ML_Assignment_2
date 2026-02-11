# Import Dependencies
from dataPreProcFeatureEng import dataPreProcFeatureEng
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import joblib

_, _, Xs_train, Xs_test, y_train, y_test = dataPreProcFeatureEng()

model = KNeighborsClassifier(n_neighbors=5)
model.fit(Xs_train,y_train)

pred = model.predict(Xs_test)
prob = model.predict_proba(Xs_test)[:,1]

print("Accuracy:", accuracy_score(y_test,pred))
print("AUC:", roc_auc_score(y_test,prob))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("F1:", f1_score(y_test,pred))
print("MCC:", matthews_corrcoef(y_test,pred))

joblib.dump(model,"model/knn.pkl")