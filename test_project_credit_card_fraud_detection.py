!pip install imbalanced-learn==0.12.3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report ,confusion_matrix ,precision_recall_curve,auc
from sklearn.preprocessing import StandardScaler ,MinMaxScaler ,FunctionTransformer ,PolynomialFeatures
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.ensemble import RandomForestClassifier ,VotingClassifier
from sklearn.svm import SVC, OneClassSVM
from xgboost import XGBClassifier as xgbc
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA
import time

"""# Testing time"""



y_test=df_test["Class"]
x_test=df_test.drop(columns=["Class"])

x_test=history_rf[3]['pipeline_pre'].transform(x_test)

y_test_pred=history_rf[3]["model"].predict(x_test)
y_test_pred_prop=history_rf[3]["model"].predict_proba(x_test) [:,1]

test_report = classification_report(y_test, y_test_pred)
confusion_test = confusion_matrix(y_test, y_test_pred)

print("Classification Report:")
print(test_report)

print("Confusion Matrix:")
print(confusion_test)

pt,rt,tht=precision_recall_curve(y_test,y_test_pred_prop)

pr_auc_test=auc(rt,pt)

print(f"pr_auc_test= {pr_auc_test}")

plt.plot(tht,pt[:-1],label="p_test")
plt.plot(tht,rt[:-1],label="r_test")
plt.title("pr_curve_test")
plt.xlabel("thresholds")
plt.legend()
plt.show()



"""# Testing time"""

y_test=df_test["Class"]
x_test=df_test.drop(columns=["Class"])

x_test=history_knn[2]['pipeline_pre'].transform(x_test)

print(x_test.shape)

start=time.time()
y_test_pred=history_knn[2]["model"].predict(x_test)
end=time.time()

y_test_pred_prop=history_knn[2]["model"].predict_proba(x_test) [:,1]

test_report = classification_report(y_test, y_test_pred)
confusion_test = confusion_matrix(y_test, y_test_pred)

print(f"takes {round(end-start)} s")

print("Classification Report:")
print(test_report)

print("Confusion Matrix:")
print(confusion_test)

pt,rt,tht=precision_recall_curve(y_test,y_test_pred_prop)

pr_auc_test=auc(rt,pt)

print(f"pr_auc_test= {pr_auc_test}")

plt.plot(tht,pt[:-1],label="p_test")
plt.plot(tht,rt[:-1],label="r_test")
plt.title("pr_curve_test")
plt.xlabel("thresholds")
plt.legend()
plt.show()



y_test=df_test["Class"]
x_test=df_test.drop(columns=["Class"])

x_test=history_knn[1]['pipeline_pre'].transform(x_test)

print(x_test.shape)

start=time.time()
y_test_pred=history_knn[1]["model"].predict(x_test)
end=time.time()

y_test_pred_prop=history_knn[1]["model"].predict_proba(x_test) [:,1]

test_report = classification_report(y_test, y_test_pred)
confusion_test = confusion_matrix(y_test, y_test_pred)

print(f"takes {round(end-start)} s")
print()
print("Classification Report:")
print(test_report)

print("Confusion Matrix:")
print(confusion_test)

pt,rt,tht=precision_recall_curve(y_test,y_test_pred_prop)

pr_auc_test=auc(rt,pt)

print(f"pr_auc_test= {pr_auc_test}")

plt.plot(tht,pt[:-1],label="p_test")
plt.plot(tht,rt[:-1],label="r_test")
plt.title("pr_curve_test")
plt.xlabel("thresholds")
plt.legend()
plt.show()



y_test=df_test["Class"]
x_test=df_test.drop(columns=["Class"])

x_test=history_knn[4]['pipeline_pre'].transform(x_test)

print(x_test.shape)
  # 8 is the best till now and 7

start=time.time()
y_test_pred=history_knn[4]["model"].predict(x_test)
end=time.time()

y_test_pred_prop=history_knn[4]["model"].predict_proba(x_test) [:,1]

test_report = classification_report(y_test, y_test_pred)
confusion_test = confusion_matrix(y_test, y_test_pred)

print(f"takes {round(end-start)} s")
print()

print("Classification Report:")
print(test_report)

print("Confusion Matrix:")
print(confusion_test)
print()

pt,rt,tht=precision_recall_curve(y_test,y_test_pred_prop)

pr_auc_test=auc(rt,pt)

print(f"pr_auc_test= {pr_auc_test}")

plt.plot(tht,pt[:-1],label="p_test")
plt.plot(tht,rt[:-1],label="r_test")
plt.title("pr_curve_test")
plt.xlabel("thresholds")
plt.legend()
plt.show()



y_test=df_test["Class"]
x_test=df_test.drop(columns=["Class"])

x_test=history_knn[5]['pipeline_pre'].transform(x_test)

print(x_test.shape)
  # 8 is the best till now and 7

start=time.time()
y_test_pred=history_knn[5]["model"].predict(x_test)
end=time.time()

y_test_pred_prop=history_knn[5]["model"].predict_proba(x_test) [:,1]

test_report = classification_report(y_test, y_test_pred)
confusion_test = confusion_matrix(y_test, y_test_pred)

print(f"takes {round(end-start)} s")
print()

print("Classification Report:")
print(test_report)

print("Confusion Matrix:")
print(confusion_test)
print()

pt,rt,tht=precision_recall_curve(y_test,y_test_pred_prop)

pr_auc_test=auc(rt,pt)

print(f"pr_auc_test= {pr_auc_test}")

plt.plot(tht,pt[:-1],label="p_test")
plt.plot(tht,rt[:-1],label="r_test")
plt.title("pr_curve_test")
plt.xlabel("thresholds")
plt.legend()
plt.show()

