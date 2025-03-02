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

def train_val_models(model,x_train,y_train,x_val,y_val):

    model.fit(x_train,y_train)

    y_train_predict=model.predict(x_train)
    y_train_predict_prop=model.predict_proba(x_train)[:,1]

    y_val_predict=model.predict(x_val)
    y_val_predict_prop=model.predict_proba(x_val)[:,1]

    confusion_train=confusion_matrix(y_train,y_train_predict)
    train_report=classification_report(y_train,y_train_predict, output_dict=True)

    val_report=classification_report(y_val,y_val_predict, output_dict=True)
    confusion_val=confusion_matrix(y_val,y_val_predict)

    f1_train_postive=train_report['1']['f1-score']
    f1_val_postive=val_report['1']['f1-score']

    f1_macro_avg_train_postive=train_report['macro avg']['f1-score']

    f1_macro_avg_val_postive=val_report['macro avg']['f1-score']


    train_report=classification_report(y_train,y_train_predict)

    val_report=classification_report(y_val,y_val_predict)

    print(f"{model}:::\n train\n\n {confusion_train}\n\n {train_report} ")

    print(f"::::::\n     val\n\n {confusion_val}\n\n {val_report} ")

    return model, x_train,y_train,y_train_predict_prop,y_train_predict,x_val,y_val,y_val_predict_prop,y_val_predict,f1_train_postive,f1_val_postive,f1_macro_avg_train_postive,f1_macro_avg_val_postive

def pr_curve(y_train,y_train_predict_prop,y_val,y_val_predict_prop):

    pt,rt,tht=precision_recall_curve(y_train,y_train_predict_prop)
    pv,rv,thv=precision_recall_curve(y_val,y_val_predict_prop)

    pr_auc_train=auc(rt,pt)
    pr_auc_val=auc(rv,pv)

    print(f"pr_auc_train= {pr_auc_train}")
    print(f"pr_auc_val= {pr_auc_val}")

    plt.plot(tht,pt[:-1],label="p_train")
    plt.plot(tht,rt[:-1],label="r_train")
    plt.title("pr_curve_train")
    plt.xlabel("thresholds")
    plt.legend()
    plt.show()

    plt.plot(thv,pv[:-1],label="p_val")
    plt.plot(thv,rv[:-1],label="r_val")
    plt.title("pr_curve_val")
    plt.xlabel("thresholds")
    plt.legend()
    plt.show()

    return pt,rt,tht,pv,rv,thv,pr_auc_train,pr_auc_val

