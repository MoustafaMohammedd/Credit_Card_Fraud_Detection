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

def data_prepare (df_train,df_val,degree=1,over_or_under="under",under_res_factor=1,over_res_factor=1,pca_or_not="not",n_pca=7):

    y_train=df_train["Class"]
    x_train=df_train.drop(columns=["Class"])

    len_1=y_train[y_train==1].shape[0]
    len_0=y_train[y_train==0].shape[0]

    y_val=df_val["Class"]
    x_val=df_val.drop(columns=["Class"])

    if over_or_under=="under":
        under_res=RandomUnderSampler(sampling_strategy={0:round(len_1*under_res_factor),1:len_1},random_state=42)
        x_res,y_res=under_res.fit_resample(x_train,y_train)

    elif over_or_under=="over" :
        over_res=RandomOverSampler(sampling_strategy={0:len_0,1:round(len_0*over_res_factor)},random_state=42)
        x_res,y_res=over_res.fit_resample(x_train,y_train)

    elif over_or_under=="smote":
        over_res_smote=SMOTE(k_neighbors=5,random_state=42)
        x_res,y_res=over_res_smote.fit_resample(x_train,y_train)

    else:
        x_res,y_res=x_train,y_train

    if pca_or_not=="pca":
        pipeline_pre=Pipeline(steps=[("scaler",StandardScaler()),
                                     ("poly",PolynomialFeatures(degree=degree,include_bias=True)),
                                     ("pca",PCA(n_components=n_pca,random_state=42))
                                    ])
        x_train=pipeline_pre.fit_transform(x_res)
        x_val=pipeline_pre.transform(x_val)

        y_train=y_res
    else:

        pipeline_pre=Pipeline(steps=[("scaler",StandardScaler()),
                                     ("poly",PolynomialFeatures(degree=degree,include_bias=True))

                                    ])
        x_train=pipeline_pre.fit_transform(x_res)
        x_val=pipeline_pre.transform(x_val)

        y_train=y_res

    return x_train,y_train,x_val,y_val,pipeline_pre

x_train,y_train,x_val,y_val,pipeline_pre=data_prepare (df_train,df_val,degree=2,over_or_under="under",under_res_factor=1,over_res_factor=1,pca_or_not="not",n_pca=7)

