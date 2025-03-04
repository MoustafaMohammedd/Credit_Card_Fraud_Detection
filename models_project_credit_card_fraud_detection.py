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

"""# LogisticRegression model with & without resampling"""

def lr_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,cw=None,pca_or_not="not",n_pca=7):


    x_train,y_train,x_val,y_val,pipeline_pre=data_prepare (df_train,df_val,degree=degree,over_or_under=over_or_under,under_res_factor=under_res_factor,over_res_factor=over_res_factor,pca_or_not=pca_or_not,n_pca=n_pca)

    model, x_train,y_train,y_train_predict_prop,y_train_predict,x_val,y_val,y_val_predict_prop,y_val_predict,f1_train_postive,f1_val_postive,f1_macro_avg_train_postive,f1_macro_avg_val_postive=train_val_models(LogisticRegression(class_weight=cw,random_state=42),x_train,y_train,x_val,y_val)

    pt,rt,tht,pv,rv,thv,pr_auc_train,pr_auc_val=pr_curve(y_train,y_train_predict_prop,y_val,y_val_predict_prop) #(y_train,y_train_predict_prop,y_val,y_val_predict_prop)

    hist_lr={"model":model,
            "pipeline_pre":pipeline_pre,
             "over_or_under":over_or_under,
             "degree":degree ,
             "under_res_factor":under_res_factor,
             "over_res_factor":over_res_factor,
             "class_weight":cw,
             "pca_or_not":pca_or_not,
             "n_pca":n_pca,
            "pr_auc_train":pr_auc_train,
            "pr_auc_val":pr_auc_val,
            "f1_train_postive":f1_train_postive,
            "f1_val_postive":f1_val_postive,
            "f1_macro_avg_train_postive":f1_macro_avg_train_postive,
            "f1_macro_avg_val_postive":f1_macro_avg_val_postive}

    return hist_lr

history_lg={}

history_lg[1]=lr_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,cw=None)
history_lg[2]=lr_model(df_train,df_val,degree=2,over_or_under="none",under_res_factor=1,over_res_factor=1,cw=None)
history_lg[3]=lr_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,cw={0:1,1:.06})
history_lg[4]=lr_model(df_train,df_val,degree=2,over_or_under="none",under_res_factor=1,over_res_factor=1,cw={0:1,1:.06})

history_lg[5]=lr_model(df_train,df_val,degree=1,over_or_under="under",under_res_factor=1,over_res_factor=1,cw=None)
history_lg[6]=lr_model(df_train,df_val,degree=2,over_or_under="under",under_res_factor=1,over_res_factor=1,cw=None)
history_lg[7]=lr_model(df_train,df_val,degree=1,over_or_under="under",under_res_factor=2,over_res_factor=1,cw=None)
history_lg[8]=lr_model(df_train,df_val,degree=2,over_or_under="under",under_res_factor=2,over_res_factor=1,cw=None)

history_lg[9]=lr_model(df_train,df_val,degree=1,over_or_under="over",under_res_factor=2,over_res_factor=1,cw=None)
history_lg[10]=lr_model(df_train,df_val,degree=2,over_or_under="over",under_res_factor=2,over_res_factor=1,cw=None)
history_lg[11]=lr_model(df_train,df_val,degree=1,over_or_under="over",under_res_factor=2,over_res_factor=0.5,cw=None)
history_lg[12]=lr_model(df_train,df_val,degree=2,over_or_under="over",under_res_factor=2,over_res_factor=0.5,cw=None)

history_lg[13]=lr_model(df_train,df_val,degree=1,over_or_under="smote",under_res_factor=2,over_res_factor=0.5,cw=None)
history_lg[14]=lr_model(df_train,df_val,degree=2,over_or_under="smote",under_res_factor=2,over_res_factor=0.5,cw=None)

history_lg

history_lg

for k,v in history_lg.items():
    print(k,v["f1_train_postive"],v["f1_val_postive"],v["f1_macro_avg_train_postive"],v["f1_macro_avg_val_postive"])



"""# RandomForestClassifier model with & without resampling"""



def rf_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,max_depth=3,n_estimators=6,pca_or_not="not",n_pca=7):


    x_train,y_train,x_val,y_val,pipeline_pre=data_prepare (df_train,df_val,degree=degree,over_or_under=over_or_under,under_res_factor=under_res_factor,over_res_factor=over_res_factor,pca_or_not=pca_or_not,n_pca=n_pca)

    model, x_train,y_train,y_train_predict_prop,y_train_predict,x_val,y_val,y_val_predict_prop,y_val_predict,f1_train_postive,f1_val_postive,f1_macro_avg_train_postive,f1_macro_avg_val_postive=train_val_models(RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42),x_train,y_train,x_val,y_val)

    pt,rt,tht,pv,rv,thv,pr_auc_train,pr_auc_val=pr_curve(y_train,y_train_predict_prop,y_val,y_val_predict_prop)

    hist_rf={"model":model,
            "pipeline_pre":pipeline_pre,
             "over_or_under":over_or_under,
             "degree":degree ,
             "under_res_factor":under_res_factor,
             "over_res_factor":over_res_factor,
             "max_depth":max_depth,
             "n_estimators":n_estimators,
             "pca_or_not":pca_or_not,
             "n_pca":n_pca,
            "pr_auc_train":pr_auc_train,
            "pr_auc_val":pr_auc_val,
            "f1_train_postive":f1_train_postive,
            "f1_val_postive":f1_val_postive,
            "f1_macro_avg_train_postive":f1_macro_avg_train_postive,
            "f1_macro_avg_val_postive":f1_macro_avg_val_postive}

    return hist_rf

history_rf={}

history_rf[1]=rf_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,max_depth=3,n_estimators=6)
history_rf[2]=rf_model(df_train,df_val,degree=2,over_or_under="none",under_res_factor=1,over_res_factor=1,max_depth=6,n_estimators=10)
history_rf[3]=rf_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,max_depth=10,n_estimators=20)
history_rf[4]=rf_model(df_train,df_val,degree=2,over_or_under="none",under_res_factor=1,over_res_factor=1,max_depth=20,n_estimators=20)

history_rf[5]=rf_model(df_train,df_val,degree=1,over_or_under="under",under_res_factor=1,over_res_factor=1,max_depth=3,n_estimators=6)
history_rf[6]=rf_model(df_train,df_val,degree=2,over_or_under="under",under_res_factor=1,over_res_factor=1,max_depth=6,n_estimators=10)
history_rf[7]=rf_model(df_train,df_val,degree=1,over_or_under="under",under_res_factor=2,over_res_factor=1,max_depth=10,n_estimators=20)
history_rf[8]=rf_model(df_train,df_val,degree=2,over_or_under="under",under_res_factor=2,over_res_factor=1,max_depth=20,n_estimators=20)

history_rf[9]=rf_model(df_train,df_val,degree=1,over_or_under="over",under_res_factor=2,over_res_factor=1,max_depth=3,n_estimators=6)
history_rf[10]=rf_model(df_train,df_val,degree=2,over_or_under="over",under_res_factor=2,over_res_factor=1,max_depth=6,n_estimators=7)
history_rf[11]=rf_model(df_train,df_val,degree=1,over_or_under="over",under_res_factor=2,over_res_factor=0.5,max_depth=10,n_estimators=10)
history_rf[12]=rf_model(df_train,df_val,degree=2,over_or_under="over",under_res_factor=2,over_res_factor=0.5,max_depth=30,n_estimators=20)

history_rf[13]=rf_model(df_train,df_val,degree=1,over_or_under="smote",under_res_factor=2,over_res_factor=0.5,max_depth=3,n_estimators=6)
history_rf[14]=rf_model(df_train,df_val,degree=2,over_or_under="smote",under_res_factor=2,over_res_factor=0.5,max_depth=15,n_estimators=20)

history_rf

"""# SVC model with & without resampling"""



def svc_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,kernel='rbf',pca_or_not="not",n_pca=7):


    x_train,y_train,x_val,y_val,pipeline_pre=data_prepare (df_train,df_val,degree=degree,over_or_under=over_or_under,under_res_factor=under_res_factor,over_res_factor=over_res_factor,pca_or_not=pca_or_not,n_pca=n_pca)

    model, x_train,y_train,y_train_predict_prop,y_train_predict,x_val,y_val,y_val_predict_prop,y_val_predict,f1_train_postive,f1_val_postive,f1_macro_avg_train_postive,f1_macro_avg_val_postive=train_val_models(SVC(kernel=kernel,probability=True,random_state=42),x_train,y_train,x_val,y_val)

    pt,rt,tht,pv,rv,thv,pr_auc_train,pr_auc_val=pr_curve(y_train,y_train_predict_prop,y_val,y_val_predict_prop)

    hist_svc={"model":model,
            "pipeline_pre":pipeline_pre,
             "over_or_under":over_or_under,
             "degree":degree ,
              "kernel":kernel,
             "pca_or_not":pca_or_not,
             "n_pca":n_pca,
             "under_res_factor":under_res_factor,
             "over_res_factor":over_res_factor,
            "pr_auc_train":pr_auc_train,
            "pr_auc_val":pr_auc_val,
            "f1_train_postive":f1_train_postive,
            "f1_val_postive":f1_val_postive,
            "f1_macro_avg_train_postive":f1_macro_avg_train_postive,
            "f1_macro_avg_val_postive":f1_macro_avg_val_postive}

    return hist_svc

history_svc={}

history_svc[1]=svc_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,kernel='rbf')
history_svc[2]=svc_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,kernel="sigmoid")
history_svc[3]=svc_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,kernel='poly')


# history_svc[5]=svc_model(df_train,df_val,degree=1,over_or_under="under",under_res_factor=1,over_res_factor=1,kernel='rbf')
# history_svc[6]=svc_model(df_train,df_val,degree=1,over_or_under="under",under_res_factor=1,over_res_factor=1,kernel="sigmoid")
# history_svc[7]=svc_model(df_train,df_val,degree=1,over_or_under="over",under_res_factor=1,over_res_factor=1,kernel="sigmoid")
# history_svc[8]=svc_model(df_train,df_val,degree=1,over_or_under="smote",under_res_factor=1,over_res_factor=1,kernel="sigmoid")

# history_svc[9]=svc_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,kernel='poly')
# history_svc[10]=svc_model(df_train,df_val,degree=1,over_or_under="under",under_res_factor=1,over_res_factor=1,kernel='poly')
# history_svc[11]=svc_model(df_train,df_val,degree=1,over_or_under="over",under_res_factor=1,over_res_factor=1,kernel='poly')
# history_svc[12]=svc_model(df_train,df_val,degree=1,over_or_under="smote",under_res_factor=1,over_res_factor=1,kernel='poly')

history_svc



"""# One class svm"""

f=df_train["Class"]==0
df_train_not_fruad=df_train[f]

x_train,y_train,x_val,y_val,pipeline_pre=data_prepare(df_train_not_fruad,df_val,degree=1,over_or_under="none",under_res_factor=2,over_res_factor=1,pca_or_not="not",n_pca=7)

ocsvm = OneClassSVM()
ocsvm.fit(x_train)

y_val_predict = ocsvm.predict(x_val)

y_val_predict_binary = [1 if pred == -1 else 0 for pred in y_val_predict]

val_report = classification_report(y_val, y_val_predict_binary)
confusion_val = confusion_matrix(y_val, y_val_predict_binary)

print("Classification Report:")
print(val_report)

print("Confusion Matrix:")
print(confusion_val)



"""# XGBC model with & without resampling"""



def xgbc_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,pca_or_not="not",n_pca=7):


    x_train,y_train,x_val,y_val,pipeline_pre=data_prepare (df_train,df_val,degree=degree,over_or_under=over_or_under,under_res_factor=under_res_factor,over_res_factor=over_res_factor,pca_or_not=pca_or_not,n_pca=n_pca)

    model, x_train,y_train,y_train_predict_prop,y_train_predict,x_val,y_val,y_val_predict_prop,y_val_predict,f1_train_postive,f1_val_postive,f1_macro_avg_train_postive,f1_macro_avg_val_postive=train_val_models(xgbc(random_state=42),x_train,y_train,x_val,y_val)

    pt,rt,tht,pv,rv,thv,pr_auc_train,pr_auc_val=pr_curve(y_train,y_train_predict_prop,y_val,y_val_predict_prop)

    hist_xgbc={"model":model,
            "pipeline_pre":pipeline_pre,
             "over_or_under":over_or_under,
             "degree":degree ,
             "pca_or_not":pca_or_not,
             "n_pca":n_pca,
             "under_res_factor":under_res_factor,
             "over_res_factor":over_res_factor,
            "pr_auc_train":pr_auc_train,
            "pr_auc_val":pr_auc_val,
            "f1_train_postive":f1_train_postive,
            "f1_val_postive":f1_val_postive,
            "f1_macro_avg_train_postive":f1_macro_avg_train_postive,
            "f1_macro_avg_val_postive":f1_macro_avg_val_postive}

    return hist_xgbc

history_xgbc={}

history_xgbc[1]=xgbc_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1)
history_xgbc[2]=xgbc_model(df_train,df_val,degree=1,over_or_under="under",under_res_factor=1,over_res_factor=1)
history_xgbc[3]=xgbc_model(df_train,df_val,degree=1,over_or_under="over",under_res_factor=1,over_res_factor=1)
history_xgbc[4]=xgbc_model(df_train,df_val,degree=1,over_or_under="smote",under_res_factor=1,over_res_factor=1)

history_xgbc



"""# VotingClassifier for best models"""

x_train,y_train,x_val,y_val,pipeline_pre=data_prepare(df_train,df_val,degree=1,over_or_under="none",under_res_factor=2,over_res_factor=1,pca_or_not="not",n_pca=7)

model1=history_lg[2]['model']  #history_knn[2]['model']

model2=history_rf[3]["model"]

voting_clf = VotingClassifier(
    estimators=[('lg', model1),
                ('rf', model2)
               ],
    voting='soft' ,weights=[1.5,2]
)

voting_clf.fit(x_train, y_train)

y_val_pred_voting = voting_clf.predict(x_val)

y_val_pred_voting_prop=voting_clf.predict_proba(x_val) [:,1]

val_report = classification_report(y_val, y_val_pred_voting)
confusion_val = confusion_matrix(y_val, y_val_pred_voting)

print("Classification Report:")
print(val_report)

print("Confusion Matrix:")
print(confusion_val)

pt,rt,tht=precision_recall_curve(y_val,y_val_pred_voting_prop)

pr_auc_val=auc(rt,pt)

print(f"pr_auc_val= {pr_auc_val}")

plt.plot(tht,pt[:-1],label="p_val")
plt.plot(tht,rt[:-1],label="r_val")
plt.title("pr_curve_val")
plt.xlabel("thresholds")
plt.legend()
plt.show()





"""# KNN"""



def knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=5,pca_or_not="not",n_pca=7):


    x_train,y_train,x_val,y_val,pipeline_pre=data_prepare (df_train,df_val,degree=degree,over_or_under=over_or_under,under_res_factor=under_res_factor,over_res_factor=over_res_factor,pca_or_not=pca_or_not,n_pca=n_pca)

    model, x_train,y_train,y_train_predict_prop,y_train_predict,x_val,y_val,y_val_predict_prop,y_val_predict,f1_train_postive,f1_val_postive,f1_macro_avg_train_postive,f1_macro_avg_val_postive=train_val_models(KNeighborsClassifier(n_neighbors=k),x_train,y_train,x_val,y_val)

    pt,rt,tht,pv,rv,thv,pr_auc_train,pr_auc_val=pr_curve(y_train,y_train_predict_prop,y_val,y_val_predict_prop)

    hist_knn={"model":model,
            "pipeline_pre":pipeline_pre,
             "over_or_under":over_or_under,
             "degree":degree ,
              "n_neighbors":k,
             "pca_or_not":pca_or_not,
             "n_pca":n_pca,
             "under_res_factor":under_res_factor,
             "over_res_factor":over_res_factor,
            "pr_auc_train":pr_auc_train,
            "pr_auc_val":pr_auc_val,
            "f1_train_postive":f1_train_postive,
            "f1_val_postive":f1_val_postive,
            "f1_macro_avg_train_postive":f1_macro_avg_train_postive,
            "f1_macro_avg_val_postive":f1_macro_avg_val_postive}

    return hist_knn

history_knn={}

history_knn[1]=knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=3)
history_knn[2]=knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=5)
history_knn[3]=knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=7)

history_knn[4]=knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=3,pca_or_not="pca",n_pca=7)
history_knn[5]=knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=5,pca_or_not="pca",n_pca=7)
history_knn[6]=knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=3,pca_or_not="pca",n_pca=8)
history_knn[7]=knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=5,pca_or_not="pca",n_pca=8)
history_knn[8]=knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=3,pca_or_not="pca",n_pca=6)
history_knn[9]=knn_model(df_train,df_val,degree=1,over_or_under="none",under_res_factor=1,over_res_factor=1,k=5,pca_or_not="pca",n_pca=6)

history_knn
