import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,confusion_matrix,brier_score_loss
R=Path(__file__).resolve().parents[3];B=R/'vlbd_benchmark';O=B/'analysis/RQ4_EHR_ONLY_EXTENSION';Q=B/'analysis/RQ4';S=range(20260925,20260930)
m=pd.read_csv(R/'benchmark_manifest.csv').set_index('case_id');sp={k:pd.read_csv(B/'protocol/splits'/f'{k}.csv').case_id.tolist() for k in ['A_train','A_val','B_test','C_test']};e=pd.read_csv(R/'metadata/ehr.csv').set_index('case_id');co=json.load(open(B/'analysis/RQ2/configs/ehr_protocol.json'))['features'];x=e[co].apply(pd.to_numeric,errors='coerce');y=m.ht_label.astype(int)
def M(n,s):return {'LogisticRegression':LogisticRegression(C=1,max_iter=3000,random_state=s),'RandomForest':RandomForestClassifier(n_estimators=500,min_samples_leaf=3,class_weight='balanced',random_state=s,n_jobs=-1),'XGBoost':XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.8,eval_metric='logloss',random_state=s,n_jobs=8),'MLP':MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-3,max_iter=500,early_stopping=True,validation_fraction=.15,random_state=s)}[n]
def go(a,p,t):
 q=p>=t;tn,fp,fn,tp=confusion_matrix(a,q,labels=[0,1]).ravel();return {'auroc':roc_auc_score(a,p),'auprc':average_precision_score(a,p),'brier':brier_score_loss(a,p),'f1':f1_score(a,q),'balanced_accuracy':balanced_accuracy_score(a,q),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp)}
def ad(o,e,c,r,z,n,s,k,a,p,t):
 for q,v in go(a,p,t).items():o.append([e,c,str(r),z,n,s,k,q,v,t])
a=[];d=[]
for s in S:
 p=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]);tr=p.fit_transform(x.loc[sp['A_train']])
 for n in ['LogisticRegression','RandomForest','XGBoost','MLP']:
  h=M(n,s);h.fit(tr,y.loc[sp['A_train']]);v=h.predict_proba(p.transform(x.loc[sp['A_val']]))[:,1];t=max(np.linspace(.05,.95,181),key=lambda z:f1_score(y.loc[sp['A_val']],v>=z))
  for c in ['elementwise','featurewise']:
   for r in [0,.1,.3,.5,.7]:
    for z in ([0] if r==0 else [4101,4102,4103,4104,4105]):
     mk=pd.read_csv(Q/'corruption_masks'/f'{c}_rate{r}_seed{z}.csv').set_index('case_id')
     for k in ['A_val','B_test','C_test']:
      u=x.loc[sp[k]].mask(mk.loc[sp[k],co].astype(bool));ad(a,'random_ehr',c,r,z,n,s,k,y.loc[sp[k]].to_numpy(),h.predict_proba(p.transform(u))[:,1],t)
  for c in ['B','C']:
   for z in [4101,4102,4103,4104,4105]:
    mk=pd.read_csv(Q/'corruption_masks'/f'{c}_pattern_rateempirical_seed{z}.csv').set_index('case_id');u=x.loc[sp['A_val']].mask(mk.loc[sp['A_val'],co].astype(bool));ad(d,'center_structured',c+'_pattern','empirical',z,n,s,'A_val',y.loc[sp['A_val']].to_numpy(),h.predict_proba(p.transform(u))[:,1],t)
  print('done',n,s,flush=True)
C=['experiment','condition','rate','corruption_seed','model','model_seed','split','metric','value','threshold']
for z,name in [(a,'ehr_only_random'),(d,'ehr_only_center')]:
 q=pd.DataFrame(z,columns=C);q.to_csv(O/'tables'/f'{name}_raw.csv',index=False);q.groupby(['experiment','condition','rate','model','split','metric']).value.agg(['mean','std','count']).reset_index().rename(columns={'std':'sd','count':'n'}).to_csv(O/'tables'/f'{name}_missingness.csv',index=False)
