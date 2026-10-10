#!/usr/bin/env python3
"""Frozen-split, source-fitted EHR-only RQ2 baselines."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, recall_score, confusion_matrix, brier_score_loss
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'vlbd_benchmark'/'analysis'/'RQ4_EHR_ONLY_EXTENSION'; SEEDS=[20260925+i for i in range(5)]
for x in ['configs','predictions','per_seed_results','tables','figures','checkpoints','logs']: (OUT/x).mkdir(parents=True,exist_ok=True)
m=pd.read_csv(ROOT/'benchmark_manifest.csv'); e=pd.read_csv(ROOT/'metadata/ehr.csv')
splits={k:pd.read_csv(ROOT/'vlbd_benchmark'/'protocol'/'splits'/f'{k}.csv').case_id.tolist() for k in ['A_train','A_val','B_test','C_test']}
drop={'case_id','patient_id','center','potential_post_treatment_leakage','leakage_flagged_columns'}
features=[c for c in e if c not in drop]
X=e.set_index('case_id')[features].apply(pd.to_numeric,errors='coerce'); y=m.set_index('case_id').ht_label.astype(int)
def ece(y,p,bins=10):
 z=np.linspace(0,1,bins+1); ans=0
 for i in range(bins):
  q=(p>=z[i])&(p<(z[i+1] if i<bins-1 else 1.00001))
  if q.any(): ans+=q.mean()*abs(p[q].mean()-y[q].mean())
 return ans
def met(y,p,t):
 q=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel(); return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),f1=f1_score(y,q),balanced_accuracy=balanced_accuracy_score(y,q),sensitivity=recall_score(y,q),specificity=tn/(tn+fp),brier=brier_score_loss(y,p),ece=ece(y,p))
def boot(y,p,t,seed):
 r=np.random.default_rng(seed); out=[]
 for _ in range(2000):
  ix=r.integers(0,len(y),len(y));
  if len(np.unique(np.asarray(y)[ix]))==2: out.append(met(np.asarray(y)[ix],np.asarray(p)[ix],t))
 return {k:(np.quantile([x[k] for x in out],.025),np.quantile([x[k] for x in out],.975),len(out)) for k in out[0]}
rows=[]
for seed in SEEDS:
 tr,va=splits['A_train'],splits['A_val']; prep=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]); xx=prep.fit_transform(X.loc[tr]); xv={z:prep.transform(X.loc[splits[z]]) for z in ['A_val','B_test','C_test']}
 models={'LogisticRegression':LogisticRegression(C=1,max_iter=3000,random_state=seed),'RandomForest':RandomForestClassifier(n_estimators=500,min_samples_leaf=3,class_weight='balanced',random_state=seed,n_jobs=-1),'XGBoost':XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.8,eval_metric='logloss',random_state=seed,n_jobs=8),'MLP':MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-3,max_iter=500,early_stopping=True,validation_fraction=.15,random_state=seed)}
 for model_idx,(name,model) in enumerate(models.items()):
  model.fit(xx,y.loc[tr]); pv=model.predict_proba(xv['A_val'])[:,1]; grid=np.linspace(.05,.95,181); threshold=max(grid,key=lambda t:f1_score(y.loc[va],pv>=t))
  for s in ['A_val','B_test','C_test']:
   p=pv if s=='A_val' else model.predict_proba(xv[s])[:,1]; yy=y.loc[splits[s]].to_numpy(); d=met(yy,p,threshold); ci=boot(yy,p,threshold,seed+100*model_idx+['A_val','B_test','C_test'].index(s))
   pd.DataFrame({'case_id':splits[s],'y_true':yy,'probability':p,'threshold':threshold}).to_csv(OUT/'predictions'/f'ehr_{name}_seed{seed}_{s}.csv',index=False)
   for k,v in d.items(): rows.append(dict(modality='EHR',model=name,seed=seed,split=s,metric=k,value=v,ci_low=ci[k][0],ci_high=ci[k][1],bootstrap_valid=ci[k][2],threshold=threshold))
pd.DataFrame(rows).to_csv(OUT/'rq2_ehr_per_seed_results.csv',index=False)
pd.DataFrame(rows).groupby(['modality','model','split','metric']).agg(mean=('value','mean'),sd=('value','std'),ci_low=('ci_low','mean'),ci_high=('ci_high','mean'),n_seeds=('seed','nunique')).reset_index().to_csv(OUT/'tables'/'RQ2_2_ehr_performance.csv',index=False)
json.dump({'seeds':SEEDS,'features':features,'preprocessing':'median imputation + missingness indicators + standard scaling, fitted only on A_train','threshold':'maximal F1 on A_val only'},open(OUT/'configs'/'ehr_protocol.json','w'),indent=2)
