#!/usr/bin/env python3
"""Strict source-only five-fold OOF EHR probabilities for RQ3 fusion."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'vlbd_benchmark'/'analysis'/'RQ3'
SEED=20260925
m=pd.read_csv(ROOT/'benchmark_manifest.csv').set_index('case_id')
train=pd.read_csv(ROOT/'vlbd_benchmark'/'protocol'/'splits'/'A_train.csv').case_id.tolist()
e=pd.read_csv(ROOT/'metadata'/'ehr.csv')
drop={'case_id','patient_id','center','potential_post_treatment_leakage','leakage_flagged_columns'}
features=[c for c in e.columns if c not in drop]
X=e.set_index('case_id')[features].apply(pd.to_numeric,errors='coerce').loc[train]
y=m.loc[train,'ht_label'].astype(int)
assert len(train)==384 and X.index.tolist()==train
folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
allout={name:[] for name in ['LogisticRegression','RandomForest','XGBoost','MLP']}
for fold,(tr,te) in enumerate(folds.split(X,y),1):
    tr_ids=X.index[tr].tolist(); te_ids=X.index[te].tolist()
    prep=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())])
    xx=prep.fit_transform(X.loc[tr_ids]); xv=prep.transform(X.loc[te_ids])
    models={
      'LogisticRegression':LogisticRegression(C=1,max_iter=3000,random_state=SEED+fold),
      'RandomForest':RandomForestClassifier(n_estimators=500,min_samples_leaf=3,class_weight='balanced',random_state=SEED+fold,n_jobs=-1),
      'XGBoost':XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.8,eval_metric='logloss',random_state=SEED+fold,n_jobs=8),
      'MLP':MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-3,max_iter=500,early_stopping=True,validation_fraction=.15,random_state=SEED+fold)
    }
    for name,model in models.items():
        model.fit(xx,y.loc[tr_ids])
        allout[name].append(pd.DataFrame({'case_id':te_ids,'y_true':y.loc[te_ids].to_numpy(),'probability':model.predict_proba(xv)[:,1],'fold':fold}))
        print('done',name,'fold',fold,flush=True)
for name,parts in allout.items():
    d=pd.concat(parts,ignore_index=True).sort_values('case_id')
    assert len(d)==384 and d.case_id.nunique()==384 and set(d.case_id)==set(train)
    d.to_csv(OUT/'oof'/f'ehr_{name}_A_train_oof.csv',index=False)
json.dump({'source_split':'A_train only','n_cases':384,'folds':5,'splitter':'StratifiedKFold','random_state':SEED,'features':features},open(OUT/'configs'/'ehr_oof_protocol.json','w'),indent=2)
