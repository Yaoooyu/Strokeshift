#!/usr/bin/env python3
"""RQ3 probability fusion: strict A_train OOF fitting, fixed RQ2 evaluation predictions."""
from pathlib import Path
import json, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, recall_score, confusion_matrix, brier_score_loss
ROOT=Path(__file__).resolve().parents[4]
RQ2=ROOT/'vlbd_benchmark'/'analysis'/'RQ2'/'predictions'
OUT=ROOT/'vlbd_benchmark'/'analysis'/'RQ3'
(OUT/'probability_fusion'/'predictions').mkdir(parents=True,exist_ok=True)
(OUT/'tables').mkdir(parents=True,exist_ok=True)
seeds=[20260925+i for i in range(5)]
cts=['SimpleCNN','ResNet18','DenseNet121','SwinTransformer']
ehrs=['LogisticRegression','RandomForest','XGBoost','MLP']
splits=['A_val','B_test','C_test']
def merge(a,b):
    x=a[['case_id','y_true','probability']].rename(columns={'probability':'p_ct'})
    y=b[['case_id','y_true','probability']].rename(columns={'probability':'p_ehr'})
    d=x.merge(y,on='case_id',validate='one_to_one',suffixes=('_ct','_ehr'))
    assert len(d)==len(a)==len(b) and (d.y_true_ct==d.y_true_ehr).all()
    return d.rename(columns={'y_true_ct':'y_true'}).drop(columns='y_true_ehr')
def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); result=0.
    for i in range(bins):
        z=(p>=edges[i]) & (p<(edges[i+1] if i<bins-1 else 1.000001))
        if z.any(): result+=z.mean()*abs(p[z].mean()-y[z].mean())
    return result
def metrics(y,p,t):
    q=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
    return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'f1':f1_score(y,q),'balanced_accuracy':balanced_accuracy_score(y,q),'sensitivity':recall_score(y,q),'specificity':tn/(tn+fp),'brier':brier_score_loss(y,p),'ece':ece(y,p)}
def threshold(y,p):
    grid=np.linspace(.05,.95,181)
    return float(max(grid,key=lambda t:f1_score(y,p>=t)))
def read_oof(kind,name):
    return pd.read_csv(OUT/'oof'/f'{kind}_{name}_A_train_oof.csv')
params=[]; results=[]
for ct,ehr in itertools.product(cts,ehrs):
    o=merge(read_oof('ct',ct),read_oof('ehr',ehr))
    y=o.y_true.to_numpy(); pc=o.p_ct.to_numpy(); pe=o.p_ehr.to_numpy()
    candidates=np.linspace(0,1,101)
    scores=np.array([roc_auc_score(y,w*pc+(1-w)*pe) for w in candidates])
    best=np.flatnonzero(scores==scores.max())
    w=float(candidates[best[np.argmin(abs(candidates[best]-.5))]])
    stack=LogisticRegression(C=1,max_iter=3000,random_state=20260925).fit(o[['p_ct','p_ehr']],y)
    params.append({'ct_model':ct,'ehr_model':ehr,'weight_ct':w,'oof_weighted_auroc':float(scores.max()),'stack_intercept':float(stack.intercept_[0]),'stack_coef_ct':float(stack.coef_[0,0]),'stack_coef_ehr':float(stack.coef_[0,1]),'oof_n':len(o)})
    for seed,sp in itertools.product(seeds,splits):
        a=pd.read_csv(RQ2/f'ct_{ct}_seed{seed}_{sp}.csv')
        b=pd.read_csv(RQ2/f'ehr_{ehr}_seed{seed}_{sp}.csv')
        d=merge(a,b); yy=d.y_true.to_numpy()
        probs={'equal_average':.5*d.p_ct.to_numpy()+.5*d.p_ehr.to_numpy(),
               'oof_weighted_average':w*d.p_ct.to_numpy()+(1-w)*d.p_ehr.to_numpy(),
               'oof_logistic_stacking':stack.predict_proba(d[['p_ct','p_ehr']])[:,1]}
        for method,pred in probs.items():
            # threshold choice occurs only on the current seed A_val predictions
            if sp=='A_val': t=threshold(yy,pred)
            else:
                av=merge(pd.read_csv(RQ2/f'ct_{ct}_seed{seed}_A_val.csv'),pd.read_csv(RQ2/f'ehr_{ehr}_seed{seed}_A_val.csv'))
                avp={'equal_average':.5*av.p_ct.to_numpy()+.5*av.p_ehr.to_numpy(),
                     'oof_weighted_average':w*av.p_ct.to_numpy()+(1-w)*av.p_ehr.to_numpy(),
                     'oof_logistic_stacking':stack.predict_proba(av[['p_ct','p_ehr']])[:,1]}[method]
                t=threshold(av.y_true.to_numpy(),avp)
            out=pd.DataFrame({'case_id':d.case_id,'y_true':yy,'probability':pred,'threshold':t})
            out.to_csv(OUT/'probability_fusion'/'predictions'/f'{method}_ct-{ct}_ehr-{ehr}_seed{seed}_{sp}.csv',index=False)
            for metric,value in metrics(yy,pred,t).items():
                results.append({'ct_model':ct,'ehr_model':ehr,'method':method,'seed':seed,'split':sp,'metric':metric,'value':value,'threshold':t})
    print('done',ct,ehr,flush=True)
pd.DataFrame(params).to_csv(OUT/'tables'/'rq3_probability_fusion_parameters.csv',index=False)
r=pd.DataFrame(results); r.to_csv(OUT/'rq3_probability_per_seed_results.csv',index=False)
r.groupby(['ct_model','ehr_model','method','split','metric']).agg(mean=('value','mean'),sd=('value','std'),n_seeds=('seed','nunique')).reset_index().to_csv(OUT/'tables'/'rq3_probability_summary.csv',index=False)
json.dump({'oof_fitting_data':'A_train only; 5-fold strict OOF','external_data_use':'B/C evaluation only','threshold':'maximal F1 on each seed A_val only','methods':['equal_average','oof_weighted_average','oof_logistic_stacking'],'pairs':16,'seeds':seeds},open(OUT/'configs'/'probability_fusion_protocol.json','w'),indent=2)
