#!/usr/bin/env python3
"""RQ3 bootstrap CIs and paired comparisons; entities selected on A_val only."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,recall_score,confusion_matrix,brier_score_loss
ROOT=Path(__file__).resolve().parents[4]; RQ3=ROOT/'vlbd_benchmark'/'analysis'/'RQ3'; TAB=RQ3/'tables'; N=2000
u=pd.read_csv(RQ3/'rq3_unified_per_seed_results.csv')
s=pd.read_csv(TAB/'rq3_unified_summary.csv')
base=s[(s.group.isin(['base_ct','base_ehr']))&(s.split=='A_val')&(s.metric=='auroc')].sort_values('mean',ascending=False).iloc[0]
prob=s[(s.group.str.startswith('probability_'))&(s.split=='A_val')&(s.metric=='auroc')].sort_values('mean',ascending=False).iloc[0]
rep=s[(s.group=='representation')&(s.split=='A_val')&(s.metric=='auroc')].sort_values('mean',ascending=False).iloc[0]
entities=[('source_selected_base',base.group,base.entity),('source_selected_probability_fusion',prob.group,prob.entity),('source_selected_representation_fusion',rep.group,rep.entity)]
hi={'auroc','auprc','f1','balanced_accuracy','sensitivity','specificity'}
def ece(y,p):
 ans=0.;edges=np.linspace(0,1,11)
 for i in range(10):
  q=(p>=edges[i])&(p<(edges[i+1] if i<9 else 1.00001))
  if q.any():ans+=q.mean()*abs(p[q].mean()-y[q].mean())
 return ans
def met(y,p,t):
 q=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
 return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'f1':f1_score(y,q),'balanced_accuracy':balanced_accuracy_score(y,q),'sensitivity':recall_score(y,q),'specificity':tn/(tn+fp),'brier':brier_score_loss(y,p),'ece':ece(y,p)}
def meanpred(group,entity,sp):
 z=u[(u.group==group)&(u.entity==entity)&(u.split==sp)]
 paths=z.prediction_file.drop_duplicates().tolist(); ds=[pd.read_csv(ROOT/x).sort_values('case_id').reset_index(drop=True) for x in paths]
 assert len(ds)==5 and all((d.case_id==ds[0].case_id).all() and (d.y_true==ds[0].y_true).all() for d in ds)
 return ds[0].case_id.to_numpy(),ds[0].y_true.to_numpy(),np.mean([d.probability.to_numpy() for d in ds],axis=0),float(np.mean([d.threshold.iloc[0] for d in ds]))
cache={};ci=[]
for label,g,e in entities:
 for sp in ['A_val','B_test','C_test']:
  ids,y,p,t=meanpred(g,e,sp);cache[(label,sp)]=(ids,y,p,t)
  point=met(y,p,t); rng=np.random.default_rng(abs(hash((label,sp,20260925)))%(2**32))
  vals={k:[] for k in point}
  while len(vals['auroc'])<N:
   ix=rng.integers(0,len(y),len(y))
   if len(np.unique(y[ix]))<2:continue
   z=met(y[ix],p[ix],t)
   for k in vals:vals[k].append(z[k])
  for k,v in point.items():ci.append({'label':label,'group':g,'entity':e,'split':sp,'metric':k,'value':v,'ci_low':float(np.quantile(vals[k],.025)),'ci_high':float(np.quantile(vals[k],.975)),'bootstrap_replicates':N})
ci=pd.DataFrame(ci);ci.to_csv(TAB/'rq3_bootstrap_confidence_intervals.csv',index=False)
pairs=[]
for label,_,_ in entities[1:]:
 for sp in ['A_val','B_test','C_test']:
  _,y,a,ta=cache[(label,sp)];_,yb,b,tb=cache[('source_selected_base',sp)];assert (y==yb).all()
  rng=np.random.default_rng(abs(hash((label,sp,20260926)))%(2**32));vals={k:[] for k in hi|{'brier','ece'}}
  while len(vals['auroc'])<N:
   ix=rng.integers(0,len(y),len(y))
   if len(np.unique(y[ix]))<2:continue
   ma,mb=met(y[ix],a[ix],ta),met(y[ix],b[ix],tb)
   for k in vals:vals[k].append((ma[k]-mb[k]) if k in hi else (mb[k]-ma[k]))
  for k,v in vals.items():pairs.append({'comparison':label+' minus source_selected_base','split':sp,'metric':k,'improvement':float(np.mean(v)),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975)),'bootstrap_replicates':N})
pd.DataFrame(pairs).to_csv(TAB/'rq3_paired_bootstrap_comparisons.csv',index=False)
json.dump({'selection':'All reported entities selected solely by maximum mean A_val AUROC in their prespecified group; B/C not used for selection','n_bootstrap':N,'entities':[{'label':x[0],'group':x[1],'entity':x[2]} for x in entities],'paired_comparisons':'Seed-mean probabilities, shared resamples; positive means improvement'},open(RQ3/'configs'/'bootstrap_protocol.json','w'),indent=2)
print('BOOTSTRAP_DONE',len(ci),len(pairs),base.group,base.entity,prob.group,prob.entity,rep.entity)
