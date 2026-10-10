#!/usr/bin/env python3
"""Unified RQ3 evaluation, source-only gain reference, and external degradation."""
from pathlib import Path
import re,json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,recall_score,confusion_matrix,brier_score_loss
ROOT=Path(__file__).resolve().parents[4]; RQ2=ROOT/'vlbd_benchmark'/'analysis'/'RQ2'/'predictions'; RQ3=ROOT/'vlbd_benchmark'/'analysis'/'RQ3'; TAB=RQ3/'tables'
def ece(y,p):
 ans=0.;edges=np.linspace(0,1,11)
 for i in range(10):
  q=(p>=edges[i])&(p<(edges[i+1] if i<9 else 1.00001))
  if q.any():ans+=q.mean()*abs(p[q].mean()-y[q].mean())
 return ans
def met(y,p,t):
 q=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
 return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'f1':f1_score(y,q),'balanced_accuracy':balanced_accuracy_score(y,q),'sensitivity':recall_score(y,q),'specificity':tn/(tn+fp),'brier':brier_score_loss(y,p),'ece':ece(y,p)}
records=[]; index={}
def add(group,entity,seed,sp,path,components=''):
 d=pd.read_csv(path);assert d.case_id.nunique()==len(d) and d.probability.between(0,1).all()
 t=float(d.threshold.iloc[0]); index[(group,entity,int(seed),sp)]=d
 for metric,val in met(d.y_true.to_numpy(),d.probability.to_numpy(),t).items():records.append({'group':group,'entity':entity,'seed':int(seed),'split':sp,'metric':metric,'value':val,'threshold':t,'components':components,'prediction_file':str(path.relative_to(ROOT))})
r1=re.compile(r'^(ct|ehr)_(.+)_seed(\d+)_(A_val|B_test|C_test)\.csv$')
for p in RQ2.glob('*.csv'):
 q=r1.match(p.name)
 if q:add('base_'+q.group(1),q.group(2),q.group(3),q.group(4),p)
r2=re.compile(r'^(equal_average|oof_weighted_average|oof_logistic_stacking)_ct-(.+)_ehr-(.+)_seed(\d+)_(A_val|B_test|C_test)\.csv$')
for p in (RQ3/'probability_fusion'/'predictions').glob('*.csv'):
 q=r2.match(p.name)
 if q:add('probability_'+q.group(1),f'CT:{q.group(2)} + EHR:{q.group(3)}',q.group(4),q.group(5),p,q.group(2)+'|'+q.group(3))
r3=re.compile(r'^(concat_mlp|gmu|attention|cross_modal_transformer)_seed(\d+)_(A_val|B_test|C_test)\.csv$')
for p in (RQ3/'representation_fusion'/'predictions').glob('*.csv'):
 q=r3.match(p.name)
 if q:add('representation',q.group(1),q.group(2),q.group(3),p)
out=pd.DataFrame(records);assert len(out)==7200,len(out)
out.to_csv(RQ3/'rq3_unified_per_seed_results.csv',index=False)
summary=out.groupby(['group','entity','split','metric']).agg(mean=('value','mean'),sd=('value','std'),n_seeds=('seed','nunique')).reset_index()
summary.to_csv(TAB/'rq3_unified_summary.csv',index=False)
# Pairwise probability-fusion gain relative to the better matched component, without external selection.
hi={'auroc','auprc','f1','balanced_accuracy','sensitivity','specificity'}; gains=[]
for row in out[out.group.str.startswith('probability_')].itertuples(index=False):
 ct,ehr=row.components.split('|'); a=index[('base_ct',ct,row.seed,row.split)]; b=index[('base_ehr',ehr,row.seed,row.split)]; f=index[(row.group,row.entity,row.seed,row.split)]
 ma=met(a.y_true.to_numpy(),a.probability.to_numpy(),float(a.threshold.iloc[0]))[row.metric]; mb=met(b.y_true.to_numpy(),b.probability.to_numpy(),float(b.threshold.iloc[0]))[row.metric]
 ref=max(ma,mb) if row.metric in hi else min(ma,mb)
 gain=row.value-ref if row.metric in hi else ref-row.value
 gains.append({'group':row.group,'entity':row.entity,'seed':row.seed,'split':row.split,'metric':row.metric,'fusion_value':row.value,'best_component_value':ref,'gain_over_best_component':gain})
g=pd.DataFrame(gains);g.to_csv(TAB/'rq3_probability_fusion_gain_per_seed.csv',index=False)
g.groupby(['group','entity','split','metric']).agg(mean_gain=('gain_over_best_component','mean'),sd_gain=('gain_over_best_component','std'),n_seeds=('seed','nunique')).reset_index().to_csv(TAB/'rq3_probability_fusion_gain_summary.csv',index=False)
# Source-only base reference: select entity on A_val mean AUROC, then document its external comparisons for representation models.
base=summary[(summary.group.isin(['base_ct','base_ehr']))&(summary.split=='A_val')&(summary.metric=='auroc')]
winner=base.loc[base['mean'].idxmax()]
rep=out[out.group=='representation'].copy(); ref=out[(out.group==winner.group)&(out.entity==winner.entity)]
cmp=rep.merge(ref,on=['seed','split','metric'],suffixes=('_representation','_source_base'),validate='many_to_one')
cmp['difference']=np.where(cmp.metric.isin(hi),cmp.value_representation-cmp.value_source_base,cmp.value_source_base-cmp.value_representation)
cmp['source_selected_base_group']=winner.group;cmp['source_selected_base_entity']=winner.entity
cmp.to_csv(TAB/'rq3_representation_vs_source_selected_base.csv',index=False)
# Shift/degradation: positive value always means decline from A_val to target.
pv=out.pivot_table(index=['group','entity','seed','metric'],columns='split',values='value').reset_index();degs=[]
for target in ['B_test','C_test']:
 d=pv.dropna(subset=['A_val',target]).copy();d['target']=target;d['degradation']=np.where(d.metric.isin(hi),d['A_val']-d[target],d[target]-d['A_val']);degs.append(d)
deg=pd.concat(degs,ignore_index=True);deg.to_csv(TAB/'rq3_external_degradation_per_seed.csv',index=False)
deg.groupby(['group','entity','target','metric']).agg(mean_degradation=('degradation','mean'),sd_degradation=('degradation','std'),n_seeds=('seed','nunique')).reset_index().to_csv(TAB/'rq3_external_degradation_summary.csv',index=False)
json.dump({'metrics':['auroc','auprc','f1','balanced_accuracy','sensitivity','specificity','brier','ece'],'predictions_evaluated':900,'per_seed_metric_rows':7200,'source_selected_reference':{'group':winner.group,'entity':winner.entity,'criterion':'maximum mean A_val AUROC among frozen base models only'},'B_C_role':'held-out evaluation only'},open(RQ3/'configs'/'unified_evaluation_protocol.json','w'),indent=2)
print('UNIFIED_DONE',len(out),len(summary),winner.group,winner.entity)
