#!/usr/bin/env python3
"""RQ5 paired metrics, bootstrap, and data-efficiency summaries from frozen predictions."""
from pathlib import Path
import hashlib
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
OUT=Path(__file__).resolve().parents[1];P=OUT/'predictions'; T=OUT/'tables';T.mkdir(exist_ok=True)
rows=[]; boots=[]
def m(y,p): return {'AUROC':roc_auc_score(y,p),'AUPRC':average_precision_score(y,p)}
for ap in sorted((P/'adapted').glob('*.csv')):
 zpath=P/'zero_shot_matched'/ap.name
 a=pd.read_csv(ap).sort_values('case_id').reset_index(drop=True);z=pd.read_csv(zpath).sort_values('case_id').reset_index(drop=True)
 assert a.case_id.tolist()==z.case_id.tolist() and (a.y_true==z.y_true).all() and np.isfinite(a.probability).all() and np.isfinite(z.probability).all()
 tag=ap.stem; prefix,center,seedpart,shotpart=tag.rsplit('_',3);model=prefix.replace('_',' ');seed=int(seedpart.replace('seed',''));shot=int(shotpart.replace('shot','')); y=a.y_true.to_numpy();mm0=m(y,z.probability.to_numpy());mm1=m(y,a.probability.to_numpy())
 row={'Model':model,'Center':center,'Seed':seed,'Shot':shot,'NQuery':len(a),'NSupport':shot}
 for k in mm0:row[f'ZeroShot_{k}']=mm0[k];row[f'Adapted_{k}']=mm1[k];row[f'Delta_{k}']=mm1[k]-mm0[k]
 # retain operational metrics already calculated by runner
 rr=pd.read_csv(OUT/'results'/'rq5_seed_results.csv');q=rr[(rr.Model==model)&(rr.Center==center)&(rr.Seed==seed)&(rr.Shot==shot)]
 for metric in ['Brier','ECE','F1','BalancedAccuracy','Sensitivity','Specificity']:
  row['ZeroShot_'+metric]=float(q[q.Type=='zero_shot'][metric].iloc[0]);row['Adapted_'+metric]=float(q[q.Type=='adapted'][metric].iloc[0]);row['Delta_'+metric]=row['Adapted_'+metric]-row['ZeroShot_'+metric]
 rows.append(row)
 rng=np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:8],16)); ds={'AUROC':[],'AUPRC':[]}
 for i in range(2000):
  ix=rng.integers(0,len(y),len(y)); yy=y[ix]
  if yy.min()==yy.max():continue
  for metric,fn in [('AUROC',roc_auc_score),('AUPRC',average_precision_score)]:ds[metric].append(fn(yy,a.probability.to_numpy()[ix])-fn(yy,z.probability.to_numpy()[ix]))
 for metric,vals in ds.items():boots.append({'Model':model,'Center':center,'Seed':seed,'Shot':shot,'Metric':metric,'Delta':row['Delta_'+metric],'CI95_low':float(np.quantile(vals,.025)),'CI95_high':float(np.quantile(vals,.975)),'BootstrapValid':len(vals),'BootstrapRequested':2000,'CI_excludes_zero':bool(np.quantile(vals,.025)>0 or np.quantile(vals,.975)<0)})
allr=pd.DataFrame(rows).sort_values(['Model','Center','Shot','Seed']);allr.to_csv(T/'rq5_all_results.csv',index=False)
b=pd.DataFrame(boots).sort_values(['Model','Center','Shot','Seed','Metric']);b.to_csv(T/'rq5_paired_bootstrap.csv',index=False)
metric_cols=[c for c in allr if c.startswith(('ZeroShot_','Adapted_','Delta_'))]
agg=allr.groupby(['Model','Center','Shot'],as_index=False)[metric_cols].agg(['mean','std']);agg.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in agg.columns];agg=agg.rename(columns={'Model_':'Model','Center_':'Center','Shot_':'Shot'});agg.to_csv(T/'rq5_data_efficiency_summary.csv',index=False)
var=allr.groupby(['Model','Center','Shot'],as_index=False)[['Delta_AUROC','Delta_AUPRC']].agg(['mean','std','min','max']);var.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in var.columns];var=var.rename(columns={'Model_':'Model','Center_':'Center','Shot_':'Shot'});var.to_csv(T/'rq5_seed_variability.csv',index=False)
print('all',len(allr),'boot',len(b),'summary',len(agg),'finite',np.isfinite(allr.select_dtypes('number').to_numpy()).all(),'valid_boot_min',int(b.BootstrapValid.min()))
