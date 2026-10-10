from pathlib import Path
import pandas as pd,numpy as np
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1];rng=np.random.default_rng(20260930)
models=['drfuse','medfuse','m3care','irene'];frames=[]
for name in models:
 d=pd.read_csv(OUT/'per_seed_results'/f'{name}.csv');d['source']='SOTA_extension';frames.append(d[['source','model','seed','split','metric','value']])
sota=pd.concat(frames,ignore_index=True)
def summarize(d,keys):
 rows=[]
 for key,g in d.groupby(keys,dropna=False):
  a=g.value.to_numpy();boot=np.mean(rng.choice(a,(10000,len(a)),replace=True),axis=1)
  row=dict(zip(keys,key if isinstance(key,tuple) else (key,)));row.update(n=len(a),mean=a.mean(),sd=a.std(ddof=1),ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975));rows.append(row)
 return pd.DataFrame(rows)
sum_sota=summarize(sota,['model','split','metric']);sum_sota.to_csv(OUT/'tables'/'sota_aggregate_results.csv',index=False)
rq=pd.read_csv(ROOT/'analysis/RQ3/rq3_unified_per_seed_results.csv');rq=rq.rename(columns={'entity':'model'})[['group','model','seed','split','metric','value']];rqsum=summarize(rq,['group','model','split','metric']);rqsum.to_csv(OUT/'tables'/'frozen_rq3_summary_readonly.csv',index=False)
comparison=pd.concat([sum_sota.assign(group='SOTA_extension')[['group','model','split','metric','n','mean','sd','ci95_low','ci95_high']],rqsum[['group','model','split','metric','n','mean','sd','ci95_low','ci95_high']]],ignore_index=True);comparison.to_csv(OUT/'tables'/'rq3_sota_comparison_readonly.csv',index=False)
lines=['# RQ3-SOTA Extension report','','## Scope','Four faithful-adaptation multimodal models were trained with five fixed seeds using A_train only. A_val alone supplied early stopping and thresholds. B_test and C_test were evaluation-only. Frozen RQ3 was read only.','','## SOTA aggregate results','']
view=sum_sota[sum_sota.metric.isin(['auroc','auprc'])].pivot_table(index=['model','split'],columns='metric',values=['mean','sd'],aggfunc='first').reset_index();view.columns=['_'.join(c).strip('_') for c in view.columns]
lines+=['| Model | Split | AUROC mean ± SD | AUPRC mean ± SD |','|---|---|---:|---:|']
for _,r in view.iterrows():lines.append(f"| {r['model']} | {r['split']} | {r['mean_auroc']:.4f} ± {r['sd_auroc']:.4f} | {r['mean_auprc']:.4f} ± {r['sd_auprc']:.4f} |")
lines+=['','## Artifacts','- `tables/sota_aggregate_results.csv`: five-seed mean, SD, and nonparametric seed-bootstrap intervals.','- `tables/frozen_rq3_summary_readonly.csv`: read-only summary of every frozen RQ3 result.','- `tables/rq3_sota_comparison_readonly.csv`: combined descriptive comparison; this file makes no external-domain selection.','','## QC','All four SOTA methods have 15 prediction files each (five seeds × A_val/B_test/C_test) and 105 metric rows each. No RQ4 work was started.']
(OUT/'RQ3_SOTA_EXTENSION_REPORT.md').write_text('\n'.join(lines)+'\n')
print('UNIFIED_EVALUATION_PASS',len(sum_sota),len(rqsum),len(comparison))
