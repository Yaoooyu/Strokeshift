from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
R=Path(__file__).parent/'analysis'/'RQ2';T=R/'tables';F=R/'figures';T.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
rng=np.random.default_rng(20260925); ci=[]
for p in sorted((R/'predictions').glob('*.csv')):
 d=pd.read_csv(p);q=p.stem.split('_');mod='CT' if q[0]=='ct' else 'EHR';model=q[1];seed=int(q[2][4:]);split='_'.join(q[3:]);y=d.y_true.to_numpy();z=d.probability.to_numpy();vals=[]
 for _ in range(2000):
  ix=rng.integers(0,len(y),len(y))
  if len(np.unique(y[ix]))==2:vals.append([roc_auc_score(y[ix],z[ix]),average_precision_score(y[ix],z[ix]),brier_score_loss(y[ix],z[ix])])
 for i,k in enumerate(['auroc','auprc','brier']):ci.append([mod,model,seed,split,k,np.quantile([v[i] for v in vals],.025),np.quantile([v[i] for v in vals],.975),len(vals)])
ci=pd.DataFrame(ci,columns=['modality','model','seed','split','metric','bootstrap_ci_low','bootstrap_ci_high','bootstrap_valid']);ci.to_csv(T/'bootstrap_confidence_intervals.csv',index=False)
a=pd.concat([pd.read_csv(R/'rq2_ct_per_seed_results.csv'),pd.read_csv(R/'rq2_ehr_per_seed_results.csv')]);a=a.merge(ci,on=['modality','model','seed','split','metric'],how='left');a.to_csv(R/'rq2_all_results.csv',index=False)
s=a.groupby(['modality','model','split','metric']).agg(mean=('value','mean'),seed_sd=('value','std'),ci_low=('bootstrap_ci_low','mean'),ci_high=('bootstrap_ci_high','mean'),n_seeds=('seed','nunique')).reset_index();s.to_csv(T/'rq2_performance_summary.csv',index=False)
w=a[a.metric.isin(['auroc','auprc','brier'])].pivot(index=['modality','model','seed','metric'],columns='split',values='value').reset_index();o=[]
for r in w.itertuples():
 for ex in ['B_test','C_test']:
  ext=getattr(r,ex);deg=r.A_val-ext if r.metric!='brier' else ext-r.A_val;o.append([r.modality,r.model,r.seed,r.metric,ex[0],r.A_val,ext,deg,ext/r.A_val if r.metric!='brier' else np.nan])
d=pd.DataFrame(o,columns=['modality','model','seed','metric','external_center','a_val','external','degradation','retention']);d.to_csv(R/'rq2_performance_drop.csv',index=False);ds=d.groupby(['modality','model','metric','external_center']).agg(mean_degradation=('degradation','mean'),seed_sd=('degradation','std'),retention=('retention','mean')).reset_index();ds.to_csv(T/'RQ2_3_performance_degradation.csv',index=False)
for mod in ['CT','EHR']:
 x=s[(s.modality==mod)&(s.metric=='auroc')];fig,ax=plt.subplots(figsize=(7,4))
 for n,g in x.groupby('model'):
  g=g.set_index('split').reindex(['A_val','B_test','C_test']);ax.plot(['A-val','B','C'],g['mean'],'o-',label=n)
 ax.set(ylim=(0,1),ylabel='AUROC',title=f'{mod}-only performance across centers');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(F/f'{mod.lower()}_performance.png',dpi=400);plt.close(fig)
for metric in ['auroc','auprc']:
 x=ds[ds.metric==metric];fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(len(x)),x.mean_degradation,yerr=x.seed_sd);ax.axhline(0,color='k');ax.set_xticks(range(len(x)),[f'{r.model}\n{r.external_center}' for r in x.itertuples()],rotation=55,ha='right',fontsize=7);ax.set_ylabel('Degradation (positive=worse)');ax.set_title(metric.upper()+' external performance drop');fig.tight_layout();fig.savefig(F/f'{metric}_drop.png',dpi=400);plt.close(fig)
ct=ds[(ds.modality=='CT')&(ds.metric=='auroc')].groupby('external_center').mean_degradation.mean();eh=ds[(ds.modality=='EHR')&(ds.metric=='auroc')].groupby('external_center').mean_degradation.mean()
(R/'RQ2_report.md').write_text(f'''# RQ2 report\n\nFive seeds per model; all preprocessing/model selection/thresholds use A only. B/C are untouched external evaluations. Bootstrap CIs use 2,000 replicates from stored predictions and are separate from seed SD.\n\nMean AUROC degradation: CT A→B={ct['B']:.3f}, A→C={ct['C']:.3f}; EHR A→B={eh['B']:.3f}, A→C={eh['C']:.3f}.\n\nUse `rq2_performance_drop.csv`, `tables/bootstrap_confidence_intervals.csv`, and primary performance summary for model-specific inference. B=41 and C=78 require cautious interpretation; only two external centers preclude correlation inference for RQ1×RQ2.\n''')
print('complete')
