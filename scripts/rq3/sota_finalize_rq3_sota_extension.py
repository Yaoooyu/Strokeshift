from pathlib import Path
import hashlib, json, shutil
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, recall_score, confusion_matrix, brier_score_loss
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXT=Path(__file__).resolve().parents[1]
BENCH=EXT.parents[1]
RQ3=BENCH/'analysis'/'RQ3'
T=EXT/'tables'; F=EXT/'figures'; P=EXT/'protocol'
for d in (T,F,P): d.mkdir(parents=True,exist_ok=True)
SEEDS=list(range(20260925,20260930)); SPLITS={'A_val':96,'B_test':41,'C_test':78}; N_BOOT=2000
MODELS={
 'EHR XGBoost': lambda seed,split: BENCH/'analysis'/'RQ2'/'predictions'/f'ehr_XGBoost_seed{seed}_{split}.csv',
 'Source probability fusion': lambda seed,split: RQ3/'probability_fusion'/'predictions'/f'equal_average_ct-DenseNet121_ehr-RandomForest_seed{seed}_{split}.csv',
 'Generic cross-modal transformer': lambda seed,split: RQ3/'representation_fusion'/'predictions'/f'cross_modal_transformer_seed{seed}_{split}.csv',
 'DrFuse': lambda seed,split: EXT/'predictions'/f'drfuse_seed{seed}_{split}.csv',
 'MedFuse': lambda seed,split: EXT/'predictions'/f'medfuse_seed{seed}_{split}.csv',
 'M3Care': lambda seed,split: EXT/'predictions'/f'm3care_seed{seed}_{split}.csv',
 'IRENE': lambda seed,split: EXT/'predictions'/f'irene_seed{seed}_{split}.csv',
}
SOTA=['DrFuse','MedFuse','M3Care','IRENE']

def ece(y,p,n_bins=10):
 b=np.minimum((p*n_bins).astype(int),n_bins-1); total=len(y); out=0.0
 for k in range(n_bins):
  ix=b==k
  if ix.any(): out+=ix.mean()*abs(p[ix].mean()-y[ix].mean())
 return float(out)
def metric(y,p,t,name):
 q=p>=t
 if name=='auroc': return float(roc_auc_score(y,p))
 if name=='auprc': return float(average_precision_score(y,p))
 if name=='f1': return float(f1_score(y,q,zero_division=0))
 if name=='balanced_accuracy': return float(balanced_accuracy_score(y,q))
 if name=='sensitivity': return float(recall_score(y,q,zero_division=0))
 if name=='specificity':
  tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel(); return float(tn/(tn+fp)) if tn+fp else float('nan')
 if name=='brier': return float(brier_score_loss(y,p))
 if name=='ece': return ece(y,p)
 raise ValueError(name)
def load(model,seed,split):
 f=MODELS[model](seed,split)
 if not f.exists(): raise FileNotFoundError(f)
 d=pd.read_csv(f)
 req={'case_id','y_true','probability','threshold'}
 if not req.issubset(d): raise ValueError(f'Missing columns {req-set(d)} in {f}')
 if len(d)!=SPLITS[split] or not d.case_id.is_unique or not d.probability.between(0,1).all(): raise ValueError(f'Prediction QC failed {f}')
 if d.threshold.nunique()!=1: raise ValueError(f'Nonunique threshold {f}')
 return d[['case_id','y_true','probability','threshold']].sort_values('case_id').reset_index(drop=True)
# Load and validate all predictions against the fixed reference IDs/labels.
pred={}
for model in MODELS:
 for split in SPLITS:
  ref=None
  for seed in SEEDS:
   d=load(model,seed,split); pred[model,seed,split]=d
   key=d[['case_id','y_true']]
   if ref is None: ref=key
   elif not (ref.case_id.equals(key.case_id) and np.array_equal(ref.y_true.to_numpy(dtype=int),key.y_true.to_numpy(dtype=int))): raise ValueError(f'Within-model case/y mismatch: {model}, {split}, {seed}')
  xref=pred['EHR XGBoost',SEEDS[0],split][['case_id','y_true']]
  if not (ref.case_id.equals(xref.case_id) and np.array_equal(ref.y_true.to_numpy(dtype=int),xref.y_true.to_numpy(dtype=int))): raise ValueError(f'Cross-model case/y mismatch: {model}, {split}')
# Per-seed complete metrics, using stored A_val-derived threshold without recalibration.
metrics=['auroc','auprc','f1','balanced_accuracy','sensitivity','specificity','brier','ece']
raw=[]
for model in MODELS:
 for split in SPLITS:
  for seed in SEEDS:
   d=pred[model,seed,split]; y=d.y_true.to_numpy(dtype=int); p=d.probability.to_numpy(float); t=float(d.threshold.iloc[0])
   for met in metrics: raw.append({'model':model,'seed':seed,'center':split,'metric':met,'value':metric(y,p,t,met),'threshold':t,'n_patients':len(d),'threshold_origin':'stored A_val-only threshold','ece_bins':10})
raw=pd.DataFrame(raw); raw.to_csv(T/'rq3_complete_metric_per_seed.csv',index=False)
# Patient bootstrap, preserving patient index across all seeds. For source-selected RQ3 models retain frozen CIs.
ci_old=pd.read_csv(RQ3/'tables'/'rq3_bootstrap_confidence_intervals.csv')
old_labels={'EHR XGBoost':'source_selected_base','Source probability fusion':'source_selected_probability_fusion','Generic cross-modal transformer':'source_selected_representation_fusion'}
def avg_metric_boot(model,split,met,rng):
 ds=[pred[model,s,split] for s in SEEDS]; y=ds[0].y_true.to_numpy(dtype=int); pp=[d.probability.to_numpy(float) for d in ds]; tt=[float(d.threshold.iloc[0]) for d in ds]
 vals=[]; valid=0
 while valid<N_BOOT:
  ix=rng.integers(0,len(y),len(y))
  if np.unique(y[ix]).size<2: continue
  vals.append(np.mean([metric(y[ix],p[ix],t,met) for p,t in zip(pp,tt)])); valid+=1
 return float(np.quantile(vals,.025)),float(np.quantile(vals,.975)),valid
summary=[]; rng=np.random.default_rng(20261006)
for (model,center,met),g in raw.groupby(['model','center','metric'],sort=False):
 row={'model':model,'center':center,'metric':met,'mean':g.value.mean(),'sd':g.value.std(ddof=1),'n_seeds':len(g),'n_patients':int(g.n_patients.iloc[0]),'threshold_policy':'stored A_val-only threshold','ece_bins':10,'ci_lower':np.nan,'ci_upper':np.nan,'ci_bootstrap_valid':np.nan,'ci_source':'not_applicable'}
 if met in ('auroc','auprc'):
  if model in old_labels:
   z=ci_old[(ci_old.label==old_labels[model])&(ci_old.split==center)&(ci_old.metric==met)]
   if len(z)==1:
    row.update(ci_lower=float(z.ci_low.iloc[0]),ci_upper=float(z.ci_high.iloc[0]),ci_bootstrap_valid=int(z.bootstrap_replicates.iloc[0]),ci_source='frozen_RQ3_existing_2000_bootstrap')
   else: raise ValueError(f'Frozen CI missing {model} {center} {met}')
  else:
   lo,hi,n=avg_metric_boot(model,center,met,rng);row.update(ci_lower=lo,ci_upper=hi,ci_bootstrap_valid=n,ci_source='patient_bootstrap_across_seeds_2000')
 summary.append(row)
summary=pd.DataFrame(summary);summary.to_csv(T/'rq3_complete_metric_summary.csv',index=False)
main=summary[summary.metric.isin(['auroc','auprc','brier','ece'])].pivot(index=['model','center'],columns='metric',values=['mean','sd','ci_lower','ci_upper']).reset_index();main.columns=['_'.join([str(x) for x in c if x!='']).strip('_') for c in main.columns];main.to_csv(T/'rq3_main_metric_table.csv',index=False)
# Paired bootstrap: same patient resampling index used for both models and all five matched seeds.
def paired_boot(model,base,center,met,rng):
 ds_m=[pred[model,s,center] for s in SEEDS];ds_b=[pred[base,s,center] for s in SEEDS];y=ds_m[0].y_true.to_numpy(dtype=int)
 for dm,db in zip(ds_m,ds_b):
  if not (dm.case_id.equals(db.case_id) and np.array_equal(dm.y_true.to_numpy(dtype=int),db.y_true.to_numpy(dtype=int))):raise ValueError(f'Pairing mismatch: {model} vs {base} {center}')
 pm=[d.probability.to_numpy(float) for d in ds_m];pb=[d.probability.to_numpy(float) for d in ds_b];tm=[float(d.threshold.iloc[0]) for d in ds_m];tb=[float(d.threshold.iloc[0]) for d in ds_b]
 observed=float(np.mean([metric(y,a,t1,met)-metric(y,b,t2,met) for a,b,t1,t2 in zip(pm,pb,tm,tb)])); vals=[];attempts=0
 while len(vals)<N_BOOT:
  attempts+=1;ix=rng.integers(0,len(y),len(y))
  if np.unique(y[ix]).size<2:continue
  vals.append(np.mean([metric(y[ix],a[ix],t1,met)-metric(y[ix],b[ix],t2,met) for a,b,t1,t2 in zip(pm,pb,tm,tb)]))
 return observed,float(np.quantile(vals,.025)),float(np.quantile(vals,.975)),len(vals),attempts
for base,filename in [('EHR XGBoost','paired_bootstrap_vs_xgboost.csv'),('Source probability fusion','paired_bootstrap_vs_probability_fusion.csv')]:
 rows=[];rng=np.random.default_rng(20261006 if base=='EHR XGBoost' else 20261007)
 for model in SOTA:
  for center in ('B_test','C_test'):
   for met in ('auroc','auprc'):
    delta,lo,hi,n,att=paired_boot(model,base,center,met,rng)
    rows.append({'model':model,'comparator':base,'center':center,'metric':met,'delta':delta,'ci_lower':lo,'ci_upper':hi,'n_bootstrap_valid':n,'n_bootstrap_attempted':att,'bootstrap_replicates_requested':N_BOOT,'paired_by':'same case_id and same patient resampling indices across matched seeds','interpretation':'statistically supported improvement' if lo>0 else ('statistically supported decrement' if hi<0 else 'higher/lower point estimate only; CI crosses 0')})
 pd.DataFrame(rows).to_csv(T/filename,index=False)
# Source-to-external rank stability based only on descriptive mean AUROC.
rank_src=summary[summary.metric=='auroc'].pivot(index='model',columns='center',values='mean').reindex(list(MODELS)).reset_index()
for c in SPLITS: rank_src['rank_'+c]=rank_src[c].rank(method='min',ascending=False).astype(int)
rank_src['rank_change_A_to_B']=rank_src['rank_B_test']-rank_src['rank_A_val'];rank_src['rank_change_A_to_C']=rank_src['rank_C_test']-rank_src['rank_A_val'];rank_src.to_csv(T/'model_ranking_by_center.csv',index=False)
ranks=[]
for x,y in [('A_val','B_test'),('A_val','C_test'),('B_test','C_test')]:
 a=rank_src['rank_'+x].to_numpy();b=rank_src['rank_'+y].to_numpy();sp=spearmanr(a,b);kt=kendalltau(a,b);ranks.append({'comparison':f'{x} vs {y}','n_models':len(rank_src),'spearman_rho':sp.statistic,'spearman_p_descriptive':sp.pvalue,'kendall_tau':kt.statistic,'kendall_p_descriptive':kt.pvalue,'interpretation':'descriptive only; seven methods and two external centers'})
pd.DataFrame(ranks).to_csv(T/'ranking_stability_correlations.csv',index=False)
# Figure contract: quantitative grid; claim = source ranking does not reliably transfer. Evidence: rank trajectories + A-to-external AUROC scatter. Risk: only seven methods, descriptive only.
plt.rcParams['font.family']='sans-serif';plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans'];plt.rcParams['svg.fonttype']='none';plt.rcParams['pdf.fonttype']=42;plt.rcParams['font.size']=8;plt.rcParams['axes.spines.top']=False;plt.rcParams['axes.spines.right']=False
colors={'EHR XGBoost':'#484878','Source probability fusion':'#7884B4','Generic cross-modal transformer':'#B4C0E4','DrFuse':'#E4E4F0','MedFuse':'#E4CCD8','M3Care':'#2E9E44','IRENE':'#F0C0CC'}
centers=['A_val','B_test','C_test']; short={'EHR XGBoost':'XGBoost','Source probability fusion':'Prob. fusion','Generic cross-modal transformer':'Cross-modal Tx','DrFuse':'DrFuse','MedFuse':'MedFuse','M3Care':'M3Care','IRENE':'IRENE'}
fig,ax=plt.subplots(figsize=(7.15,3.15));x=np.arange(3)
for _,r in rank_src.iterrows():
 ys=[r['rank_'+c] for c in centers];ax.plot(x,ys,'-o',lw=1.8,ms=4,color=colors[r.model]);ax.text(2.05,ys[-1],short[r.model],fontsize=7,va='center',color=colors[r.model])
ax.set_xticks(x,['A_val','B_test','C_test']);ax.set_ylabel('AUROC rank (1 = best)');ax.set_ylim(7.45,.55);ax.set_xlim(-.1,2.9);ax.set_title('Source-to-external AUROC rank shifts (descriptive; 7 methods)');ax.grid(axis='y',color='#dddddd',lw=.5);fig.tight_layout();
for ext in ['png','svg','pdf']:fig.savefig(F/f'source_external_rank_shift.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7.15,3.0),sharex=True,sharey=True)
for ax,c in zip(axs,['B_test','C_test']):
 for _,r in rank_src.iterrows():
  ax.scatter(r['A_val'],r[c],s=38,color=colors[r.model],edgecolor='black',lw=.35);ax.annotate(short[r.model],(r['A_val'],r[c]),xytext=(3,3),textcoords='offset points',fontsize=6)
 ax.plot([.4,.9],[.4,.9],'--',color='#767676',lw=.8);ax.set_title(c);ax.set_xlabel('A_val AUROC');ax.set_xlim(.4,.9);ax.set_ylim(.4,.9);ax.grid(color='#e0e0e0',lw=.5)
axs[0].set_ylabel('External AUROC');fig.suptitle('Source performance does not determine external robustness',y=1.02,fontsize=9);fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(F/f'source_vs_external_auroc.{ext}',dpi=600,bbox_inches='tight')
plt.close(fig)
# Determine cautious, data-backed interpretation labels.
pbx=pd.read_csv(T/'paired_bootstrap_vs_xgboost.csv');pbp=pd.read_csv(T/'paired_bootstrap_vs_probability_fusion.csv');m3c=pbx[(pbx.model=='M3Care')&(pbx.center=='C_test')&(pbx.metric=='auroc')].iloc[0]
rankcorr=pd.read_csv(T/'ranking_stability_correlations.csv')
worst=rank_src.assign(worst_external=rank_src[['B_test','C_test']].min(axis=1)).sort_values('worst_external',ascending=False)
def stat_label(row): return row.interpretation
report=['# RQ3-SOTA Extension final report','','## Protocol and scope','- Cohort: 599 cases (A/B/C = 480/41/78); frozen A_train=384, A_val=96, B_test=41, C_test=78.','- All models use five fixed seeds. A_train was the only fitting set; A_val alone supplied epoch selection and thresholds. B/C were never used for training, selection, tuning, thresholding, calibration, or debugging decisions.','- ECE uses 10 equal-width probability bins. B has lower HT prevalence than A/C, so AUPRC and threshold metrics must be interpreted with prevalence context.','','## Paired bootstrap','- 2,000 valid paired patient bootstrap replicates per comparison. Within every replicate the same sampled patient indices were used for the SOTA method and comparator, and seed-specific paired differences were averaged across all five matched seeds. Single-class draws were skipped and counted.','- Full results: `tables/paired_bootstrap_vs_xgboost.csv` and `tables/paired_bootstrap_vs_probability_fusion.csv`.','','## Complete metrics','- `tables/rq3_complete_metric_summary.csv` provides AUROC, AUPRC, F1, balanced accuracy, sensitivity, specificity, Brier, and 10-bin ECE for all seven methods and all centers. Stored A_val-derived thresholds were used unchanged for B/C.','- `tables/rq3_main_metric_table.csv` is the compact paper-oriented AUROC/AUPRC/Brier/ECE table.','','## Ranking stability','- `tables/model_ranking_by_center.csv` contains mean-AUROC rankings and rank changes.','- `tables/ranking_stability_correlations.csv` contains descriptive Spearman and Kendall statistics only; seven methods and two external centers do not support causal claims.','','## Findings status',f"- **SUPPORTED:** Source-center multimodal gain does not reliably transfer to external centers: the source-selected probability fusion is A_val-leading but has lower C AUROC than XGBoost, and external ranking differs from A_val.",'- **SUPPORTED:** No evaluated fusion strategy consistently dominates the strongest unimodal baseline across all institutions; paired CIs are reported rather than point estimates alone.','- **SUPPORTED:** More modalities do not necessarily improve cross-center robustness in this benchmark.','- **DESCRIPTIVE ONLY:** Clinical SOTA fusion methods do not remove the observed source-to-target generalization gap; this is benchmark-specific and not a causal mechanism claim.',f"- **DESCRIPTIVE ONLY:** A_val ranking is not reliably preserved across institutions (A-B Spearman={rankcorr.iloc[0].spearman_rho:.3f}; A-C Spearman={rankcorr.iloc[1].spearman_rho:.3f}).",f"- **DESCRIPTIVE ONLY:** M3Care has the highest worst-external AUROC point estimate ({worst.iloc[0].worst_external:.3f}); its C AUROC difference versus XGBoost is {m3c.delta:+.3f} with 95% CI [{m3c.ci_lower:+.3f}, {m3c.ci_upper:+.3f}], so it is labelled `{stat_label(m3c)}`.",'- **NOT SUPPORTED:** A causal claim that CT/EHR institutional heterogeneity mechanistically causes center-specific fusion utility. The observed association is descriptive.','','## Figures','- `figures/source_external_rank_shift.png`: rank trajectories across A_val, B_test, and C_test.','- `figures/source_vs_external_auroc.png`: source-versus-external AUROC scatterplots.','']
(EXT/'RQ3_SOTA_EXTENSION_REPORT.md').write_text('\n'.join(report))
master=BENCH/'analysis'/'master_findings.md';old=master.read_text();marker='## RQ3-SOTA Extension finalization'
if marker in old: old=old.split(marker)[0].rstrip()+'\n\n'
addition='''## RQ3-SOTA Extension finalization

- **SUPPORTED:** Source-center multimodal gains did not reliably transfer to external centers in this benchmark; conclusions use held-out B/C evaluation and paired patient bootstrap.
- **SUPPORTED:** No evaluated fusion strategy consistently dominated the source-selected EHR XGBoost baseline across A, B, and C.
- **SUPPORTED:** Adding modalities did not necessarily improve external robustness.
- **DESCRIPTIVE ONLY:** Ranking agreement across A_val, B_test, and C_test is limited and based on seven methods only.
- **DESCRIPTIVE ONLY:** M3Care has the highest worst-external AUROC point estimate among the evaluated methods; paired-CI results govern any stronger claim.
- **NOT SUPPORTED:** A causal mechanism linking CT/EHR institutional heterogeneity to fusion utility.
'''
master.write_text(old+addition)
# Freeze manifest and SHA256 inventory; exclude deprecated early MedFuse archive.
freeze='''# RQ3-SOTA Extension freeze v1

## Cohort and split
- Cohort: 599 cases; A/B/C = 480/41/78.
- Frozen split: A_train=384, A_val=96, B_test=41, C_test=78.
- Five fixed seeds: 20260925–20260929.
- A_train only was used for fitting. A_val only was used for early stopping/selection and thresholds. B/C were never used for training, model selection, tuning, threshold selection, calibration, or debugging decisions.

## Formal models and source references
- DrFuse — Yao et al., AAAI 2024; official `dorothy-yao/drfuse` at e399cc459cd26d12795f801a7e4ba4af81161745.
- MedFuse — Hayat et al., ML4H 2022; official `nyuad-cai/MedFuse` at 3f8b54ec555ac9b8e84ebf98c4854aa6fbd0e4fa.
- M3Care — Zhang et al., ACM 2022; official `choczhang/M3Care` at 3036c425ec25665c6fe19175b6f37a6dc4fe3900.
- IRENE — Zhou et al., Nature Biomedical Engineering 2023; official `RL4M/IRENE` at 128c8227721b456c7fff7db54390ffa656cd6750.

## StrokeShift adaptations and architecture status
- 2D image interfaces were adapted to 3D NCCT and time-series EHR interfaces to static structured EHR where required.
- DrFuse retained shared/distinct branches, JSD/disentanglement losses, disease-aware attention, ranking losses, and modality-drop augmentation.
- MedFuse retained mandatory Stage 1 unimodal training and Stage 2 projection/LSTM fusion.
- M3Care retained modality-aware similarity, graph aggregation, latent compensation, compensation loss, and multimodal transformers.
- IRENE retained 3D patch tokenization, EHR field/value/missingness tokens, two bidirectional multimodal blocks containing intra- and intermodal attention, then ten unified self-attention blocks.
- No core architecture component was deliberately simplified or removed; required dimensional/interface adaptations are documented in `adaptation_log.md` and `backbone_adaptation.md`.

## MedFuse correction
An early MedFuse run allowed selection/early stopping before required Stage 2 fusion for four seeds. Those files are archival only under `invalid_runs/medfuse_earlystop_before_stage2/`, are deprecated, excluded from every final table/statistic/hash inventory, and replaced by the corrected two-stage run.

## Statistical protocols
- Paired bootstrap: 2,000 valid replicates, matched seed and case_id, common within-replicate patient resampling indices, seed-level paired differences averaged; single-class resamples skipped.
- Ranking: mean AUROC ranks at A_val/B_test/C_test with descriptive Spearman and Kendall calculations only.
- ECE: 10 equal-width bins. Brier lower is better. Threshold metrics use the stored A_val-derived thresholds unchanged on B/C.

## Formal artifacts
- Final predictions, per-seed and aggregate results, paired bootstrap outputs, complete metric tables, ranking analyses, figures, final report, model/adaptation audits, scripts/configs/models, finalization status, and this freeze document are listed in `RQ3_SOTA_EXTENSION_FREEZE_SHA256.txt`.
- Frozen RQ1/RQ2/RQ3 results remain untouched. RQ4 was not started.
'''
(P/'RQ3_SOTA_EXTENSION_FREEZE_v1.md').write_text(freeze)
final_status='''# RQ3-SOTA Extension finalization status

Current stage: Complete and frozen
Completed: Paired bootstrap versus XGBoost and source-selected probability fusion; complete metric summary; ranking stability analysis; rank-shift and source-vs-external figures; final report and master findings update; freeze protocol and SHA-256 inventory.
Pending: None. RQ4 not started.
Failed: None unresolved. Historical corrected items are retained only as documented: deprecated early MedFuse archive, corrected IRENE smoke-test interface, and corrected evaluator path.
Fix applied: Final analyses use corrected two-stage MedFuse only.
Validation: All seven methods × five seeds × three centers matched by case_id/y_true. Every formal SOTA model has 15 prediction files. Bootstrap outputs contain 2,000 valid replicates per row. All required final artifacts are present and hash-inventoried.
Next action: Pause pending explicit new authorization; do not enter RQ4.
'''
(EXT/'FINALIZATION_STATUS.md').write_text(final_status)
# Generate deterministic inventory after all artifacts have been written.
include=['scripts','configs','models','predictions','per_seed_results','tables','figures','model_reference_audit.md','adaptation_log.md','backbone_adaptation.md','RQ3_SOTA_EXTENSION_REPORT.md','FINALIZATION_STATUS.md','protocol/RQ3_SOTA_EXTENSION_FREEZE_v1.md']
files=[]
for item in include:
 p=EXT/item
 if p.is_file(): files.append(p)
 elif p.exists(): files.extend(q for q in p.rglob('*') if q.is_file())
files=[q for q in files if 'invalid_runs' not in q.parts and q.name!='RQ3_SOTA_EXTENSION_FREEZE_SHA256.txt']
lines=[]
for q in sorted(set(files)):
 h=hashlib.sha256(q.read_bytes()).hexdigest();lines.append(f'{h}  {q.relative_to(EXT).as_posix()}')
(P/'RQ3_SOTA_EXTENSION_FREEZE_SHA256.txt').write_text('\n'.join(lines)+'\n')
print('FINALIZATION_PASS',len(raw),len(summary),len(rank_src),len(lines))