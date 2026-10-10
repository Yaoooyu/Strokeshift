from pathlib import Path
import pandas as pd, numpy as np, hashlib, shutil, datetime
OUT=Path(__file__).resolve().parents[1]; V=OUT.parents[1]; T=OUT/'tables'; P=OUT/'protocol'; P.mkdir(exist_ok=True)
s=pd.read_csv(T/'rq5_data_efficiency_summary.csv'); b=pd.read_csv(T/'rq5_paired_bootstrap.csv'); a=pd.read_csv(T/'rq5_all_results.csv')
cols=['Model','Center','Shot','ZeroShot_AUROC_mean','ZeroShot_AUROC_std','Adapted_AUROC_mean','Adapted_AUROC_std','Delta_AUROC_mean','Delta_AUROC_std','ZeroShot_AUPRC_mean','Adapted_AUPRC_mean','Delta_AUPRC_mean']; d=s[cols].copy()
for c in cols[3:]: d[c]=d[c].map(lambda x:f'{x:.3f}')
relcols=['Model','Center','Shot','ZeroShot_Brier_mean','Adapted_Brier_mean','Delta_Brier_mean','ZeroShot_ECE_mean','Adapted_ECE_mean','Delta_ECE_mean']; rel=s[relcols].copy()
for c in relcols[3:]: rel[c]=rel[c].map(lambda x:f'{x:.3f}')
au=int(b[b.Metric.eq('AUROC')].CI_excludes_zero.sum()); ap=int(b[b.Metric.eq('AUPRC')].CI_excludes_zero.sum()); best=a.groupby(['Model','Center','Shot'],as_index=False).Delta_AUROC.mean().sort_values('Delta_AUROC',ascending=False).iloc[0]
report=f'''# RQ5: Target-Center Data Efficiency

## Locked design

Cohort=599; A/B/C=480/41/78; source A_train/A_val=384/96. Five fixed stratified seeds: B used nested 5/10-shot supports with fixed 31-case query; C used nested 5/10/20/40-shot supports with fixed 38-case query. Models: XGBoost, source-selected probability fusion, Generic Cross-modal Transformer, DrFuse, MedFuse, M3Care, IRENE.

Deep source trunks were frozen; terminal heads only were adapted with AdamW lr=1e-3, weight decay=1e-4, batch=2, exactly 25 updates, no target-query selection. XGBoost/fusion preserved frozen sources and used support-only rank-preserving Platt calibration; a non-positive unconstrained slope retained identity. Query labels were metrics-only.

## Completeness and uncertainty

All 210 planned units completed (150 deep and 60 reference). Predictions are case-aligned and finite. Paired bootstrap requested 2,000 replicates; minimum valid={int(b.BootstrapValid.min())}. Seed-level paired CIs excluding zero: AUROC {au}/210; AUPRC {ap}/210. Bootstrap patient uncertainty and five-seed support-sampling variability remain separate.

## Discrimination results

Values are five-seed mean; Delta=adapted minus matched-query zero-shot.

{d.sort_values(['Center','Model','Shot']).to_csv(index=False)}

## Reliability results

Negative Brier/ECE delta is an improvement.

{rel.sort_values(['Center','Model','Shot']).to_csv(index=False)}

## Interpretation

### SUPPORTED

- Complete pre-specified matched-query matrix; no query-label fitting or selection.
- For deep head-only adaptation, no seed-level paired AUROC CI excluded zero; the studied 5--40 label budgets do not establish stable discrimination recovery.
- Rank-preserving calibration cannot change AUROC ranking; its relevant outcomes are Brier/ECE and threshold behavior.
- More labels were not uniformly better; B/C and model trajectories differ.

### DESCRIPTIVE ONLY

- Largest mean AUROC delta: {best.Model}, {best.Center}, {int(best.Shot)} shots, {best.Delta_AUROC:.3f}; no consistent CI support means this is not a superiority claim.
- Relative efficiency and seed sensitivity vary by model, center, and shot.

### NOT SUPPORTED

- A universal best target-data-efficient multimodal model, monotonic label benefit, or a claim that 5--40 labels repair RQ3 transfer failure.

## Cross-RQ boundary

RQ5 adds a bounded result to RQ1 institutional heterogeneity, RQ2 modality-specific external degradation, RQ3 non-transferable fusion gain, and RQ4 missingness insufficiency: this constrained small-label protocol did not yield a statistically stable universal repair.

Primary artifacts are the four `tables/rq5_*.csv` files.
'''
(OUT/'RQ5_report.md').write_text(report)
master=V/'master_findings.md';txt=master.read_text() if master.exists() else '# Master findings\n'
section='''\n\n## RQ5 Target-Center Data Efficiency\n\n### SUPPORTED\n\n- All seven fixed models completed nested B (5/10) and C (5/10/20/40) adaptation over five seeds with matched-query comparisons.\n- Fixed head-only deep adaptation and rank-preserving calibration provided no seed-level paired-AUROC-CI evidence of stable universal discrimination recovery.\n- More labels were not uniformly better, and B/C behavior differs.\n\n### DESCRIPTIVE ONLY\n\n- Relative efficiency and support-seed sensitivity vary by model, center, and shot.\n\n### NOT SUPPORTED\n\n- A universally most target-data-efficient multimodal method or reliable repair of RQ3 transfer failure.\n'''
if '## RQ5 Target-Center Data Efficiency' not in txt: master.write_text(txt.rstrip()+section+'\n')
shutil.copy2(master,OUT/'master_findings_snapshot.md')
freeze=f'''# RQ5 FREEZE v1

Frozen {datetime.datetime.now(datetime.timezone.utc).isoformat()}. Cohort 599; A/B/C=480/41/78; A_train/A_val=384/96. B fixed-query nested 5/10 and C fixed-query nested 5/10/20/40; seeds 20260925--20260929. Models: XGBoost, source-selected probability fusion, Generic Cross-modal Transformer, DrFuse, MedFuse, M3Care, IRENE.

Deep recipe: frozen source trunk, terminal head only, AdamW lr 1e-3, wd 1e-4, batch 2, 25 updates; no early stopping/query selection. Reference sources are fixed with support-only non-decreasing Platt calibration (identity for non-positive slope). Query labels were metrics-only. All 210 units and paired 2,000-replicate bootstrap are complete. Deep checkpoint states are exact terminal-head states with frozen-source reference; see RQ5_checkpoint_storage.md. Figures are deferred. Do not infer universal robustness, monotonic label benefit, or a repaired RQ3 failure.
'''
(P/'RQ5_FREEZE_v1.md').write_text(freeze)
files=[]
for rel in ['RQ5_protocol.md','configs','scripts','splits','predictions','checkpoints','results','tables','RQ5_report.md','master_findings_snapshot.md','RQ5_checkpoint_storage.md','RQ5_monotonic_calibration_correction.md','protocol/RQ5_FREEZE_v1.md']:
 p=OUT/rel; files += [p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()]
lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT)) for p in sorted(set(files))]
(P/'RQ5_FREEZE_SHA256.txt').write_text('\n'.join(lines)+'\n')
for x in lines:
 h,r=x.split('  ',1); assert hashlib.sha256((OUT/r).read_bytes()).hexdigest()==h
(OUT/'STATUS.md').write_text('# RQ5 STATUS\n\nCurrent stage: COMPLETE AND FROZEN\nCompleted jobs: 210 / 210 core units\nPending jobs: 0\nRunning jobs: 0\nFailed jobs: 0 complete units\nLast validation: prediction completeness, finite values, paired bootstrap, tables, report, master append, and SHA256 verification passed.\nCurrent issue: none unresolved.\nNext action: pause; do not enter RQ6.\n')
with open(OUT/'PROGRESS_LOG.md','a') as f:f.write('\n[FINAL]\nCompleted: 210/210; bootstrap/tables/report/master/freeze/SHA256 verified.\nProblems detected: none unresolved.\nNext goal: none; RQ5 frozen.\n')
print('FREEZE_OK',len(lines),'min_boot',int(b.BootstrapValid.min()),'ci',au,ap)
