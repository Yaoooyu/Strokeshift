from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.spines.right":False,"axes.spines.top":False})
O=Path(__file__).resolve().parents[1];B=O.parents[1];T=O/"tables";F=O/"figures"
def sm(d,k):return d.groupby(k+["metric"]).value.agg(["mean","std","count"]).reset_index().rename(columns={"std":"sd","count":"n"})
def fig(x,n):
 for e in ["svg","pdf","png"]:plt.savefig(F/(n+"."+e),dpi=300,bbox_inches="tight")
 plt.close()
e=pd.read_csv(T/"rq4_ehr_raw_metrics.csv");m=pd.read_csv(T/"rq4_modality_raw_metrics.csv");c=pd.concat([pd.read_csv(T/"rq4_ct_expanded_raw_metrics.csv"),pd.read_csv(T/"rq4_ct_raw_metrics.csv"),pd.read_csv(T/"rq4_probability_fusion_ct_raw_metrics.csv")])
r=sm(e[e.experiment=="random_ehr"],["experiment","condition","level","model","split"]);r.to_csv(T/"rq4_random_missingness.csv",index=False)
s=sm(e[e.experiment=="center_structured"],["experiment","condition","level","model","split"]);s.to_csv(T/"rq4_center_structured_missingness.csv",index=False)
u=sm(m,["experiment","condition","level","model","split"]);u.to_csv(T/"rq4_modality_missing.csv",index=False)
v=sm(c,["experiment","condition","level","model","split"]);v.to_csv(T/"rq4_ct_degradation.csv",index=False)
pd.concat([r.assign(domain="random"),s.assign(domain="structured"),u.assign(domain="modality"),v.assign(domain="ct")],sort=False).to_csv(T/"rq4_robustness_summary.csv",index=False)
q=r[(r.metric=="auroc")&(r.split=="A_val")&(r.condition=="elementwise")]
for z in ["generic","m3care","drfuse"]:
 a=q[q.model==z].sort_values("level");plt.plot(a.level.astype(float)*100,a["mean"],marker="o",label=z)
plt.xlabel("Element-wise EHR missingness (%)");plt.ylabel("AUROC");plt.legend();fig(q,"rq4_random_missingness_curves")
q=v[(v.metric=="auroc")&(v.split=="C_test")&(v.condition=="noise")]
for z in ["source_selected_probability_fusion","generic","m3care","drfuse"]:
 a=q[q.model==z].copy();a["x"]=a.level.map({"Low":1,"Medium":2,"High":3});a=a.sort_values("x");plt.plot(a.x,a["mean"],marker="o",label=z)
plt.xticks([1,2,3],["Low","Medium","High"]);plt.xlabel("Gaussian CT noise");plt.ylabel("C-test AUROC");plt.legend(fontsize=6);fig(q,"rq4_ct_degradation_curves")
for dat,name in [(s,"rq4_center_structured_compare"),(u,"rq4_modality_missing")]:
 q=dat[(dat.metric=="auroc")&(dat.split=="A_val")];q.pivot(index="model",columns="condition",values="mean").plot.bar();plt.ylabel("AUROC");plt.xlabel("");fig(q,name)
q=v[(v.metric=="auroc")&(v.split=="C_test")&(v.condition=="noise")&(v.level=="High")].pivot(index="model",columns="condition",values="mean");plt.imshow(q,aspect="auto",cmap="viridis");plt.yticks(range(len(q)),q.index);plt.xticks([0],["High noise"]);plt.colorbar(label="AUROC");fig(q,"rq4_robustness_heatmap")
(O/"RQ4_report.md").write_text("# RQ4 report\n\nRQ4 is inference-only. All frozen RQ1-RQ3 families are indexed in `tables/rq4_baseline_coverage.csv`; only input-affected families were re-run. Probability fusion is locked to frozen RQ3 source-selected equal-average DenseNet121 plus RandomForest. RQ2 EHR estimators absent from the frozen archive were not reconstructed. All summary tables and figure source data are under `tables/`. B/C analyses are descriptive.\n")
mf=B/"master_findings.md";old=mf.read_text() if mf.exists() else "# Master findings\n";mf.write_text(old+"\n## RQ4\nInference-only robustness results are documented in `RQ4/RQ4_report.md`.\n")
(O/"STATUS.md").write_text("# RQ4 status\n\nCurrent stage: summary tables, figures, and report generated; final validation and freeze pending.\n")
