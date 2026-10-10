from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
EXT=Path(__file__).resolve().parents[1];T=EXT/"tables";F=EXT/"figures"
d=pd.read_csv(T/"model_ranking_by_center.csv")
plt.rcParams["font.family"]="sans-serif";plt.rcParams["font.sans-serif"]=["Arial","DejaVu Sans","Liberation Sans"];plt.rcParams["svg.fonttype"]="none";plt.rcParams["pdf.fonttype"]=42;plt.rcParams["font.size"]=8;plt.rcParams["axes.spines.top"]=False;plt.rcParams["axes.spines.right"]=False
colors={"EHR XGBoost":"#484878","Source probability fusion":"#7884B4","Generic cross-modal transformer":"#B4C0E4","DrFuse":"#E4E4F0","MedFuse":"#E4CCD8","M3Care":"#2E9E44","IRENE":"#F0C0CC"}
short={"EHR XGBoost":"XGBoost","Source probability fusion":"Prob. fusion","Generic cross-modal transformer":"Cross-modal Tx","DrFuse":"DrFuse","MedFuse":"MedFuse","M3Care":"M3Care","IRENE":"IRENE"}
centers=["A_val","B_test","C_test"]
fig,ax=plt.subplots(figsize=(7.15,3.15));x=np.arange(3)
for _,r in d.iterrows():
 ys=[r["rank_"+c] for c in centers];ax.plot(x,ys,"-o",lw=1.8,ms=4,color=colors[r.model]);ax.text(2.05,ys[-1],short[r.model],fontsize=7,va="center",color=colors[r.model])
ax.set_xticks(x,["A_val","B_test","C_test"]);ax.set_ylabel("AUROC rank (1 = best)");ax.set_ylim(7.45,.55);ax.set_xlim(-.1,2.9);ax.set_title("Source-to-external AUROC rank shifts (descriptive; 7 methods)");ax.grid(axis="y",color="#dddddd",lw=.5);fig.tight_layout()
for fmt in ["png","svg","pdf"]:fig.savefig(F/f"source_external_rank_shift.{fmt}",dpi=600,bbox_inches="tight")
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7.15,3.0),sharex=True,sharey=True)
offsets={
 "B_test":{"EHR XGBoost":(4,5),"Source probability fusion":(4,5),"Generic cross-modal transformer":(4,5),"DrFuse":(4,5),"MedFuse":(4,5),"M3Care":(4,5),"IRENE":(4,5)},
 "C_test":{"EHR XGBoost":(4,8),"Source probability fusion":(4,5),"Generic cross-modal transformer":(-4,-12),"DrFuse":(4,8),"MedFuse":(4,5),"M3Care":(4,-10),"IRENE":(4,5)}}
for ax,c in zip(axs,["B_test","C_test"]):
 for _,r in d.iterrows():
  ax.scatter(r["A_val"],r[c],s=38,color=colors[r.model],edgecolor="black",lw=.35);ax.annotate(short[r.model],(r["A_val"],r[c]),xytext=offsets[c][r.model],textcoords="offset points",fontsize=6,ha=("right" if c=="C_test" and r.model=="Generic cross-modal transformer" else "left"))
 ax.plot([.4,.9],[.4,.9],"--",color="#767676",lw=.8);ax.set_title(c);ax.set_xlabel("A_val AUROC");ax.set_xlim(.4,.9);ax.set_ylim(.4,.9);ax.grid(color="#e0e0e0",lw=.5)
axs[0].set_ylabel("External AUROC");fig.suptitle("Source performance does not determine external robustness",y=1.02,fontsize=9);fig.tight_layout()
for fmt in ["png","svg","pdf"]:fig.savefig(F/f"source_vs_external_auroc.{fmt}",dpi=600,bbox_inches="tight")
plt.close(fig)
print("FIGURE_REEXPORT_PASS")
