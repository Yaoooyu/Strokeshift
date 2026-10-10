#!/usr/bin/env python3
"""Reproducible protocol freeze and RQ1 heterogeneity analysis for StrokeShift-VLDB.

Run from the frozen benchmark workspace:
PYTHONPATH=vlbd_benchmark/.python_deps python3 vlbd_benchmark/run_rq1.py
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 20260925
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "vlbd_benchmark"
PROTOCOL = OUT / "protocol"
SPLITS = PROTOCOL / "splits"
RQ1 = OUT / "analysis" / "RQ1"
TABLES = RQ1 / "tables"
FIGS = RQ1 / "figures"
LOGS = OUT / "logs"
CENTERS = ["A", "B", "C"]
PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]
COLORS = {"A": "#4C78A8", "B": "#F58518", "C": "#54A24B"}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def mkdirs():
    for d in (OUT, PROTOCOL, SPLITS, RQ1, TABLES, FIGS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt_num(x, digits=2):
    return "NA" if pd.isna(x) else f"{float(x):.{digits}f}"


def fmt_p(x):
    if pd.isna(x): return "NA"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def savefig(fig, stem: str):
    for suffix, kwargs in ((".png", {"dpi": 450}), (".svg", {}), (".pdf", {})):
        fig.savefig(FIGS / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    denom = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return np.nan if denom == 0 else (np.mean(a) - np.mean(b)) / denom


def cramers_v(table):
    table = np.asarray(table, float)
    if table.size == 0 or min(table.shape) < 2 or table.sum() == 0: return np.nan, np.nan
    try:
        chi2, p, _, _ = stats.chi2_contingency(table)
    except ValueError:
        return np.nan, np.nan
    n = table.sum(); phi2 = chi2 / n
    r, k = table.shape
    denom = min(k - 1, r - 1)
    return (math.sqrt(phi2 / denom) if denom else np.nan), p


def continuous_pair(a, b):
    a, b = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(), pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2: return dict(smd=np.nan, wasserstein=np.nan, ks_stat=np.nan, ks_p=np.nan, test_p=np.nan)
    try: kw = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError: kw = np.nan
    ks = stats.ks_2samp(a, b)
    return dict(smd=smd(a,b), wasserstein=wasserstein_distance(a,b), ks_stat=ks.statistic, ks_p=ks.pvalue, test_p=kw)


def describe_cont(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if x.empty: return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, q1=np.nan, q3=np.nan)
    return dict(n=len(x), mean=x.mean(), sd=x.std(ddof=1), median=x.median(), q1=x.quantile(.25), q3=x.quantile(.75))


def frozen_write(path: Path, df: pd.DataFrame):
    cols = ["case_id", "center", "ht_label"]
    candidate = df[cols].sort_values("case_id").reset_index(drop=True)
    if path.exists():
        existing = pd.read_csv(path)[cols].sort_values("case_id").reset_index(drop=True)
        if not existing.equals(candidate):
            raise RuntimeError(f"Refusing to change frozen split: {path}")
    else:
        candidate.to_csv(path, index=False)


def audit_and_split(manifest, ehr, labels, meta):
    errors, notes = [], []
    if manifest.case_id.nunique() != 599 or len(manifest) != 599: errors.append("Manifest must contain exactly 599 unique case_id values.")
    actual = manifest.center.value_counts().to_dict()
    if actual != {"A":480,"C":78,"B":41}: errors.append(f"Unexpected center counts: {actual}")
    if manifest.ht_label.isna().any() or manifest.center.isna().any(): errors.append("Manifest has missing HT label or center.")
    if not manifest[["ct_native_available","ct_processed_available","ehr_available"]].astype(str).apply(lambda c: c.str.lower().eq("true")).all().all(): errors.append("Availability flags are not all True.")
    for col in ("ct_native_path", "ct_processed_path"):
        absent = [p for p in manifest[col] if not (ROOT / p).is_file()]
        if absent: errors.append(f"{len(absent)} invalid image paths in {col}; first: {absent[0]}")
    for frame, name in ((ehr,"EHR"),(labels,"labels")):
        if set(frame.case_id) != set(manifest.case_id): errors.append(f"{name} case IDs do not match manifest.")
    if len(meta) != 1198 or meta.groupby("case_id").size().nunique() != 1 or meta.groupby("case_id").size().iloc[0] != 2:
        errors.append("Imaging metadata does not contain exactly native and processed rows for each case.")
    if set(meta.representation.dropna()) != {"native","processed"}: errors.append("Unexpected image representations.")
    notes.extend([
        "The manifest is the sole case-entry point. No source 607-case directory was scanned for eligibility.",
        "CT_native and CT_processed are two representations of the same case and are analysed paired by case_id; they are never treated as independent samples.",
        "The six unresolved duplicate-image/EHR cases and two incomplete-coverage cases remain absent from this 599-case frozen cohort.",
    ])
    a = manifest.query("center == 'A'").copy().reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(a, a.ht_label))
    frozen_write(SPLITS / "A_train.csv", a.iloc[tr_idx])
    frozen_write(SPLITS / "A_val.csv", a.iloc[va_idx])
    frozen_write(SPLITS / "B_test.csv", manifest.query("center == 'B'"))
    frozen_write(SPLITS / "C_test.csv", manifest.query("center == 'C'"))
    rows=[]
    for name in ("A_train","A_val","B_test","C_test"):
        x=pd.read_csv(SPLITS/f"{name}.csv"); rows.append(dict(split=name,n=len(x),positive=int(x.ht_label.sum()),negative=int((1-x.ht_label).sum()),prevalence=x.ht_label.mean(),sha256=sha256(SPLITS/f"{name}.csv")))
    pd.DataFrame(rows).to_csv(TABLES/"frozen_split_summary.csv",index=False)
    report = ["# Data integrity report", "", "## Result", "", "**PASS**" if not errors else "**FAIL — no data were altered.**", "", "## Checks"]
    report += [f"- Frozen manifest rows / unique case IDs: {len(manifest)} / {manifest.case_id.nunique()}", f"- Centers: A={actual.get('A',0)}, B={actual.get('B',0)}, C={actual.get('C',0)}", f"- All cases have native CT, processed CT, EHR, and HT label: {'yes' if not errors else 'see exceptions'}", f"- Valid native and processed paths checked against the QC workspace: yes", ""]
    report += ["## Notes"] + [f"- {x}" for x in notes] + ["", "## Exceptions"] + ([f"- {x}" for x in errors] if errors else ["- None."])
    (PROTOCOL/"data_integrity_report.md").write_text("\n".join(report)+"\n")
    if errors: raise RuntimeError("; ".join(errors))
    return pd.DataFrame(rows)


def cohort_stats(manifest):
    records=[]
    for c in CENTERS:
        x=manifest.query("center == @c").ht_label.astype(int)
        records.append(dict(center=c,n=len(x),ht_positive=int(x.sum()),ht_negative=int((1-x).sum()),ht_prevalence=x.mean()))
    table=pd.DataFrame(records); table.to_csv(TABLES/"cohort_overview.csv",index=False)
    ct=pd.crosstab(manifest.center, manifest.ht_label).reindex(CENTERS, fill_value=0)
    chi2,p,_,_=stats.chi2_contingency(ct); v,_=cramers_v(ct.to_numpy())
    (TABLES/"ht_prevalence_test.csv").write_text(f"test,statistic,p_value,cramers_v\nchi_square,{chi2},{p},{v}\n")
    fig,ax=plt.subplots(figsize=(5.1,3.2)); bars=ax.bar(table.center,table.ht_prevalence*100,color=[COLORS[c] for c in CENTERS]); ax.set_ylabel("HT prevalence (%)"); ax.set_ylim(0,max(5,table.ht_prevalence.max()*130)); ax.set_title("Cohort size and HT prevalence")
    for b,r in zip(bars,table.itertuples()): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.5,f"{r.ht_prevalence:.1%}\n(n={r.n})",ha="center",va="bottom")
    ax.text(.02,.96,f"Global χ² p={fmt_p(p)}; Cramér’s V={fmt_num(v)}",transform=ax.transAxes,va="top")
    savefig(fig,"cohort_overview")
    return table, dict(chi2=chi2,p=p,v=v)


def classify_features(ehr):
    excluded={"case_id","patient_id","center","potential_post_treatment_leakage","leakage_flagged_columns"}
    features=[c for c in ehr.columns if c not in excluded and not ehr[c].isna().all()]
    cont=[]; cat=[]
    for c in features:
        numeric=pd.to_numeric(ehr[c],errors="coerce")
        if numeric.notna().mean()>.95 and numeric.nunique(dropna=True)>6: cont.append(c)
        else: cat.append(c)
    return features,cont,cat


def ehr_analysis(ehr):
    features, cont, cat=classify_features(ehr)
    pd.DataFrame([{ "variable":x,"type":"continuous" if x in cont else "categorical", "leakage_flagged": x in {"stent_implantation","mechanical_thrombectomy"}} for x in features]).to_csv(TABLES/"ehr_variable_dictionary.csv",index=False)
    desc=[]; shifts=[]; miss=[]; patient=ehr[["case_id","center"]].copy(); patient["n_missing_variables"]=ehr[features].isna().sum(axis=1)
    for f in features:
        missrow={"variable":f}
        for c in CENTERS: missrow[f"missing_rate_{c}"]=ehr.loc[ehr.center.eq(c),f].isna().mean()
        for a,b in PAIRS: missrow[f"abs_missing_diff_{a}_{b}"]=abs(missrow[f"missing_rate_{a}"]-missrow[f"missing_rate_{b}"])
        miss.append(missrow)
        if f in cont:
            numeric=pd.to_numeric(ehr[f],errors="coerce")
            try: gp=stats.kruskal(*[numeric[ehr.center.eq(c)].dropna() for c in CENTERS]).pvalue
            except ValueError: gp=np.nan
            for c in CENTERS:
                d=describe_cont(numeric[ehr.center.eq(c)]); desc.append(dict(variable=f,type="continuous",center=c,missing_rate=missrow[f"missing_rate_{c}"],mean=d["mean"],sd=d["sd"],median=d["median"],q1=d["q1"],q3=d["q3"],n=d["n"],category=np.nan,n_category=np.nan,percent=np.nan))
            for a,b in PAIRS:
                res=continuous_pair(numeric[ehr.center.eq(a)],numeric[ehr.center.eq(b)])
                shifts.append(dict(variable=f,type="continuous",comparison=f"{a}-{b}",global_test="Kruskal-Wallis",global_p=gp,**res,cramers_v=np.nan,js_divergence=np.nan,abs_prevalence_difference=np.nan,ranking_metric=abs(res["smd"]) if pd.notna(res["smd"]) else np.nan))
        else:
            vals=ehr[f].fillna("<missing>").astype(str); levels=sorted(vals.unique())
            tab=pd.crosstab(ehr.center,vals).reindex(index=CENTERS,columns=levels,fill_value=0); gv,gp=cramers_v(tab)
            for c in CENTERS:
                n=(ehr.center==c).sum()
                for lev in levels:
                    count=int(tab.loc[c,lev]); desc.append(dict(variable=f,type="categorical",center=c,missing_rate=missrow[f"missing_rate_{c}"],mean=np.nan,sd=np.nan,median=np.nan,q1=np.nan,q3=np.nan,n=n,category=lev,n_category=count,percent=count/n if n else np.nan))
            for a,b in PAIRS:
                pairtab=tab.loc[[a,b]]; pv,pp=cramers_v(pairtab)
                pa=(pairtab.loc[a]/pairtab.loc[a].sum()).to_numpy(); pb=(pairtab.loc[b]/pairtab.loc[b].sum()).to_numpy()
                js=float(jensenshannon(pa,pb,base=2)**2) if len(pa)>1 else 0.0
                shifts.append(dict(variable=f,type="categorical",comparison=f"{a}-{b}",global_test="chi-square",global_p=gp,smd=np.nan,wasserstein=np.nan,ks_stat=np.nan,ks_p=np.nan,test_p=pp,cramers_v=pv,js_divergence=js,abs_prevalence_difference=float(np.max(np.abs(pa-pb))),ranking_metric=pv))
    desc=pd.DataFrame(desc); shifts=pd.DataFrame(shifts); missing=pd.DataFrame(miss)
    desc.to_csv(TABLES/"ehr_descriptive_statistics.csv",index=False); shifts.to_csv(RQ1/"ehr_shift_summary.csv",index=False); shifts.to_csv(TABLES/"ehr_shift_summary.csv",index=False); missing.to_csv(RQ1/"ehr_missingness.csv",index=False); missing.to_csv(TABLES/"ehr_missingness.csv",index=False); patient.to_csv(TABLES/"ehr_patient_missingness.csv",index=False)
    # Heatmaps
    rank=shifts.groupby("variable").ranking_metric.max().sort_values(ascending=False)
    chosen=list(rank.head(min(20,len(rank))).index)
    mat=shifts[shifts.variable.isin(chosen)].pivot(index="variable",columns="comparison",values="ranking_metric").reindex(chosen)
    fig,ax=plt.subplots(figsize=(5.4,max(3.3,.32*len(mat)+1))); im=ax.imshow(mat.fillna(0),cmap="magma",aspect="auto"); ax.set_xticks(range(3),mat.columns); ax.set_yticks(range(len(mat)),mat.index); ax.set_title("EHR cross-center shift (feature-specific effect metric)"); fig.colorbar(im,ax=ax,label="|SMD| (continuous) / Cramér’s V (categorical)"); savefig(fig,"ehr_feature_shift_heatmap")
    mm=missing.set_index("variable")[[f"missing_rate_{c}" for c in CENTERS]].reindex(features)
    fig,ax=plt.subplots(figsize=(5.4,max(3.3,.28*len(mm)+1))); im=ax.imshow(mm.fillna(0)*100,cmap="Blues",aspect="auto",vmin=0,vmax=max(1,(mm.max().max()*100))); ax.set_xticks(range(3),CENTERS); ax.set_yticks(range(len(mm)),mm.index); ax.set_title("EHR missingness by center"); fig.colorbar(im,ax=ax,label="Missing (%)"); savefig(fig,"ehr_missingness_heatmap")
    fig,ax=plt.subplots(figsize=(6.2,3.3)); pm=patient.groupby("center").n_missing_variables.mean().reindex(CENTERS); ax.bar(CENTERS,pm,color=[COLORS[c] for c in CENTERS]); ax.set_ylabel("Mean missing EHR variables per patient"); ax.set_title("Per-patient EHR missingness"); savefig(fig,"ehr_missingness_per_patient")
    # Selected distributions: highest ranked continuous and categorical
    top_cont=[x for x in rank.index if x in cont][:4]; top_cat=[x for x in rank.index if x in cat][:4]
    panels=top_cont+top_cat
    if panels:
        fig,axes=plt.subplots(2,4,figsize=(11,5.4)); axes=axes.ravel()
        for ax,f in zip(axes,panels):
            if f in cont:
                for c in CENTERS: ax.hist(pd.to_numeric(ehr.loc[ehr.center.eq(c),f],errors="coerce").dropna(),bins=18,density=True,histtype="step",lw=1.5,color=COLORS[c],label=c)
                ax.set_title(f); ax.legend(title="Center",fontsize=6)
            else:
                tab=pd.crosstab(ehr.center,ehr[f].fillna("<missing>").astype(str),normalize="index").reindex(CENTERS,fill_value=0); tab.plot(kind="bar",stacked=True,ax=ax,color=plt.cm.tab20.colors[:len(tab.columns)],legend=False); ax.set_title(f); ax.set_ylabel("Proportion")
        for ax in axes[len(panels):]: ax.axis("off")
        fig.suptitle("Selected EHR distributions by center",y=1.01); savefig(fig,"selected_ehr_distributions")
    return features,cont,cat,shifts,missing


def parse_shape(x):
    try: return tuple(ast.literal_eval(x))
    except Exception: return (np.nan,np.nan,np.nan)


def ct_pair_stats(df, variables, prefix):
    rows=[]
    for v in variables:
        try: gp=stats.kruskal(*[df.loc[df.center.eq(c),v].dropna() for c in CENTERS]).pvalue
        except ValueError: gp=np.nan
        for c in CENTERS:
            d=describe_cont(df.loc[df.center.eq(c),v]); rows.append(dict(variable=v,section=prefix,record_type="center_summary",center=c,comparison=np.nan,global_p=gp,mean=d["mean"],sd=d["sd"],median=d["median"],q1=d["q1"],q3=d["q3"],n=d["n"],smd=np.nan,wasserstein=np.nan,ks_stat=np.nan,ks_p=np.nan))
        for a,b in PAIRS:
            res=continuous_pair(df.loc[df.center.eq(a),v],df.loc[df.center.eq(b),v]); rows.append(dict(variable=v,section=prefix,record_type="pairwise",center=np.nan,comparison=f"{a}-{b}",global_p=gp,mean=np.nan,sd=np.nan,median=np.nan,q1=np.nan,q3=np.nan,n=np.nan,**res))
    return pd.DataFrame(rows)


def ct_analysis(meta):
    native=meta.query("representation == 'native'").copy(); processed=meta.query("representation == 'processed'").copy()
    for d in (native,processed):
        shapes=d["shape"].map(parse_shape); d[["shape_x","shape_y","shape_z"]]=pd.DataFrame(shapes.tolist(),index=d.index)
        for x in ("spacing_x","spacing_y","spacing_z","num_slices","slice_thickness","intensity_min","intensity_max"):
            d[x]=pd.to_numeric(d[x],errors="coerce")
        d["axial_coverage_mm"]=(d.num_slices-1)*d.spacing_z
        d["inplane_resolution_mm"]=(d.spacing_x*d.spacing_y)**.5
        d["voxel_count"]=d.shape_x*d.shape_y*d.shape_z
    vars=["shape_x","shape_y","shape_z","num_slices","spacing_x","spacing_y","spacing_z","slice_thickness","axial_coverage_mm","inplane_resolution_mm","voxel_count"]
    acq=ct_pair_stats(native,vars,"native_acquisition")
    acq.to_csv(RQ1/"ct_acquisition_summary.csv",index=False); acq.to_csv(TABLES/"ct_acquisition_summary.csv",index=False)
    native.to_csv(TABLES/"native_ct_per_case_metadata.csv",index=False)
    # Scanner variables are described even if absent.
    scanner=[]
    for v in ("scanner_manufacturer","scanner_model","series_description","protocol_name","reconstruction_kernel"):
        for c in CENTERS:
            x=native.loc[native.center.eq(c),v].fillna("").replace("", "<not available>")
            for level,n in x.value_counts().items(): scanner.append(dict(variable=v,center=c,level=level,n=n,percent=n/len(x)))
    pd.DataFrame(scanner).to_csv(TABLES/"ct_scanner_metadata_distribution.csv",index=False)
    # Acquisition distribution panels (a note is drawn when no nonmissing thickness is present).
    plotvars=[("slice_thickness","Slice thickness (mm)"),("spacing_z","Z spacing (mm)"),("num_slices","Number of slices"),("inplane_resolution_mm","In-plane resolution (mm)"),("axial_coverage_mm","Axial coverage (mm)")]
    fig,axes=plt.subplots(2,3,figsize=(10,6)); axes=axes.ravel()
    for ax,(v,label) in zip(axes,plotvars):
        if native[v].notna().sum()==0:
            ax.text(.5,.5,"Not available in\ntransfer metadata",ha="center",va="center"); ax.set_title(label); ax.set_axis_off(); continue
        for c in CENTERS: ax.hist(native.loc[native.center.eq(c),v].dropna(),bins=18,histtype="step",density=True,lw=1.5,color=COLORS[c],label=c)
        ax.set_title(label); ax.legend(fontsize=6)
    axes[-1].axis("off"); fig.suptitle("Native CT acquisition distributions",y=.99); savefig(fig,"native_ct_acquisition_distributions")
    # Scanner chart: only manufacturer to avoid a panel of blank fields.
    tab=pd.crosstab(native.center,native.scanner_manufacturer.fillna("").replace("","<not available>")).reindex(CENTERS,fill_value=0)
    fig,ax=plt.subplots(figsize=(6,3.5)); tab.plot(kind="bar",stacked=True,ax=ax,color=plt.cm.Pastel1.colors[:len(tab.columns)]); ax.set_ylabel("Cases"); ax.set_xlabel("Center"); ax.set_title("Native CT scanner manufacturer metadata"); ax.legend(title="Manufacturer",fontsize=7); savefig(fig,"native_ct_scanner_distribution")
    # Processed effect summary.
    rows=[]
    for rep,d in (("native",native),("processed",processed)):
        for v in vars+["intensity_min","intensity_max"]:
            for c in CENTERS:
                z=describe_cont(d.loc[d.center.eq(c),v]); rows.append(dict(representation=rep,variable=v,center=c,**z))
    pd.DataFrame(rows).to_csv(TABLES/"processed_ct_effect_summary.csv",index=False)
    return native,processed,acq


def intensity_analysis(native):
    records=[]; n=len(native)
    for i,r in enumerate(native.itertuples(index=False),1):
        path=ROOT/getattr(r,"path")
        vol=np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32,copy=False)
        finite=vol[np.isfinite(vol)]
        fg=finite[finite>-500]; bw=finite[(finite>=-50)&(finite<=100)]
        def one(x,prefix):
            if len(x)==0: return {prefix+k:np.nan for k in ("n","mean","sd","p01","p05","p50","p95","p99")}
            return {prefix+"n":len(x),prefix+"mean":float(x.mean()),prefix+"sd":float(x.std()),prefix+"p01":float(np.quantile(x,.01)),prefix+"p05":float(np.quantile(x,.05)),prefix+"p50":float(np.quantile(x,.5)),prefix+"p95":float(np.quantile(x,.95)),prefix+"p99":float(np.quantile(x,.99))}
        records.append(dict(case_id=r.case_id,center=r.center,**one(finite,"whole_"),**one(fg,"foreground_nonair_"),**one(bw,"brain_window_")))
        if i%50==0: print(f"native intensity QC: {i}/{n}",flush=True)
    d=pd.DataFrame(records); d.to_csv(TABLES/"native_ct_intensity_per_case.csv",index=False)
    summary=ct_pair_stats(d,["whole_mean","whole_sd","whole_p05","whole_p50","whole_p95","foreground_nonair_mean","brain_window_mean","brain_window_sd"],"native_intensity")
    summary.to_csv(TABLES/"native_ct_intensity_summary.csv",index=False)
    return d,summary


def heterogeneity_map(ehr_shifts, acq):
    er=ehr_shifts.groupby("variable").ranking_metric.max().sort_values(ascending=False).head(12).index.tolist()
    e=ehr_shifts[ehr_shifts.variable.isin(er)].copy(); e["row"]="EHR: "+e.variable
    a=acq[acq.record_type.eq("pairwise")].copy(); a["row"]="CT: "+a.variable; a["ranking_metric"]=a.smd.abs()
    x=pd.concat([e[["row","comparison","ranking_metric"]],a[["row","comparison","ranking_metric"]]])
    mat=x.pivot_table(index="row",columns="comparison",values="ranking_metric",aggfunc="first").fillna(0)
    fig,ax=plt.subplots(figsize=(5.7,max(4,.32*len(mat)+1))); im=ax.imshow(mat,cmap="magma",aspect="auto"); ax.set_xticks(range(3),mat.columns); ax.set_yticks(range(len(mat)),mat.index); ax.set_title("Cross-center heterogeneity map"); fig.colorbar(im,ax=ax,label="|SMD| or Cramér’s V"); savefig(fig,"cross_center_heterogeneity_map")
    mat.to_csv(TABLES/"cross_center_heterogeneity_map.csv")


def write_protocol(manifest, split_summary, cohort, ht_test, features, cont, cat, native):
    exclusions=pd.read_csv(ROOT/"excluded_cases.csv")
    md=f"""# StrokeShift-VLDB benchmark protocol v1

## Frozen cohort

- Sole case entry point: `benchmark_manifest.csv`.
- Cohort: 599 cases (A=480, B=41, C=78). The original transfer remains immutable.
- Eight excluded cases: six from three unresolved identical-CT/EHR-linkage pairs and two insufficient whole-head NCCT-coverage cases. See `excluded_cases.csv`.
- Each case has one native and one processed CT representation; these are paired representations, not separate samples.

## Split freeze

- Source development center: A (n=480); stratified 80/20 train/validation split, seed `{SEED}`.
- A_train={int(split_summary.loc[split_summary.split.eq('A_train'),'n'].iloc[0])}; A_val={int(split_summary.loc[split_summary.split.eq('A_val'),'n'].iloc[0])}.
- B (n=41) and C (n=78) are untouched external test centers. No few-shot support/query split is created in this protocol.
- The split CSV SHA256 values are recorded in `analysis/RQ1/tables/frozen_split_summary.csv`; reruns refuse a changed frozen split.

## RQ1 analysis scope

RQ1 analyses center heterogeneity in native CT, structured EHR, and HT prevalence. No HT prediction model is trained. Potential post-treatment EHR fields (`stent_implantation`, `mechanical_thrombectomy`) are flagged in descriptive tables and are never used for a prediction model.

## CT representations

- `CT_native`: acquisition and intensity heterogeneity analyses.
- `CT_processed`: paired auxiliary analysis of the effect of provided preprocessing on image shape, spacing, and recorded range.
"""
    (PROTOCOL/"benchmark_protocol_v1.md").write_text(md)
    contract="""# Figure contract — RQ1

Core conclusion: Pre-treatment clinical data may show institution-specific heterogeneity in structured EHR, missingness, native CT acquisition, and HT prevalence.

Archetype: quantitative grid. Backend: Python. Outputs: SVG/PDF plus 450-dpi PNG. Hero evidence: cross-center heterogeneity map and native CT acquisition distributions. Supporting evidence: cohort prevalence, EHR shifts/missingness, scanner metadata, and selected distributions. Statistics: Kruskal-Wallis and pairwise Mann-Whitney U for continuous variables; chi-square/Cramér’s V for categorical variables; standardized mean difference, Wasserstein distance, and KS statistic where applicable. Reviewer risks: B is small (n=41), scanner fields may be unavailable, multiple testing is descriptive rather than confirmatory, and no causal inference is made.
"""
    (RQ1/"figure_contract.md").write_text(contract)


def write_report(cohort, ht, shifts, missing, native, acq, intensity):
    rank=shifts.groupby("variable").ranking_metric.max().sort_values(ascending=False)
    top=", ".join(rank.head(8).index)
    ct=acq[acq.record_type.eq("pairwise")].copy(); ct["abs_smd"]=ct.smd.abs(); ct_rank=ct.groupby("variable").abs_smd.max().sort_values(ascending=False)
    pair=ct.groupby("comparison").abs_smd.mean().sort_values(ascending=False)
    zero_missing=bool((missing.filter(like="missing_rate_").fillna(0).to_numpy()==0).all())
    manufacturers=native.scanner_manufacturer.replace("",np.nan).notna().sum()
    r=f"""# RQ1 — Cross-Center Data Heterogeneity

## Scope and methods

This report analyses only the frozen 599-case benchmark manifest. It does not train a hemorrhage-prediction model. Continuous variables use Kruskal-Wallis global tests and pairwise Mann-Whitney U / KS tests, with standardized mean differences (SMD) and Wasserstein distances. Categorical variables use chi-square and Cramér’s V. Results are descriptive; effect sizes are reported alongside p-values.

## Direct answers

1. **HT prevalence:** A={cohort.loc[cohort.center.eq('A'),'ht_prevalence'].iloc[0]:.1%}, B={cohort.loc[cohort.center.eq('B'),'ht_prevalence'].iloc[0]:.1%}, C={cohort.loc[cohort.center.eq('C'),'ht_prevalence'].iloc[0]:.1%}. Global chi-square p={fmt_p(ht['p'])}, Cramér’s V={fmt_num(ht['v'])}; interpret the p-value and small-center estimates cautiously.
2. **Largest EHR shifts:** by the pre-specified feature-specific effect-size ranking: {top}. Exact pairwise metrics are in `ehr_shift_summary.csv`.
3. **EHR missingness:** {'all analysed EHR fields are complete in this transferred table' if zero_missing else 'missingness differs by field/center; see the heatmap and table'}. The transfer documentation states values and missingness were preserved without new imputation; this analysis does not treat imputed values as original observations.
4. **CT acquisition differences:** leading native-CT numeric differences by maximum |SMD| are {', '.join(ct_rank.head(6).index)}. See `ct_acquisition_summary.csv` and distributions.
5. **Largest CT acquisition pairwise shift:** {pair.index[0] if len(pair) else 'not estimable'} by mean |SMD| across available acquisition variables; this is a descriptive summary, not a single validated composite endpoint.
6. **Preprocessing effect:** processed CT records have the supplied standardized 128×128×128 / 1-mm representation and bounded normalized intensity range where recorded, reducing original resolution/shape/range variation.
7. **Residual processed heterogeneity:** center-specific EHR differences remain by construction; processed-image metadata should be interpreted from `processed_ct_effect_summary.csv` rather than as independent samples.
8. **CT versus EHR:** compare effect-size tables and the heterogeneity map; these metrics are variable-specific and should not be treated as a single cross-modality score.
9. **Stable empirical findings:** frozen cohort counts, excluded-case provenance, HT prevalence table, native-CT geometry, and per-case summary-statistic comparisons are reproducible from the supplied script.
10. **Cautions:** Center B has n=41; sparse categorical levels and unavailable scanner metadata ({manufacturers}/{len(native)} native records have nonblank manufacturer) constrain inference. This is heterogeneity description, not causal evidence.

## Native CT intensity

Native CT recorded ranges include CT-like negative air values and high-density values. Per-case (not pooled-voxel) summaries for whole volume, non-air foreground (>−500 HU), and brain window [−50,100] HU are in `native_ct_intensity_per_case.csv`; center comparisons are in `native_ct_intensity_summary.csv`.
"""
    (RQ1/"RQ1_report.md").write_text(r)


def figure_qc():
    from PIL import Image
    rows=[]
    for p in sorted(FIGS.glob("*.png")):
        im=Image.open(p); arr=np.asarray(im.convert("L")); rows.append(dict(file=p.name,width_px=im.width,height_px=im.height,nonwhite_fraction=float((arr<250).mean()),valid=bool(im.width>300 and im.height>200 and (arr<250).mean()>.01)))
    d=pd.DataFrame(rows); d.to_csv(TABLES/"figure_qc.csv",index=False)
    if not d.valid.all(): raise RuntimeError("Figure QC failed")


def main():
    mkdirs()
    manifest=pd.read_csv(ROOT/"benchmark_manifest.csv"); ehr=pd.read_csv(ROOT/"metadata/ehr.csv"); labels=pd.read_csv(ROOT/"metadata/labels.csv"); meta=pd.read_csv(ROOT/"metadata/imaging_metadata.csv")
    split_summary=audit_and_split(manifest,ehr,labels,meta)
    cohort,ht=cohort_stats(manifest)
    features,cont,cat,shifts,missing=ehr_analysis(ehr)
    native,processed,acq=ct_analysis(meta)
    intensity,intensity_summary=intensity_analysis(native)
    heterogeneity_map(shifts,acq)
    write_protocol(manifest,split_summary,cohort,ht,features,cont,cat,native)
    write_report(cohort,ht,shifts,missing,native,acq,intensity)
    figure_qc()
    (LOGS/"run_metadata.json").write_text(json.dumps({"seed":SEED,"manifest_sha256":sha256(ROOT/"benchmark_manifest.csv"),"n_cases":599,"status":"complete"},indent=2)+"\n")
    print("RQ1 complete",flush=True)

if __name__ == "__main__": main()
