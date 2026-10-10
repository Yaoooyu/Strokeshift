#!/usr/bin/env python3
"""Generate and validate RQ5 nested target support/query splits."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
ROOT=Path(__file__).resolve().parents[4]  # QC root
OUT=Path(__file__).resolve().parents[1]
SEEDS=[20260925,20260926,20260927,20260928,20260929]
base=ROOT/'vlbd_benchmark'/'protocol'/'splits'
out=OUT/'splits'; out.mkdir(parents=True,exist_ok=True)
summary=[]
def choose_nested(pool, levels, seed):
    current=pool.reset_index(drop=True)
    result={levels[-1]:current.copy()}
    for level_i,n in enumerate(reversed(levels[:-1])):
        sss=StratifiedShuffleSplit(n_splits=1,train_size=n,random_state=seed+101*(level_i+1))
        idx,_=next(sss.split(current,current.ht_label))
        current=current.iloc[sorted(idx)].reset_index(drop=True)
        result[n]=current.copy()
    return result
for center,max_support,levels in [('B',10,[5,10]),('C',40,[5,10,20,40])]:
    full=pd.read_csv(base/f'{center}_test.csv').sort_values('case_id').reset_index(drop=True)
    for seed in SEEDS:
        sss=StratifiedShuffleSplit(n_splits=1,train_size=max_support,random_state=seed)
        support_idx,query_idx=next(sss.split(full,full.ht_label))
        pool=full.iloc[sorted(support_idx)].reset_index(drop=True)
        query=full.iloc[sorted(query_idx)].reset_index(drop=True)
        nested=choose_nested(pool,levels,seed)
        query.to_csv(out/f'{center}_seed{seed}_query.csv',index=False)
        summary.append({'center':center,'seed':seed,'role':'query','shot':0,'n':len(query),'n_positive':int(query.ht_label.sum()),'n_negative':int((1-query.ht_label).sum())})
        for shot,df in nested.items():
            df.to_csv(out/f'{center}_seed{seed}_support{shot}.csv',index=False)
            summary.append({'center':center,'seed':seed,'role':'support','shot':shot,'n':len(df),'n_positive':int(df.ht_label.sum()),'n_negative':int((1-df.ht_label).sum())})
            assert set(df.case_id).issubset(set(pool.case_id)) and set(df.case_id).isdisjoint(set(query.case_id))
            assert df.ht_label.nunique()==2, (center,seed,shot,'support lacks a class')
        assert query.ht_label.nunique()==2, (center,seed,'query lacks a class')
        for low,high in zip(levels[:-1],levels[1:]): assert set(nested[low].case_id).issubset(set(nested[high].case_id)),(center,seed,low,high)
pd.DataFrame(summary).to_csv(out/'split_label_composition.csv',index=False)
(out/'split_freeze.json').write_text(json.dumps({'seeds':SEEDS,'method':'StratifiedShuffleSplit maximum support then nested stratified subsampling','B_levels':[5,10],'C_levels':[5,10,20,40],'query_fixed_within_center_seed':True},indent=2)+'\n')
print(pd.DataFrame(summary).to_string(index=False))
