from pathlib import Path
import pandas as pd
out=Path(__file__).resolve().parents[1]; expected={'A_val':96,'B_test':41,'C_test':78}
files=sorted((out/'predictions').glob('m3care_seed*_*.csv'));assert len(files)==15,len(files)
for f in files:
 d=pd.read_csv(f);split=next(k for k in expected if f.stem.endswith('_'+k));assert len(d)==expected[split],(f,len(d));assert set(d.columns)>={'case_id','y_true','probability','threshold'};assert d.case_id.is_unique and d.probability.between(0,1).all()
r=pd.read_csv(out/'per_seed_results'/'m3care.csv');assert len(r)==105 and set(r.seed)==set(range(20260925,20260930))
print('M3CARE_SANITY_PASS files=15 predictions=1075 metrics=105')
