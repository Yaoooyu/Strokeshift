import pandas as pd
from pathlib import Path
E=Path(__file__).resolve().parents[1]
pbx=pd.read_csv(E/'tables/paired_bootstrap_vs_xgboost.csv');pbp=pd.read_csv(E/'tables/paired_bootstrap_vs_probability_fusion.csv');met=pd.read_csv(E/'tables/rq3_complete_metric_summary.csv');rank=pd.read_csv(E/'tables/model_ranking_by_center.csv');corr=pd.read_csv(E/'tables/ranking_stability_correlations.csv')
print('---VS_XGB---');print(pbx[['model','center','metric','delta','ci_lower','ci_upper','n_bootstrap_valid','interpretation']].to_string(index=False))
print('---M3_VS_PROB---');print(pbp[pbp.model=='M3Care'][['center','metric','delta','ci_lower','ci_upper','interpretation']].to_string(index=False))
print('---BRIER_ECE---');print(met[met.metric.isin(['brier','ece'])][['model','center','metric','mean','sd']].to_string(index=False))
print('---RANK---');print(rank.to_string(index=False))
print('---CORR---');print(corr.to_string(index=False))
