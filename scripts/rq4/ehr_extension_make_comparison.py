from pathlib import Path
import pandas as pd, numpy as np
O=Path(__file__).resolve().parent;B=O.parents[1];T=O/'tables';R=B/'analysis/RQ4'
e=pd.read_csv(T/'ehr_only_random_missingness.csv');c=pd.read_csv(T/'ehr_only_center_missingness.csv');rr=pd.read_csv(R/'tables/rq4_random_missingness.csv');cc=pd.read_csv(R/'tables/rq4_center_structured_missingness.csv');z=[]
def add(n,f,sc,v,cl): z.append({'model':n,'family':f,'scenario':sc,'absolute_auroc':v,'delta_from_clean':None if pd.isna(v) else v-cl})
for n in ['LogisticRegression','RandomForest','XGBoost','MLP']:
 q=e[(e.metric=='auroc')&(e.condition=='elementwise')&(e.split=='A_val')&(e.model==n)];cl=q[q.rate.astype(str)=='0.0']['mean'].iloc[0]
 for a,b in [('0.0','Clean'),('0.3','30% missingness'),('0.5','50% missingness'),('0.7','70% missingness')]:add(n,'EHR-only',b,q[q.rate.astype(str)==a]['mean'].iloc[0],cl)
 for a,b in [('B_pattern','B-pattern'),('C_pattern','C-pattern')]:add(n,'EHR-only',b,c[(c.metric=='auroc')&(c.model==n)&(c.condition==a)]['mean'].iloc[0],cl)
for n in ['generic','drfuse','m3care']:
 q=rr[(rr.metric=='auroc')&(rr.condition=='elementwise')&(rr.split=='A_val')&(rr.model==n)];cl=q[q.level.astype(str)=='0.0']['mean'].iloc[0]
 for a,b in [('0.0','Clean'),('0.3','30% missingness'),('0.5','50% missingness'),('0.7','70% missingness')]:add(n,'Multimodal',b,q[q.level.astype(str)==a]['mean'].iloc[0],cl)
 for a,b in [('B_pattern','B-pattern'),('C_pattern','C-pattern')]:add(n,'Multimodal',b,cc[(cc.metric=='auroc')&(cc.model==n)&(cc.condition==a)]['mean'].iloc[0],cl)
for sc in ['Clean','30% missingness','50% missingness','70% missingness','B-pattern','C-pattern']: z.append({'model':'source_selected_probability_fusion','family':'Multimodal','scenario':sc,'absolute_auroc':np.nan,'delta_from_clean':np.nan})
pd.DataFrame(z).to_csv(T/'ehr_only_vs_multimodal_robustness.csv',index=False)
