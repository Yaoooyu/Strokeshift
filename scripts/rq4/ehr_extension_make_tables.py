from pathlib import Path
import pandas as pd, numpy as np
O=Path(__file__).resolve().parent; T=O/'tables'; R=O.parents[1]/'RQ4'
r=pd.read_csv(T/'ehr_only_random_missingness.csv'); c=pd.read_csv(T/'ehr_only_center_missingness.csv')
q=r[(r.metric=='auroc')&(r.split=='A_val')]; b=q[q.rate.astype(str)=='0.0'].set_index('model')['mean']; out=[]
for n in b.index:
 d={'Model':n,'Clean AUROC':b[n]}
 for typ,lab in [('elementwise','Element-wise'),('featurewise','Feature-wise')]:
  for rate in ['0.1','0.3','0.5','0.7']:
   v=q[(q.model==n)&(q.condition==typ)&(q.rate.astype(str)==rate)]['mean'].iloc[0]; d[lab+' '+rate+' AUROC']=v; d[lab+' '+rate+' Drop']=v-b[n]
 for pat in ['B_pattern','C_pattern']:
  v=c[(c.metric=='auroc')&(c.model==n)&(c.condition==pat)]['mean'].iloc[0]; d[pat+' AUROC']=v; d[pat+' Drop']=v-b[n]
 out.append(d)
pd.DataFrame(out).to_csv(T/'ehr_only_missingness_summary.csv',index=False)
mm=pd.read_csv(R/'tables/rq4_random_missingness.csv'); mm=mm[(mm.metric=='auroc')&(mm.condition=='elementwise')&(mm.split=='A_val')&(mm.model.isin(['generic','m3care','drfuse']))].rename(columns={'level':'rate'}); mm['family']='multimodal'
ee=q.copy();ee['family']='ehr_only';z=pd.concat([ee,mm],sort=False);cl=z[z.rate.astype(str)=='0.0'].set_index('model')['mean'];z['delta_from_clean']=z.apply(lambda a:a['mean']-cl.get(a['model'],np.nan),axis=1);z[z.rate.astype(str).isin(['0.0','0.3','0.5','0.7'])].to_csv(T/'ehr_only_vs_multimodal_robustness.csv',index=False)
