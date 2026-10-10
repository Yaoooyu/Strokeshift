#!/usr/bin/env python3
"""RQ3 source-only representation fusion; B/C are evaluation-only."""
from pathlib import Path
import json,random
import numpy as np,pandas as pd,nibabel as nib
import torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,recall_score,confusion_matrix,brier_score_loss
ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'vlbd_benchmark'/'analysis'/'RQ3'
PRED=OUT/'representation_fusion'/'predictions'; CKPT=OUT/'representation_fusion'/'checkpoints'
PRED.mkdir(parents=True,exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True); (OUT/'tables').mkdir(exist_ok=True)
SEEDS=[20260925+i for i in range(5)]; DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; EPOCHS=8; BATCH=6
m=pd.read_csv(ROOT/'benchmark_manifest.csv').set_index('case_id')
splits={k:pd.read_csv(ROOT/'vlbd_benchmark'/'protocol'/'splits'/f'{k}.csv').case_id.tolist() for k in ['A_train','A_val','B_test','C_test']}
e=pd.read_csv(ROOT/'metadata'/'ehr.csv').set_index('case_id')
drop={'patient_id','center','potential_post_treatment_leakage','leakage_flagged_columns'}
features=[c for c in e.columns if c not in drop]
X=e[features].apply(pd.to_numeric,errors='coerce')
prep=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())])
ehr={'A_train':prep.fit_transform(X.loc[splits['A_train']]).astype('float32')}
for sp in ['A_val','B_test','C_test']: ehr[sp]=prep.transform(X.loc[splits[sp]]).astype('float32')
cache={}
def cache_ct(ids):
 for i,cid in enumerate(ids,1):
  if cid not in cache: cache[cid]=torch.from_numpy(np.asanyarray(nib.load(str(ROOT/m.loc[cid,'ct_processed_path'])).dataobj).astype('float32'))[None]
  if i%100==0: print('cached',i,flush=True)
cache_ct(splits['A_train']+splits['A_val'])
class DS(Dataset):
 def __init__(s,sp,aug=False):s.ids=splits[sp];s.sp=sp;s.aug=aug;s.pos={x:i for i,x in enumerate(s.ids)}
 def __len__(s):return len(s.ids)
 def __getitem__(s,i):
  cid=s.ids[i]; x=cache[cid].clone()
  if s.aug:
   for d in [1,2,3]:
    if torch.rand(())<.5:x=torch.flip(x,[d])
  return x,torch.from_numpy(ehr[s.sp][i]),torch.tensor(int(m.loc[cid,'ht_label'])),cid
class CT(nn.Module):
 def __init__(s):
  super().__init__();s.net=nn.Sequential(nn.Conv3d(1,16,3,2,1),nn.BatchNorm3d(16),nn.ReLU(),nn.Conv3d(16,32,3,2,1),nn.BatchNorm3d(32),nn.ReLU(),nn.Conv3d(32,64,3,2,1),nn.BatchNorm3d(64),nn.ReLU(),nn.AdaptiveAvgPool3d(1),nn.Flatten())
 def forward(s,x):return s.net(x)
class EHR(nn.Module):
 def __init__(s,n):super().__init__();s.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,64),nn.ReLU())
 def forward(s,x):return s.net(x)
class Fusion(nn.Module):
 def __init__(s,n,kind):
  super().__init__();s.ct=CT();s.ehr=EHR(n);s.kind=kind
  if kind=='concat_mlp':s.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,2))
  elif kind=='gmu':s.gate=nn.Linear(128,64);s.head=nn.Linear(64,2)
  elif kind=='attention':s.att=nn.MultiheadAttention(64,4,batch_first=True);s.head=nn.Linear(64,2)
  else:s.pos=nn.Parameter(torch.zeros(1,2,64));s.trans=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,batch_first=True,dropout=.1),num_layers=2);s.head=nn.Linear(64,2)
 def forward(s,x,z):
  a,b=s.ct(x),s.ehr(z)
  if s.kind=='concat_mlp':return s.head(torch.cat([a,b],1))
  if s.kind=='gmu':
   g=torch.sigmoid(s.gate(torch.cat([a,b],1)));return s.head(g*a+(1-g)*b)
  tok=torch.stack([a,b],1)
  if s.kind=='attention':q,_=s.att(tok,tok,tok);return s.head(q.mean(1))
  return s.head(s.trans(tok+s.pos).mean(1))
def predict(net,ld):
 net.eval();out=[]
 with torch.no_grad():
  for x,z,y,c in ld:
   p=torch.softmax(net(x.to(DEVICE),z.to(DEVICE)),1)[:,1].cpu().numpy();out.extend(zip(c,y.numpy(),p))
 return pd.DataFrame(out,columns=['case_id','y_true','probability'])
def ece(y,p):
 ans=0.;edges=np.linspace(0,1,11)
 for i in range(10):
  q=(p>=edges[i])&(p<(edges[i+1] if i<9 else 1.00001))
  if q.any():ans+=q.mean()*abs(p[q].mean()-y[q].mean())
 return ans
def met(y,p,t):
 q=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
 return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'f1':f1_score(y,q),'balanced_accuracy':balanced_accuracy_score(y,q),'sensitivity':recall_score(y,q),'specificity':tn/(tn+fp),'brier':brier_score_loss(y,p),'ece':ece(y,p)}
def threshold(y,p):
 return float(max(np.linspace(.05,.95,181),key=lambda t:f1_score(y,p>=t)))
rows=[]; methods=['concat_mlp','gmu','attention','cross_modal_transformer']
for seed in SEEDS:
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 tr=DataLoader(DS('A_train',True),batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=True)
 va=DataLoader(DS('A_val'),batch_size=BATCH,num_workers=2,pin_memory=True)
 pos=int(m.loc[splits['A_train'],'ht_label'].sum());lossfn=nn.CrossEntropyLoss(weight=torch.tensor([1.,(384-pos)/pos],device=DEVICE))
 for method in methods:
  net=Fusion(ehr['A_train'].shape[1],method).to(DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4);best=(-1,None)
  for ep in range(EPOCHS):
   net.train()
   for x,z,y,_ in tr:opt.zero_grad();lossfn(net(x.to(DEVICE),z.to(DEVICE)),y.to(DEVICE)).backward();opt.step()
   d=predict(net,va);score=roc_auc_score(d.y_true,d.probability)
   if score>best[0]:best=(score,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
  net.load_state_dict(best[1]);torch.save(net.state_dict(),CKPT/f'{method}_seed{seed}.pt')
  for sp in ['A_val','B_test','C_test']:
   if sp in ['B_test','C_test']: cache_ct(splits[sp])
   ld=va if sp=='A_val' else DataLoader(DS(sp),batch_size=BATCH,num_workers=2,pin_memory=True)
   d=predict(net,ld); t=threshold(d.y_true.to_numpy(),d.probability.to_numpy()) if sp=='A_val' else threshold(predict(net,va).y_true.to_numpy(),predict(net,va).probability.to_numpy())
   d['threshold']=t;d.to_csv(PRED/f'{method}_seed{seed}_{sp}.csv',index=False)
   for k,v in met(d.y_true.to_numpy(),d.probability.to_numpy(),t).items():rows.append({'model':method,'seed':seed,'split':sp,'metric':k,'value':v,'threshold':t})
  del net;torch.cuda.empty_cache();print('done',method,seed,flush=True)
r=pd.DataFrame(rows);r.to_csv(OUT/'rq3_representation_per_seed_results.csv',index=False)
r.groupby(['model','split','metric']).agg(mean=('value','mean'),sd=('value','std'),n_seeds=('seed','nunique')).reset_index().to_csv(OUT/'tables'/'rq3_representation_summary.csv',index=False)
json.dump({'training':'A_train only','selection':'best AUROC epoch and F1 threshold on A_val only','external':'B/C evaluation-only after model selection','seeds':SEEDS,'epochs':EPOCHS,'models':methods,'ehr_preprocessing':'median imputation + indicators + scaling, fitted A_train only'},open(OUT/'configs'/'representation_fusion_protocol.json','w'),indent=2)
