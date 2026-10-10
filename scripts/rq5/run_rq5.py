#!/usr/bin/env python3
"""Frozen-protocol RQ5 target-support adaptation runner. Query labels are metrics-only."""
from pathlib import Path
import argparse, csv, json, os, random, sys, time, traceback
import numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, recall_score, confusion_matrix, brier_score_loss
ROOT=Path(__file__).resolve().parents[4]; OUT=Path(__file__).resolve().parents[1]; V=ROOT/'vlbd_benchmark'
SOTA=V/'analysis'/'RQ3_SOTA_EXTENSION'; RQ3=V/'analysis'/'RQ3'; RQ2=V/'analysis'/'RQ2'
sys.path.insert(0,str(SOTA))
from models.drfuse_3d import DrFuse3D
from models.medfuse_3d import MedFuse3D
from models.m3care_3d import M3Care3D
from models.irene_3d import IRENE3D
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; SEEDS=[20260925,20260926,20260927,20260928,20260929]
# Exact frozen RQ3 generic cross-modal Transformer definition.
class CT(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Conv3d(1,16,3,2,1),nn.BatchNorm3d(16),nn.ReLU(),nn.Conv3d(16,32,3,2,1),nn.BatchNorm3d(32),nn.ReLU(),nn.Conv3d(32,64,3,2,1),nn.BatchNorm3d(64),nn.ReLU(),nn.AdaptiveAvgPool3d(1),nn.Flatten())
 def forward(self,x): return self.net(x)
class EHR(nn.Module):
 def __init__(self,n): super().__init__(); self.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,64),nn.ReLU())
 def forward(self,x): return self.net(x)
class Fusion(nn.Module):
 def __init__(self,n,kind='cross_modal_transformer'):
  super().__init__(); self.ct=CT();self.ehr=EHR(n);self.kind=kind;self.pos=nn.Parameter(torch.zeros(1,2,64));self.trans=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,batch_first=True,dropout=.1),num_layers=2);self.head=nn.Linear(64,2)
 def forward(self,x,z): return self.head(self.trans(torch.stack([self.ct(x),self.ehr(z)],1)+self.pos).mean(1))
manifest=pd.read_csv(ROOT/'benchmark_manifest.csv').set_index('case_id'); yall=manifest.ht_label.astype(int)
ehr_raw=pd.read_csv(ROOT/'metadata'/'ehr.csv').set_index('case_id'); drop={'patient_id','center','potential_post_treatment_leakage','leakage_flagged_columns'}; features=[c for c in ehr_raw if c not in drop]; X=ehr_raw[features].apply(pd.to_numeric,errors='coerce')
a_train=pd.read_csv(V/'protocol'/'splits'/'A_train.csv').case_id.tolist(); a_val=pd.read_csv(V/'protocol'/'splits'/'A_val.csv').case_id.tolist()
prep=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]); prep.fit(X.loc[a_train]); XE={cid:row.astype('float32') for cid,row in zip(X.index,prep.transform(X))}
ctcache={}
def ct(cid):
 if cid not in ctcache: ctcache[cid]=torch.from_numpy(np.asanyarray(nib.load(str(ROOT/manifest.loc[cid,'ct_processed_path'])).dataobj).astype('float32'))[None]
 return ctcache[cid]
def inputs(ids,needs_sequence=False):
 c=torch.stack([ct(x) for x in ids]); e=torch.from_numpy(np.stack([XE[x] for x in ids])); return c,e[:,None,:] if needs_sequence else e
def ece(y,p,bins=10):
 ans=0.; edges=np.linspace(0,1,bins+1)
 for i in range(bins):
  q=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else 1.00001))
  if q.any(): ans+=q.mean()*abs(float(p[q].mean())-float(y[q].mean()))
 return float(ans)
def metrics(y,p,t):
 y=np.asarray(y).astype(int);p=np.clip(np.asarray(p,dtype=float),0,1);q=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
 return dict(AUROC=float(roc_auc_score(y,p)),AUPRC=float(average_precision_score(y,p)),Brier=float(brier_score_loss(y,p)),ECE=ece(y,p),F1=float(f1_score(y,q,zero_division=0)),BalancedAccuracy=float(balanced_accuracy_score(y,q)),Sensitivity=float(recall_score(y,q,zero_division=0)),Specificity=float(tn/(tn+fp) if tn+fp else np.nan))
def append_log(msg):
 with open(OUT/'logs'/'core.log','a') as f:f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+msg+'\n')
def write_result(row):
 p=OUT/'results'/'rq5_seed_results.csv'; first=not p.exists();
 with open(p,'a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if first:w.writeheader()
  w.writerow(row)
def threshold_for(model,seed):
 if model=='Generic Cross-modal Transformer': p=RQ3/'representation_fusion'/'predictions'/f'cross_modal_transformer_seed{seed}_A_val.csv'
 else: p=SOTA/'predictions'/f'{model.lower().replace(" ","_")}_seed{seed}_A_val.csv'
 d=pd.read_csv(p); return float(d.threshold.iloc[0]) if 'threshold' in d else .5
def build(model,seed):
 n=len(next(iter(XE.values())))
 if model=='Generic Cross-modal Transformer': net=Fusion(n); ck=RQ3/'representation_fusion'/'checkpoints'/f'cross_modal_transformer_seed{seed}.pt'; head=net.head; seq=False
 elif model=='DrFuse': net=DrFuse3D(n); ck=SOTA/'checkpoints'/'drfuse'/f'seed{seed}.pt'; head=net.final; seq=True
 elif model=='MedFuse': net=MedFuse3D(n); ck=SOTA/'checkpoints'/'medfuse'/f'seed{seed}.pt'; head=net.cls; seq=True
 elif model=='M3Care': net=M3Care3D(n); ck=SOTA/'checkpoints'/'m3care'/f'seed{seed}.pt'; head=net.cls; seq=False
 elif model=='IRENE': net=IRENE3D(n); ck=SOTA/'checkpoints'/'unified_transformer'/f'seed{seed}.pt'; head=net.head; seq=False
 else: raise ValueError(model)
 state=torch.load(ck,map_location='cpu',weights_only=True); net.load_state_dict(state); return net.to(DEVICE),head,seq
def logits(net,model,c,e):
 if model=='Generic Cross-modal Transformer': return net(c,e)[:,1]-net(c,e)[:,0]
 if model=='DrFuse': return net(e,c,torch.ones(len(c),device=DEVICE))['pf']
 if model=='MedFuse': return net(e,c,torch.ones(len(c),device=DEVICE))[2]
 if model=='M3Care': return net(e,c,torch.ones((len(c),2),device=DEVICE))[0]
 if model=='IRENE': return net(e,c)
def predict(net,model,ids,seq):
 net.eval(); out=[]
 with torch.no_grad(),torch.autocast(device_type='cuda',enabled=DEVICE=='cuda'):
  for i in range(0,len(ids),2):
   z=ids[i:i+2];c,e=inputs(z,seq); out.extend(torch.sigmoid(logits(net,model,c.to(DEVICE),e.to(DEVICE))).float().cpu().numpy().tolist())
 return np.asarray(out)
def adapt(net,head,model,support,seq):
 for p in net.parameters():p.requires_grad=False
 for p in head.parameters():p.requires_grad=True
 net.eval(); head.train(); opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4); ys=torch.tensor(yall.loc[support].to_numpy(),dtype=torch.float32,device=DEVICE)
 rng=np.random.default_rng(7919+len(support)+int(ys.sum().item()))
 st=time.time()
 for step in range(25):
  ix=rng.choice(len(support),size=min(2,len(support)),replace=False); ids=[support[i] for i in ix]; c,e=inputs(ids,seq); target=ys[ix]
  opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type='cuda',enabled=DEVICE=='cuda'):
   loss=nn.functional.binary_cross_entropy_with_logits(logits(net,model,c.to(DEVICE),e.to(DEVICE)),target)
  loss.backward(); opt.step()
 return time.time()-st,sum(p.numel() for p in head.parameters())
def save_head_checkpoint(net,model,tag):
 prefix={'Generic Cross-modal Transformer':'head.','DrFuse':'final.','MedFuse':'cls.','M3Care':'cls.','IRENE':'head.'}[model]
 head={k:v.detach().cpu() for k,v in net.state_dict().items() if k.startswith(prefix)}
 target=OUT/'checkpoints'/'head_only'/f'{tag}_head_only.pt'
 torch.save({'format':'RQ5_head_only_v1','source_checkpoint_reference':'frozen source checkpoint for this model/seed','terminal_head_prefix':prefix,'state_dict':head},target)
 chk=torch.load(target,map_location='cpu',weights_only=True)
 assert set(chk['state_dict'])==set(head)

def deep_job(model,center,seed,shot):
 tag=f'{model.replace(" ","_")}_{center}_seed{seed}_shot{shot}'; ap=OUT/'predictions'/'adapted'/f'{tag}.csv'
 if ap.exists(): append_log('SKIP existing '+tag);return
 support=pd.read_csv(OUT/'splits'/f'{center}_seed{seed}_support{shot}.csv').case_id.tolist(); query=pd.read_csv(OUT/'splits'/f'{center}_seed{seed}_query.csv').case_id.tolist(); yy=yall.loc[query].to_numpy()
 torch.manual_seed(seed+shot);np.random.seed(seed+shot);random.seed(seed+shot); net,head,seq=build(model,seed); z=predict(net,model,query,seq); zp=OUT/'predictions'/'zero_shot_matched'/f'{tag}.csv'; pd.DataFrame({'case_id':query,'y_true':yy,'probability':z}).to_csv(zp,index=False)
 elapsed,nparams=adapt(net,head,model,support,seq); a=predict(net,model,query,seq);t=threshold_for(model,seed); pd.DataFrame({'case_id':query,'y_true':yy,'probability':a,'source_probability':z,'threshold':t}).to_csv(ap,index=False);save_head_checkpoint(net,model,tag)
 for typ,p in [('zero_shot',z),('adapted',a)]:
  r={'Model':model,'Center':center,'Seed':seed,'Shot':shot,'Type':typ,'AdaptationType':'head_only','AdaptationSeconds':elapsed if typ=='adapted' else 0.,'TrainableParameters':nparams if typ=='adapted' else 0.,'NQuery':len(query),'NSupport':len(support),'Threshold':t};r.update(metrics(yy,p,t));write_result(r)
 append_log('DONE '+tag+f' sec={elapsed:.2f}')
 del net;torch.cuda.empty_cache()
def source_models(seed):
 pp=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]);tr=pp.fit_transform(X.loc[a_train]);
 xgb=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.8,eval_metric='logloss',random_state=seed,n_jobs=8);rf=RandomForestClassifier(n_estimators=500,min_samples_leaf=3,class_weight='balanced',random_state=seed,n_jobs=-1);xgb.fit(tr,yall.loc[a_train]);rf.fit(tr,yall.loc[a_train]);return pp,xgb,rf
def score_calibration_jobs(model,center,seed,shot):
 tag=f'{model.replace(" ","_")}_{center}_seed{seed}_shot{shot}'; ap=OUT/'predictions'/'adapted'/f'{tag}.csv'
 if ap.exists():append_log('SKIP existing '+tag);return
 sup=pd.read_csv(OUT/'splits'/f'{center}_seed{seed}_support{shot}.csv').case_id.tolist();q=pd.read_csv(OUT/'splits'/f'{center}_seed{seed}_query.csv').case_id.tolist();pp,xgb,rf=source_models(seed)
 def raw(ids):
  xx=pp.transform(X.loc[ids]);
  if model=='XGBoost':return xgb.predict_proba(xx)[:,1]
  from monai.networks.nets import DenseNet121
  ck=torch.load(RQ2/'checkpoints'/f'ct_DenseNet121_seed{seed}.pt',map_location='cpu',weights_only=True);net=DenseNet121(spatial_dims=3,in_channels=1,out_channels=2).to(DEVICE);net.load_state_dict(ck);net.eval();cp=[]
  with torch.no_grad(),torch.autocast(device_type='cuda',enabled=DEVICE=='cuda'):
   for i in range(0,len(ids),2): cp.extend(torch.softmax(net(torch.stack([ct(x) for x in ids[i:i+2]]).to(DEVICE)),1)[:,1].float().cpu().numpy().tolist())
  return .5*(np.asarray(cp)+rf.predict_proba(xx)[:,1])
 s=raw(sup);z=raw(q);cal=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,random_state=seed);cal.fit(s.reshape(-1,1),yall.loc[sup]); slope=float(cal.coef_[0,0])
 # Frozen protocol requires a non-decreasing score calibration: preserve rank/AUROC.
 # A non-positive unconstrained Platt slope is unsupported by tiny support data; retain identity.
 if slope>0: a=cal.predict_proba(z.reshape(-1,1))[:,1];calibration='increasing_platt'
 else: a=z.copy();calibration='identity_nonpositive_platt_slope'
 yy=yall.loc[q].to_numpy();zp=OUT/'predictions'/'zero_shot_matched'/f'{tag}.csv';pd.DataFrame({'case_id':q,'y_true':yy,'probability':z}).to_csv(zp,index=False);pd.DataFrame({'case_id':q,'y_true':yy,'probability':a,'source_probability':z,'threshold':.5,'calibration_slope':slope,'calibration_rule':calibration}).to_csv(ap,index=False)
 for typ,p in [('zero_shot',z),('adapted',a)]:
  r={'Model':model,'Center':center,'Seed':seed,'Shot':shot,'Type':typ,'AdaptationType':'Platt_calibration','AdaptationSeconds':0.,'TrainableParameters':2 if typ=='adapted' else 0,'NQuery':len(q),'NSupport':len(sup),'Threshold':.5};r.update(metrics(yy,p,.5));write_result(r)
 append_log('DONE '+tag)
def smoke():
 ids=a_train[:2]
 for model in ['Generic Cross-modal Transformer','DrFuse','MedFuse','M3Care','IRENE']:
  net,head,seq=build(model,SEEDS[0]); p=predict(net,model,ids,seq); assert len(p)==2 and np.isfinite(p).all(),model; del net;torch.cuda.empty_cache();append_log('SMOKE_OK '+model)
 print('SMOKE_OK')
def run(stage):
 models=['Generic Cross-modal Transformer','DrFuse','MedFuse','M3Care','IRENE'] if stage in ['deep','all'] else []
 refs=['XGBoost','Source-selected probability fusion'] if stage in ['reference','all'] else []
 jobs=[(c,s,shot) for c,shots in [('B',[5,10]),('C',[5,10,20,40])] for s in SEEDS for shot in shots]
 for model in models:
  for c,s,shot in jobs: deep_job(model,c,s,shot)
 for model in refs:
  for c,s,shot in jobs: score_calibration_jobs(model,c,s,shot)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--stage',choices=['deep','reference','all'],default='all');args=ap.parse_args()
 try:
  smoke() if args.smoke else run(args.stage)
 except Exception:
  append_log('FATAL\n'+traceback.format_exc());raise
