#!/usr/bin/env python3
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,nibabel as nib,torch,torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,recall_score,confusion_matrix,brier_score_loss
from monai.networks.nets import resnet18,DenseNet121
HERE=Path(__file__).resolve(); ROOT=HERE.parents[4]; BENCH=ROOT/'vlbd_benchmark'; OUT=BENCH/'analysis'/'RQ4'; RQ2=BENCH/'analysis'/'RQ2'; RQ3=BENCH/'analysis'/'RQ3'; EXT=BENCH/'analysis'/'RQ3_SOTA_EXTENSION'
sys.path.insert(0,str(EXT)); from models.m3care_3d import M3Care3D
from models.drfuse_3d import DrFuse3D
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; MSEEDS=list(range(20260925,20260930)); CSEEDS=list(range(4101,4106)); NSEEDS=list(range(5201,5206))
class CT(nn.Module):
 def __init__(s): super().__init__();s.net=nn.Sequential(nn.Conv3d(1,16,3,2,1),nn.BatchNorm3d(16),nn.ReLU(),nn.Conv3d(16,32,3,2,1),nn.BatchNorm3d(32),nn.ReLU(),nn.Conv3d(32,64,3,2,1),nn.BatchNorm3d(64),nn.ReLU(),nn.AdaptiveAvgPool3d(1),nn.Flatten())
 def forward(s,x):return s.net(x)
class EHR(nn.Module):
 def __init__(s,n):super().__init__();s.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,64),nn.ReLU())
 def forward(s,x):return s.net(x)
class Fusion(nn.Module):
 def __init__(s,n):
  super().__init__();s.ct=CT();s.ehr=EHR(n);s.pos=nn.Parameter(torch.zeros(1,2,64));s.trans=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,batch_first=True,dropout=.1),num_layers=2);s.head=nn.Linear(64,2)
 def forward(s,x,z):return s.head(s.trans(torch.stack([s.ct(x),s.ehr(z)],1)+s.pos).mean(1))
def ece(y,p):
 e=np.linspace(0,1,11);return sum(q.mean()*abs(p[q].mean()-y[q].mean()) for i in range(10) if ((q:=(p>=e[i])&(p<(e[i+1] if i<9 else 1.000001))).any()))
def metrics(y,p,t):
 q=p>=t;tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel();return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),f1=f1_score(y,q),balanced_accuracy=balanced_accuracy_score(y,q),sensitivity=recall_score(y,q),specificity=tn/(tn+fp),brier=brier_score_loss(y,p),ece=ece(y,p))
def load_data():
 m=pd.read_csv(ROOT/'benchmark_manifest.csv').set_index('case_id');splits={k:pd.read_csv(BENCH/'protocol'/'splits'/f'{k}.csv').case_id.tolist() for k in ['A_train','A_val','B_test','C_test']}; e=pd.read_csv(ROOT/'metadata'/'ehr.csv').set_index('case_id');cols=[c for c in e.columns if c not in {'patient_id','center','potential_post_treatment_leakage','leakage_flagged_columns'}];raw=e[cols].apply(pd.to_numeric,errors='coerce'); prep=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]); prep.fit(raw.loc[splits['A_train']]); return m,splits,raw,cols,prep
def ct_tensor(m,ids,mut=None):
 a=[]
 for cid in ids:
  x=np.asanyarray(nib.load(str(ROOT/m.loc[cid,'ct_processed_path'])).dataobj).astype('float32'); a.append(mut(x,cid) if mut else x)
 return torch.from_numpy(np.stack(a))[:,None]
def load_model(name,seed,n):
 if name=='generic': net=Fusion(n); p=RQ3/'representation_fusion'/'checkpoints'/f'cross_modal_transformer_seed{seed}.pt'
 elif name=='m3care': net=M3Care3D(n);p=EXT/'checkpoints'/'m3care'/f'seed{seed}.pt'
 elif name=='drfuse': net=DrFuse3D(n);p=EXT/'checkpoints'/'drfuse'/f'seed{seed}.pt'
 elif name=='densenet': net=DenseNet121(spatial_dims=3,in_channels=1,out_channels=2);p=RQ2/'checkpoints'/f'ct_DenseNet121_seed{seed}.pt'
 net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));return net.to(DEVICE).eval()
def encode(net,name,ct):
 o=[]
 with torch.no_grad(),torch.autocast(DEVICE if DEVICE=='cuda' else 'cpu',enabled=DEVICE=='cuda'):
  for a in ct.split(2 if name=='drfuse' else 6):
   if name=='generic':o.append(net.ct(a.to(DEVICE)).float().cpu())
   elif name=='m3care':o.append(net.ct(a.to(DEVICE)).float().cpu())
   elif name=='drfuse':o.append(torch.stack([net.ct_shared(a.to(DEVICE)),net.ct_distinct(a.to(DEVICE))],1).float().cpu())
   else:o.append(torch.softmax(net(a.to(DEVICE)),1)[:,1].float().cpu()[:,None])
 return torch.cat(o)
def from_emb(net,name,ehr,emb,avail=(1,1)):
 out=[]; bs=128
 with torch.no_grad(),torch.autocast(DEVICE if DEVICE=='cuda' else 'cpu',enabled=DEVICE=='cuda'):
  for z,c in zip(ehr.split(bs),emb.split(bs)):
   z=z.to(DEVICE);c=c.to(DEVICE)
   if name=='generic':
    a=c if avail[1] else torch.zeros_like(c);b=net.ehr(z if avail[0] else torch.zeros_like(z));p=torch.softmax(net.head(net.trans(torch.stack([a,b],1)+net.pos).mean(1)),1)[:,1]
   elif name=='m3care':
    zz=[net.ehr(z),c];oo=[]
    for i,h in enumerate(zz):
     s=net.sim[i](h);aa=torch.softmax(s@s.T/(h.shape[-1]**.5),-1);g=net.gcn[i](aa@h);im=net.comp[i](g);oo.append(h if avail[i] else im)
    tok=torch.stack(oo,1)+net.types(torch.arange(2,device=DEVICE))[None];p=torch.sigmoid(net.cls(net.mm2(net.mm1(tok)).mean(1)).squeeze(1))
   elif name=='drfuse':
    es,ed,_=net.ehr((z if avail[0] else torch.zeros_like(z))[:,None]); es=net.project(es); cs=net.project(c[:,0]);cd=c[:,1];pair=torch.full((len(z),),float(avail[1]),device=DEVICE) # native pair switch for CT
    term1=torch.stack([es+cs,es+cs,es,cs],2);term2=torch.stack([torch.zeros_like(es),torch.zeros_like(es),es,cs],2);shared=torch.logsumexp(term1,2)-torch.logsumexp(term2,2);shared=pair[:,None]*shared+(1-pair[:,None])*es
    feats=torch.stack([ed,shared,cd],1);q,v,k=net.attn(feats).chunk(3,dim=-1);qm=pair[:,None]*q.mean(1)+(1-pair[:,None])*q[:,:2].mean(1);logits=torch.einsum('bd,bnkd->bnk',qm,k.unsqueeze(1))/(q.shape[-1]**.5);mask=torch.ones_like(logits);mask[pair==0,:,-1]=0;w=torch.softmax(logits.masked_fill(mask==0,float('-inf')),dim=-1);p=torch.sigmoid(net.final(torch.matmul(w,v).squeeze(1)).squeeze(1))
   else:p=c[:,0]
   out.append(p.float().cpu())
 return torch.cat(out).numpy()
def threshold(name,seed):
 paths={'generic':RQ3/'representation_fusion'/'predictions'/f'cross_modal_transformer_seed{seed}_A_val.csv','m3care':EXT/'predictions'/f'm3care_seed{seed}_A_val.csv','drfuse':EXT/'predictions'/f'drfuse_seed{seed}_A_val.csv','densenet':RQ2/'predictions'/f'ct_DenseNet121_seed{seed}_A_val.csv'};return float(pd.read_csv(paths[name]).threshold.iloc[0])
def rows(res,experiment,condition,level,cseed,name,mseed,split,ids,y,p,t):
 for k,v in metrics(y,p,t).items():res.append(dict(experiment=experiment,condition=condition,level=str(level),corruption_seed=cseed,model=name,model_seed=mseed,split=split,metric=k,value=v,n=len(ids),threshold=t))
def save_masks(kind,rate,seed,ids,cols,mask):
 q=pd.DataFrame(mask,index=ids,columns=cols);q.index.name='case_id';q.to_csv(OUT/'corruption_masks'/f'{kind}_rate{rate}_seed{seed}.csv')
def run_ehr():
 m,sp,raw,cols,prep=load_data(); ids_all=sum([sp[k] for k in ['A_val','B_test','C_test']],[]);ct=ct_tensor(m,ids_all);yall=m.loc[ids_all,'ht_label'].astype(int).to_numpy();starts=np.cumsum([0,len(sp['A_val']),len(sp['B_test']),len(sp['C_test'])]);res=[]
 for name in ['generic','m3care','drfuse']:
  for ms in MSEEDS:
   net=load_model(name,ms,prep.transform(raw.loc[sp['A_val']]).shape[1]);emb=encode(net,name,ct);t=threshold(name,ms)
   for typ in ['elementwise','featurewise']:
    for rate in [0,.1,.3,.5,.7]:
     for cs in ([0] if rate==0 else CSEEDS):
      rng=np.random.default_rng(100000+cs+int(rate*1000)+(0 if typ=='elementwise' else 50000)); mask=np.zeros((len(ids_all),len(cols)),bool); obs=~raw.loc[ids_all].isna().to_numpy()
      if typ=='elementwise': mask=(rng.random(mask.shape)<rate)&obs
      else:
       take=rng.choice(len(cols),size=round(rate*len(cols)),replace=False) if rate else [];mask[:,take]=True;mask&=obs
      xr=raw.loc[ids_all].copy();xr=pd.DataFrame(np.where(mask,np.nan,xr.to_numpy(copy=True)),index=xr.index,columns=xr.columns); zz=torch.from_numpy(prep.transform(xr).astype('float32'));p=from_emb(net,name,zz,emb)
      for j,k in enumerate(['A_val','B_test','C_test']):a,b=starts[j],starts[j+1];rows(res,'random_ehr',typ,rate,cs,name,ms,k,ids_all[a:b],yall[a:b],p[a:b],t)
      if ms==MSEEDS[0]:save_masks(typ,rate,cs,ids_all,cols,mask)
   rates=pd.read_csv(BENCH/'analysis'/'RQ1'/'ehr_missingness.csv').set_index('variable')
   for center in ['B','C']:
    for cs in CSEEDS:
     rng=np.random.default_rng(200000+cs+(0 if center=='B' else 1000));r=np.array([rates.loc[c,'missing_rate_'+center] for c in cols]);mask=(rng.random((len(sp['A_val']),len(cols)))<r)&(~raw.loc[sp['A_val']].isna().to_numpy());xr=raw.loc[sp['A_val']].copy();xr=pd.DataFrame(np.where(mask,np.nan,xr.to_numpy(copy=True)),index=xr.index,columns=xr.columns);z=torch.from_numpy(prep.transform(xr).astype('float32'));p=from_emb(net,name,z,emb[:len(sp['A_val'])]);rows(res,'center_structured',f'{center}_pattern','empirical',cs,name,ms,'A_val',sp['A_val'],yall[:len(sp['A_val'])],p,t)
     if ms==MSEEDS[0]:save_masks(f'{center}_pattern','empirical',cs,sp['A_val'],cols,mask)
   del net;torch.cuda.empty_cache();print('EHR done',name,ms,flush=True)
 pd.DataFrame(res).to_csv(OUT/'tables'/'rq4_ehr_raw_metrics.csv',index=False)
def run_modality():
 m,sp,raw,cols,prep=load_data();ids=sum([sp[k] for k in ['A_val','B_test','C_test']],[]);ct=ct_tensor(m,ids);yy=m.loc[ids,'ht_label'].astype(int).to_numpy();starts=np.cumsum([0,len(sp['A_val']),len(sp['B_test']),len(sp['C_test'])]);z=torch.from_numpy(prep.transform(raw.loc[ids]).astype('float32'));res=[]
 for name in ['generic','m3care','drfuse']:
  for ms in MSEEDS:
   net=load_model(name,ms,z.shape[1]);emb=encode(net,name,ct);t=threshold(name,ms)
   for cond,av in [('clean',(1,1)),('ehr_missing',(0,1)),('ct_missing',(1,0))]:
    p=from_emb(net,name,z,emb,av)
    for j,k in enumerate(['A_val','B_test','C_test']):a,b=starts[j],starts[j+1];rows(res,'modality_missing',cond,'whole',0,name,ms,k,ids[a:b],yy[a:b],p[a:b],t)
   del net;torch.cuda.empty_cache();print('Modality done',name,ms,flush=True)
 pd.DataFrame(res).to_csv(OUT/'tables'/'rq4_modality_raw_metrics.csv',index=False)
def degrade(x,cid,kind,level,seed):
 if kind=='resolution':
  f={'Low':.75,'Medium':.5,'High':.25}[level];z=torch.from_numpy(x)[None,None];q=torch.nn.functional.interpolate(z,scale_factor=f,mode='trilinear',align_corners=False);return torch.nn.functional.interpolate(q,size=x.shape,mode='trilinear',align_corners=False)[0,0].numpy()
 rng=np.random.default_rng(seed+int(hashlib.sha1(cid.encode()).hexdigest()[:8],16));s={'Low':.05,'Medium':.10,'High':.20}[level];return x+rng.normal(0,s*max(x.std(),1e-6),size=x.shape).astype('float32')
def run_ct():
 m,sp,raw,cols,prep=load_data();ids=sp['A_val']+sp['C_test'];y=m.loc[ids,'ht_label'].astype(int).to_numpy();cut=len(sp['A_val']);z=torch.from_numpy(prep.transform(raw.loc[ids]).astype('float32'));res=[]
 for kind in ['resolution','noise']:
  for lev in ['Low','Medium','High']:
   ss=[0] if kind=='resolution' else NSEEDS
   for cs in ss:
    ct=ct_tensor(m,ids,lambda x,c:degrade(x,c,kind,lev,cs))
    for name in ['densenet','generic','m3care','drfuse']:
     for ms in MSEEDS:
      net=load_model(name,ms,z.shape[1]);emb=encode(net,name,ct);p=from_emb(net,name,z,emb) if name!='densenet' else emb[:,0].numpy();t=threshold(name,ms);rows(res,'ct_degradation',kind,lev,cs,name,ms,'A_val',ids[:cut],y[:cut],p[:cut],t);rows(res,'ct_degradation',kind,lev,cs,name,ms,'C_test',ids[cut:],y[cut:],p[cut:],t);del net;torch.cuda.empty_cache()
    print('CT done',kind,lev,cs,flush=True)
 pd.DataFrame(res).to_csv(OUT/'tables'/'rq4_ct_raw_metrics.csv',index=False)
def sanity():
 m,sp,raw,cols,prep=load_data();ids=sp['A_val'];ct=ct_tensor(m,ids);z=torch.from_numpy(prep.transform(raw.loc[ids]).astype('float32'));q=[]
 for name in ['generic','m3care','drfuse','densenet']:
  for ms in MSEEDS:
   net=load_model(name,ms,z.shape[1]);p=from_emb(net,name,z,encode(net,name,ct)) if name!='densenet' else encode(net,name,ct)[:,0].numpy();files={'generic':RQ3/'representation_fusion'/'predictions'/f'cross_modal_transformer_seed{ms}_A_val.csv','m3care':EXT/'predictions'/f'm3care_seed{ms}_A_val.csv','drfuse':EXT/'predictions'/f'drfuse_seed{ms}_A_val.csv','densenet':RQ2/'predictions'/f'ct_DenseNet121_seed{ms}_A_val.csv'};old=pd.read_csv(files[name]).set_index('case_id').loc[ids].probability.to_numpy();q.append(dict(model=name,seed=ms,max_abs_error=float(np.abs(p-old).max()),mean_abs_error=float(np.abs(p-old).mean()),correlation=float(np.corrcoef(p,old)[0,1])));del net;torch.cuda.empty_cache()
 pd.DataFrame(q).to_csv(OUT/'tables'/'rq4_checkpoint_sanity.csv',index=False);print(pd.DataFrame(q).to_string(index=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['sanity','ehr','modality','ct']);x=a.parse_args();getattr(sys.modules[__name__],'run_'+x.stage if x.stage!='sanity' else 'sanity')()
