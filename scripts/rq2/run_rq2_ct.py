#!/usr/bin/env python3
"""Frozen-split CT-only RQ2 baselines; selection is exclusively on A_val."""
import json, random
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,recall_score,confusion_matrix,brier_score_loss
from monai.networks.nets import resnet18,DenseNet121
from monai.networks.nets.swin_unetr import SwinTransformer
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/'vlbd_benchmark'/'analysis'/'RQ2'; SEEDS=[20260925+i for i in range(5)]; DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; EPOCHS=8; BATCH=6
for p in ['configs','predictions','per_seed_results','tables','figures','checkpoints','logs']:(OUT/p).mkdir(parents=True,exist_ok=True)
m=pd.read_csv(ROOT/'benchmark_manifest.csv').set_index('case_id'); splits={k:pd.read_csv(ROOT/'vlbd_benchmark'/'protocol'/'splits'/f'{k}.csv').case_id.tolist() for k in ['A_train','A_val','B_test','C_test']}
ids=sum([splits[k] for k in splits],[]); cache={}
for n,cid in enumerate(ids,1):
 x=np.asanyarray(nib.load(str(ROOT/m.loc[cid,'ct_processed_path'])).dataobj).astype('float32'); cache[cid]=torch.from_numpy(x)[None]
 if n%100==0:print('cached',n,flush=True)
class DS(Dataset):
 def __init__(s,ids,aug=False):s.ids=ids;s.aug=aug
 def __len__(s):return len(s.ids)
 def __getitem__(s,i):
  x=cache[s.ids[i]].clone()
  if s.aug:
   for d in [1,2,3]:
    if torch.rand(())<.5:x=torch.flip(x,[d])
  return x,torch.tensor(int(m.loc[s.ids[i],'ht_label'])),s.ids[i]
def swin():
 class M(nn.Module):
  def __init__(s):super().__init__();s.b=SwinTransformer(in_chans=1,embed_dim=24,window_size=(7,7,7),patch_size=(2,2,2),depths=(2,2,2,2),num_heads=(3,6,12,24),spatial_dims=3);s.h=nn.Linear(384,2)
  def forward(s,x):return s.h(s.b(x)[-1].mean((2,3,4)))
 return M()
def simple():
 return nn.Sequential(nn.Conv3d(1,24,3,2,1),nn.ReLU(),nn.Conv3d(24,48,3,2,1),nn.ReLU(),nn.Conv3d(48,96,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(96,2))
def models():return {'SimpleCNN':simple(),'ResNet18':resnet18(spatial_dims=3,n_input_channels=1,num_classes=2),'DenseNet121':DenseNet121(spatial_dims=3,in_channels=1,out_channels=2),'SwinTransformer':swin()}
def metrics(y,p,t):
 q=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel();return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'f1':f1_score(y,q),'balanced_accuracy':balanced_accuracy_score(y,q),'sensitivity':recall_score(y,q),'specificity':tn/(tn+fp),'brier':brier_score_loss(y,p)}
def pred(net,loader):
 net.eval();a=[]
 with torch.no_grad():
  for x,y,c in loader:a.extend(zip(c,y.numpy(),torch.softmax(net(x.to(DEVICE)),1)[:,1].cpu().numpy()))
 return pd.DataFrame(a,columns=['case_id','y_true','probability'])
rows=[]
for seed in SEEDS:
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 tr=DataLoader(DS(splits['A_train'],True),batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=True); loaders={k:DataLoader(DS(splits[k]),batch_size=BATCH,num_workers=2,pin_memory=True) for k in ['A_val','B_test','C_test']}; pos=sum(m.loc[splits['A_train'],'ht_label']);w=torch.tensor([(384-pos)/pos],device=DEVICE)
 for name,net in models().items():
  net=net.to(DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4);loss=nn.CrossEntropyLoss(weight=torch.tensor([1.,float(w)],device=DEVICE));best=(-1,None)
  for ep in range(EPOCHS):
   net.train()
   for x,y,_ in tr:opt.zero_grad();z=loss(net(x.to(DEVICE)),y.to(DEVICE));z.backward();opt.step()
   v=pred(net,loaders['A_val']);score=roc_auc_score(v.y_true,v.probability)
   if score>best[0]:best=(score,{k:v.cpu().clone() for k,v in net.state_dict().items()})
  net.load_state_dict(best[1]);torch.save(net.state_dict(),OUT/'checkpoints'/f'ct_{name}_seed{seed}.pt');v=pred(net,loaders['A_val']);grid=np.linspace(.05,.95,181);thr=max(grid,key=lambda t:f1_score(v.y_true,v.probability>=t))
  for split,ld in loaders.items():
   d=v if split=='A_val' else pred(net,ld);d['threshold']=thr;d.to_csv(OUT/'predictions'/f'ct_{name}_seed{seed}_{split}.csv',index=False)
   for k,val in metrics(d.y_true.to_numpy(),d.probability.to_numpy(),thr).items():rows.append({'modality':'CT','model':name,'seed':seed,'split':split,'metric':k,'value':val,'threshold':thr})
  print('done',name,seed,flush=True)
pd.DataFrame(rows).to_csv(OUT/'rq2_ct_per_seed_results.csv',index=False)
pd.DataFrame(rows).groupby(['modality','model','split','metric']).agg(mean=('value','mean'),sd=('value','std'),n_seeds=('seed','nunique')).reset_index().to_csv(OUT/'tables'/'RQ2_1_ct_performance.csv',index=False)
json.dump({'seeds':SEEDS,'epochs':EPOCHS,'batch_size':BATCH,'input':'CT_processed, 128 cubed; no B/C fitting or selection','augmentation':'random axis flips on A_train only','threshold':'maximal F1 on A_val only'},open(OUT/'configs'/'ct_protocol.json','w'),indent=2)
