#!/usr/bin/env python3
"""Strict source-only five-fold OOF CT probabilities for RQ3 fusion."""
from pathlib import Path
import json, random
import numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import StratifiedKFold
from monai.networks.nets import resnet18,DenseNet121
from monai.networks.nets.swin_unetr import SwinTransformer
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'vlbd_benchmark'/'analysis'/'RQ3'
SEED=20260925; EPOCHS=8; BATCH=6; DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
m=pd.read_csv(ROOT/'benchmark_manifest.csv').set_index('case_id')
ids=pd.read_csv(ROOT/'vlbd_benchmark'/'protocol'/'splits'/'A_train.csv').case_id.tolist()
assert len(ids)==384
cache={}
for i,cid in enumerate(ids,1):
    x=np.asanyarray(nib.load(str(ROOT/m.loc[cid,'ct_processed_path'])).dataobj).astype('float32')
    cache[cid]=torch.from_numpy(x)[None]
    if i%100==0: print('cached',i,flush=True)
class DS(Dataset):
    def __init__(self,ids,aug=False): self.ids=ids; self.aug=aug
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        x=cache[self.ids[i]].clone()
        if self.aug:
            for d in [1,2,3]:
                if torch.rand(())<.5: x=torch.flip(x,[d])
        return x,torch.tensor(int(m.loc[self.ids[i],'ht_label'])),self.ids[i]
def swin():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.b=SwinTransformer(in_chans=1,embed_dim=24,window_size=(7,7,7),patch_size=(2,2,2),depths=(2,2,2,2),num_heads=(3,6,12,24),spatial_dims=3)
            self.h=nn.Linear(384,2)
        def forward(self,x): return self.h(self.b(x)[-1].mean((2,3,4)))
    return M()
def simple():
    return nn.Sequential(nn.Conv3d(1,24,3,2,1),nn.ReLU(),nn.Conv3d(24,48,3,2,1),nn.ReLU(),nn.Conv3d(48,96,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(96,2))
def models():
    return {'SimpleCNN':simple(),'ResNet18':resnet18(spatial_dims=3,n_input_channels=1,num_classes=2),'DenseNet121':DenseNet121(spatial_dims=3,in_channels=1,out_channels=2),'SwinTransformer':swin()}
def pred(net,ld):
    net.eval(); out=[]
    with torch.no_grad():
        for x,y,c in ld:
            p=torch.softmax(net(x.to(DEVICE)),1)[:,1].cpu().numpy()
            out.extend(zip(c,y.numpy(),p))
    return pd.DataFrame(out,columns=['case_id','y_true','probability'])
y=m.loc[ids,'ht_label'].astype(int).to_numpy()
folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
out={k:[] for k in models()}
for fold,(tri,tei) in enumerate(folds.split(ids,y),1):
    tr=[ids[i] for i in tri]; te=[ids[i] for i in tei]
    foldseed=SEED+fold
    random.seed(foldseed); np.random.seed(foldseed); torch.manual_seed(foldseed); torch.cuda.manual_seed_all(foldseed)
    trld=DataLoader(DS(tr,True),batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=True)
    teld=DataLoader(DS(te),batch_size=BATCH,num_workers=2,pin_memory=True)
    pos=int(m.loc[tr,'ht_label'].sum()); wt=torch.tensor([1.,float((len(tr)-pos)/pos)],device=DEVICE)
    for name,net in models().items():
        net=net.to(DEVICE); opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4); loss=nn.CrossEntropyLoss(weight=wt)
        for epoch in range(EPOCHS):
            net.train()
            for x,yy,_ in trld:
                opt.zero_grad(); z=loss(net(x.to(DEVICE)),yy.to(DEVICE)); z.backward(); opt.step()
        d=pred(net,teld); d['fold']=fold; out[name].append(d)
        del net; torch.cuda.empty_cache()
        print('done',name,'fold',fold,flush=True)
for name,parts in out.items():
    d=pd.concat(parts,ignore_index=True).sort_values('case_id')
    assert len(d)==384 and d.case_id.nunique()==384 and set(d.case_id)==set(ids)
    d.to_csv(OUT/'oof'/f'ct_{name}_A_train_oof.csv',index=False)
json.dump({'source_split':'A_train only','n_cases':384,'folds':5,'splitter':'StratifiedKFold','random_state':SEED,'epochs':EPOCHS,'input':'CT_processed; 128 cubed','models':list(out)},open(OUT/'configs'/'ct_oof_protocol.json','w'),indent=2)
