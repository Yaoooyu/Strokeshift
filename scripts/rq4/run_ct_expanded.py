import sys,hashlib
import numpy as np,pandas as pd,torch,torch.nn as nn
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import run_rq4 as r
from monai.networks.nets import resnet18,DenseNet121
from monai.networks.nets.swin_unetr import SwinTransformer
class G(nn.Module):
 def __init__(s,n,k):
  super().__init__();s.ct=r.CT();s.ehr=r.EHR(n);s.k=k
  if k=='concat_mlp':s.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,2))
  elif k=='gmu':s.gate=nn.Linear(128,64);s.head=nn.Linear(64,2)
  elif k=='attention':s.att=nn.MultiheadAttention(64,4,batch_first=True);s.head=nn.Linear(64,2)
 def forward(s,x,z):
  a,b=s.ct(x),s.ehr(z)
  if s.k=='concat_mlp':return s.head(torch.cat([a,b],1))
  if s.k=='gmu':g=torch.sigmoid(s.gate(torch.cat([a,b],1)));return s.head(g*a+(1-g)*b)
  q,_=s.att(torch.stack([a,b],1),torch.stack([a,b],1),torch.stack([a,b],1));return s.head(q.mean(1))
def simple():return nn.Sequential(nn.Conv3d(1,24,3,2,1),nn.ReLU(),nn.Conv3d(24,48,3,2,1),nn.ReLU(),nn.Conv3d(48,96,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(96,2))
def swin():
 class S(nn.Module):
  def __init__(s):super().__init__();s.b=SwinTransformer(in_chans=1,embed_dim=24,window_size=(7,7,7),patch_size=(2,2,2),depths=(2,2,2,2),num_heads=(3,6,12,24),spatial_dims=3);s.h=nn.Linear(384,2)
  def forward(s,x):return s.h(s.b(x)[-1].mean((2,3,4)))
 return S()
def degrade(x,cid,k,l,seed):
 if k=='resolution':
  f={'Low':.75,'Medium':.5,'High':.25}[l];z=torch.from_numpy(x)[None,None];q=torch.nn.functional.interpolate(z,scale_factor=f,mode='trilinear',align_corners=False);return torch.nn.functional.interpolate(q,size=x.shape,mode='trilinear',align_corners=False)[0,0].numpy()
 q=np.random.default_rng(seed+int(hashlib.sha1(cid.encode()).hexdigest()[:8],16));s={'Low':.05,'Medium':.1,'High':.2}[l];return x+q.normal(0,s*max(x.std(),1e-6),x.shape).astype('float32')
def pred(net,ct,z,kind):
 a=[];bs=2 if kind.startswith('ct_') else 6
 with torch.no_grad(),torch.autocast('cuda'):
  for x,e in zip(ct.split(bs),z.split(bs)):
   o=net(x.cuda()) if kind.startswith('ct_') else net(x.cuda(),e.cuda());a.append(torch.softmax(o,1)[:,1].float().cpu())
 return torch.cat(a).numpy()
def ctnet(name):
 if name=='SimpleCNN':return simple()
 if name=='ResNet18':return resnet18(spatial_dims=3,n_input_channels=1,num_classes=2)
 if name=='DenseNet121':return DenseNet121(spatial_dims=3,in_channels=1,out_channels=2)
 return swin()
def main():
 m,sp,raw,cols,prep=r.load_data();ids=sp['A_val']+sp['C_test'];y=m.loc[ids,'ht_label'].astype(int).to_numpy();cut=len(sp['A_val']);z=torch.from_numpy(prep.transform(raw.loc[ids]).astype('float32'));out=[]
 for kind in ['resolution','noise']:
  for lev in ['Low','Medium','High']:
   for cs in ([0] if kind=='resolution' else r.NSEEDS):
    ct=r.ct_tensor(m,ids,lambda x,c:degrade(x,c,kind,lev,cs))
    for name in ['SimpleCNN','ResNet18','DenseNet121','SwinTransformer']:
     for ms in r.MSEEDS:
      net=ctnet(name);net.load_state_dict(torch.load(r.RQ2/'checkpoints'/f'ct_{name}_seed{ms}.pt',map_location='cpu',weights_only=True));net=net.cuda().eval();p=pred(net,ct,z,'ct_'+name);t=float(pd.read_csv(r.RQ2/'predictions'/f'ct_{name}_seed{ms}_A_val.csv').threshold.iloc[0])
      for split,a,b in [('A_val',0,cut),('C_test',cut,len(ids))]:r.rows(out,'ct_degradation',kind,lev,cs,'ct_'+name,ms,split,ids[a:b],y[a:b],p[a:b],t)
      del net;torch.cuda.empty_cache()
    for name in ['concat_mlp','gmu','attention']:
     for ms in r.MSEEDS:
      net=G(z.shape[1],name);net.load_state_dict(torch.load(r.RQ3/'representation_fusion'/'checkpoints'/f'{name}_seed{ms}.pt',map_location='cpu',weights_only=True));net=net.cuda().eval();p=pred(net,ct,z,name);t=float(pd.read_csv(r.RQ3/'representation_fusion'/'predictions'/f'{name}_seed{ms}_A_val.csv').threshold.iloc[0])
      for split,a,b in [('A_val',0,cut),('C_test',cut,len(ids))]:r.rows(out,'ct_degradation',kind,lev,cs,name,ms,split,ids[a:b],y[a:b],p[a:b],t)
      del net;torch.cuda.empty_cache()
    print('expanded CT',kind,lev,cs,flush=True)
 pd.DataFrame(out).to_csv(r.OUT/'tables'/'rq4_ct_expanded_raw_metrics.csv',index=False)
if __name__=='__main__':main()
