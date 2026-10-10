import torch
from torch import nn
from monai.networks.nets import resnet18
class M3Care3D(nn.Module):
 def __init__(self,n,hidden=128,heads=4):
  super().__init__();self.ehr=nn.Sequential(nn.Linear(n,hidden),nn.ReLU(),nn.Linear(hidden,hidden));self.ct=resnet18(spatial_dims=3,n_input_channels=1,num_classes=hidden)
  self.sim=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(2)])
  self.gcn=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(2)])
  self.comp=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(2)])
  self.types=nn.Embedding(2,hidden);layer1=nn.TransformerEncoderLayer(hidden,heads,hidden*4,batch_first=True,dropout=.5);layer2=nn.TransformerEncoderLayer(hidden,1,hidden*4,batch_first=True,dropout=.5);self.mm1=nn.TransformerEncoder(layer1,1);self.mm2=nn.TransformerEncoder(layer2,1);self.cls=nn.Linear(hidden,1)
 def forward(self,ehr,ct,mask):
  z=[self.ehr(ehr),self.ct(ct)]
  out=[];closs=0.
  for i,h in enumerate(z):
   s=self.sim[i](h);a=torch.softmax(s@s.T/(h.shape[-1]**.5),-1);g=self.gcn[i](a@h);missing=~mask[:,i].bool();im=self.comp[i](g)
   closs=closs+((h-im).pow(2).mean(1)*missing.float()).sum()/missing.float().sum().clamp_min(1)
   out.append(torch.where(mask[:,i,None].bool(),h,im))
  tok=torch.stack(out,1)+self.types(torch.arange(2,device=ehr.device))[None]
  h=self.mm2(self.mm1(tok)).mean(1);return self.cls(h).squeeze(1),closs
