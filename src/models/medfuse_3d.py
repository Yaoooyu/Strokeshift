import torch
from torch import nn
from monai.networks.nets import resnet50
class StaticLSTM(nn.Module):
 def __init__(self,n,hidden=128):
  super().__init__();self.lstm=nn.LSTM(n,hidden,batch_first=True);self.cls=nn.Linear(hidden,1)
 def forward(self,x):
  h,_=self.lstm(x);return h[:,-1],self.cls(h[:,-1]).squeeze(1)
class MedFuse3D(nn.Module):
 def __init__(self,n,hidden=128):
  super().__init__();self.ehr=StaticLSTM(n,hidden);self.ct=resnet50(spatial_dims=3,n_input_channels=1,num_classes=hidden);self.ctcls=nn.Linear(hidden,1);self.proj=nn.Linear(hidden,hidden);self.missing=nn.Parameter(torch.rand(hidden));self.fuse=nn.LSTM(hidden,hidden,batch_first=True);self.cls=nn.Linear(hidden,1)
 def forward(self,ehr,ct,pairs):
  eh,pe=self.ehr(ehr);ch=self.ct(ct);pc=self.ctcls(ch).squeeze(1);cp=self.proj(ch);cp=torch.where(pairs[:,None].bool(),cp,self.missing[None]);h,_=self.fuse(torch.stack([eh,cp],1));pf=self.cls(h[:,-1]).squeeze(1);return pe,pc,pf
