import torch
from torch import nn
class FeedForward(nn.Module):
 def __init__(self,h,drop=.3):
  super().__init__();self.net=nn.Sequential(nn.Linear(h,4*h),nn.GELU(),nn.Dropout(drop),nn.Linear(4*h,h),nn.Dropout(drop))
 def forward(self,x):return self.net(x)
class BiModalBlock(nn.Module):
 def __init__(self,h=128,heads=4,drop=.2):
  super().__init__();self.vi=nn.LayerNorm(h);self.cl=nn.LayerNorm(h);self.vintra=nn.MultiheadAttention(h,heads,dropout=drop,batch_first=True);self.cintra=nn.MultiheadAttention(h,heads,dropout=drop,batch_first=True);self.vcross=nn.MultiheadAttention(h,heads,dropout=drop,batch_first=True);self.ccross=nn.MultiheadAttention(h,heads,dropout=drop,batch_first=True);self.vffn=nn.Sequential(nn.LayerNorm(h),FeedForward(h));self.cffn=nn.Sequential(nn.LayerNorm(h),FeedForward(h))
 def forward(self,v,c):
  vn,cn=self.vi(v),self.cl(c);vii=self.vintra(vn,vn,vn,need_weights=False)[0];cii=self.cintra(cn,cn,cn,need_weights=False)[0];vic=self.vcross(vn,cn,cn,need_weights=False)[0];civ=self.ccross(cn,vn,vn,need_weights=False)[0];v=v+.5*(vii+vic);c=c+.5*(cii+civ);return v+self.vffn(v),c+self.cffn(c)
class IRENE3D(nn.Module):
 def __init__(self,n_fields,hidden=128,heads=4,patch=32):
  super().__init__();self.patch=nn.Conv3d(1,hidden,kernel_size=patch,stride=patch);self.cls=nn.Parameter(torch.zeros(1,1,hidden));self.vpos=nn.Parameter(torch.zeros(1,65,hidden));self.field=nn.Parameter(torch.randn(1,n_fields,hidden)*.02);self.cvalue=nn.Linear(1,hidden);self.cmissing=nn.Embedding(2,hidden);self.cpos=nn.Parameter(torch.zeros(1,n_fields,hidden));self.drop=nn.Dropout(.3);self.bidirectional=nn.ModuleList([BiModalBlock(hidden,heads,.2) for _ in range(2)]);enc=nn.TransformerEncoderLayer(hidden,heads,4*hidden,dropout=.3,batch_first=True,norm_first=True,activation='gelu');self.self_attention=nn.TransformerEncoder(enc,10);self.norm=nn.LayerNorm(hidden);self.head=nn.Linear(hidden,1);nn.init.trunc_normal_(self.cls,std=.02)
 def forward(self,ehr,ct):
  v=self.patch(ct).flatten(2).transpose(1,2);v=torch.cat((self.cls.expand(len(ct),-1,-1),v),1)+self.vpos
  missing=torch.isnan(ehr);e=torch.nan_to_num(ehr).unsqueeze(-1);c=self.cvalue(e)+self.field+self.cpos+self.cmissing(missing.long())
  v,c=self.drop(v),self.drop(c)
  for block in self.bidirectional:v,c=block(v,c)
  z=self.self_attention(torch.cat((v,c),1));return self.head(self.norm(z).mean(1)).squeeze(1)
