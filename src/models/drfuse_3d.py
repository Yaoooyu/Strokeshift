import math
import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import resnet50

class EHRBranch(nn.Module):
    def __init__(self, n_features, hidden=256, heads=4, dropout=.3):
        super().__init__()
        self.emb=nn.Linear(n_features,hidden)
        self.pos=nn.Parameter(torch.empty(1,1,hidden).uniform_(-.1,.1))
        lay=lambda: nn.TransformerEncoderLayer(hidden,heads,dropout=dropout,batch_first=True)
        self.feat=nn.TransformerEncoder(lay(),1)
        self.shared=nn.TransformerEncoder(lay(),1)
        self.distinct=nn.TransformerEncoder(lay(),1)
        self.cls=nn.Linear(hidden,1)
    def forward(self,x):
        h=self.feat(self.emb(x)+self.pos)
        s=self.shared(h).mean(1); d=self.distinct(h).mean(1)
        return s,d,self.cls(d).squeeze(1)

class DrFuse3D(nn.Module):
    def __init__(self,n_features,hidden=256,heads=4,dropout=.3):
        super().__init__()
        self.ehr=EHRBranch(n_features,hidden,heads,dropout)
        self.ct_shared=resnet50(spatial_dims=3,n_input_channels=1,num_classes=hidden)
        self.ct_distinct=resnet50(spatial_dims=3,n_input_channels=1,num_classes=hidden)
        self.project=nn.Sequential(nn.Linear(hidden,hidden*2),nn.ReLU(),nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.ehr_head=nn.Linear(hidden,1); self.ct_head=nn.Linear(hidden,1); self.shared_head=nn.Linear(hidden,1)
        self.attn=nn.Linear(hidden,3*hidden); self.final=nn.Linear(hidden,1)
    def forward(self,ehr,ct,pairs):
        es,ed,pe0=self.ehr(ehr); cs=self.ct_shared(ct); cd=self.ct_distinct(ct)
        es=self.project(es);cs=self.project(cs)
        pair=pairs.float().unsqueeze(1)
        term1=torch.stack([es+cs,es+cs,es,cs],2);term2=torch.stack([torch.zeros_like(es),torch.zeros_like(es),es,cs],2)
        shared=torch.logsumexp(term1,2)-torch.logsumexp(term2,2)
        shared=pair*shared+(1-pair)*es
        pe=self.ehr_head(ed).squeeze(1);pc=self.ct_head(cd).squeeze(1);ps=self.shared_head(shared).squeeze(1)
        feats=torch.stack([ed,shared,cd],1); qv=self.attn(feats);q,v,k=qv.chunk(3,dim=-1)
        qm=pair*q.mean(1)+(1-pair)*q[:,:2].mean(1)
        logits=torch.einsum('bd,bnkd->bnk',qm,k.unsqueeze(1))/math.sqrt(q.shape[-1])
        mask=torch.ones_like(logits);mask[pairs==0,:,-1]=0
        w=F.softmax(logits.masked_fill(mask==0,float('-inf')),dim=-1)
        f=torch.matmul(w,v).squeeze(1);pf=self.final(f).squeeze(1)
        return dict(es=es,cs=cs,ed=ed,cd=cd,shared=shared,pe=pe,pc=pc,ps=ps,pf=pf,w=w)

def drfuse_loss(o,y,pairs,lam=1.0):
    bce=nn.BCEWithLogitsLoss(reduction='none')
    p=pairs.float()
    pred=bce(o['pf'],y).mean()+bce(o['pe'],y).mean()+((bce(o['pc'],y)*p).sum()/p.sum().clamp_min(1))+bce(o['ps'],y).mean()
    a=F.log_softmax(o['es'],1);b=F.log_softmax(o['cs'],1);m=.5*(a+b)
    js=.5*(F.kl_div(a,m,reduction='batchmean',log_target=True)+F.kl_div(b,m,reduction='batchmean',log_target=True))
    de=(F.cosine_similarity(o['es'],o['ed']).abs().mean()+F.cosine_similarity(o['cs'],o['cd']).abs().mul(p).sum()/p.sum().clamp_min(1))
    raw=torch.stack([bce(o['pe'],y),bce(o['ps'],y),bce(o['pc'],y)],1).detach()
    w=o['w'].squeeze(1);rank=0.
    for i,j in [(0,1),(0,2),(1,2)]:
        target=2*(raw[:,i]<raw[:,j]).float()-1
        rank=rank+(F.margin_ranking_loss(w[:,i],w[:,j],target,reduction='none')*p).sum()/p.sum().clamp_min(1)
    return pred+lam*(js+de+rank),dict(pred=pred.detach(),jsd=js.detach(),disentangle=de.detach(),attention_rank=rank.detach())
