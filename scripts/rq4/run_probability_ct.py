import sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).resolve().parent));import run_rq4 as r
from monai.networks.nets import DenseNet121
def deg(x,cid,k,l,s):
 if k=='resolution':
  f={'Low':.75,'Medium':.5,'High':.25}[l];z=torch.from_numpy(x)[None,None];q=torch.nn.functional.interpolate(z,scale_factor=f,mode='trilinear',align_corners=False);return torch.nn.functional.interpolate(q,size=x.shape,mode='trilinear',align_corners=False)[0,0].numpy()
 q=np.random.default_rng(s+int(hashlib.sha1(cid.encode()).hexdigest()[:8],16));a={'Low':.05,'Medium':.1,'High':.2}[l];return x+q.normal(0,a*max(x.std(),1e-6),x.shape).astype('float32')
def main():
 m,sp,raw,cols,prep=r.load_data();ids=sp['A_val']+sp['C_test'];y=m.loc[ids,'ht_label'].astype(int).to_numpy();cut=len(sp['A_val']);out=[];preds=[]
 for k in ['resolution','noise']:
  for l in ['Low','Medium','High']:
   for cs in ([0] if k=='resolution' else r.NSEEDS):
    ct=r.ct_tensor(m,ids,lambda x,c:deg(x,c,k,l,cs))
    for ms in r.MSEEDS:
     net=DenseNet121(spatial_dims=3,in_channels=1,out_channels=2);net.load_state_dict(torch.load(r.RQ2/'checkpoints'/f'ct_DenseNet121_seed{ms}.pt',map_location='cpu',weights_only=True));net=net.cuda().eval();a=[]
     with torch.no_grad(),torch.autocast('cuda'):
      for x in ct.split(6):a.append(torch.softmax(net(x.cuda()),1)[:,1].float().cpu())
     pc=torch.cat(a).numpy();t=float(pd.read_csv(r.RQ3/'probability_fusion'/'predictions'/f'equal_average_ct-DenseNet121_ehr-RandomForest_seed{ms}_A_val.csv').threshold.iloc[0])
     for split,a,b in [('A_val',0,cut),('C_test',cut,len(ids))]:
      rf=pd.read_csv(r.RQ2/'predictions'/f'ehr_RandomForest_seed{ms}_{split}.csv').set_index('case_id').loc[ids[a:b]].probability.to_numpy();p=.5*(pc[a:b]+rf);r.rows(out,'ct_degradation',k,l,cs,'source_selected_probability_fusion',ms,split,ids[a:b],y[a:b],p,t)
      preds.extend(dict(case_id=cid,y_true=int(yy),probability=float(pp),split=split,condition=k,level=l,corruption_seed=cs,model_seed=ms) for cid,yy,pp in zip(ids[a:b],y[a:b],p))
     del net;torch.cuda.empty_cache()
    print('prob fusion',k,l,cs,flush=True)
 pd.DataFrame(out).to_csv(r.OUT/'tables'/'rq4_probability_fusion_ct_raw_metrics.csv',index=False);pd.DataFrame(preds).to_csv(r.OUT/'predictions'/'rq4_probability_fusion_ct_predictions.csv',index=False)
if __name__=='__main__':main()
