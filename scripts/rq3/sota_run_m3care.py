from pathlib import Path
import sys,random,json
import numpy as np,pandas as pd,nibabel as nib,torch
from torch.utils.data import Dataset,DataLoader
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,recall_score,confusion_matrix,brier_score_loss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from models.m3care_3d import M3Care3D
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parents[1];DEVICE='cuda';SEEDS=list(range(20260925,20260930));BATCH=2;ACCUM=8;EPOCHS=100;PATIENCE=10;COMP_WEIGHT=.1;MISSING_AUG=.3
for x in ['predictions','per_seed_results','logs','checkpoints/m3care','configs']:(OUT/x).mkdir(parents=True,exist_ok=True)
m=pd.read_csv(ROOT/'benchmark_manifest.csv').set_index('case_id');sp={k:pd.read_csv(ROOT/'vlbd_benchmark'/'protocol'/'splits'/f'{k}.csv').case_id.tolist() for k in ['A_train','A_val','B_test','C_test']}
e=pd.read_csv(ROOT/'metadata'/'ehr.csv').set_index('case_id');drop={'patient_id','center','potential_post_treatment_leakage','leakage_flagged_columns'};x=e[[c for c in e if c not in drop]].apply(pd.to_numeric,errors='coerce')
prep=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]);ehr={'A_train':prep.fit_transform(x.loc[sp['A_train']]).astype('float32')}
for k in ['A_val','B_test','C_test']:ehr[k]=prep.transform(x.loc[sp[k]]).astype('float32')
cache={}
def ctload(ids):
 for cid in ids:
  if cid not in cache:cache[cid]=torch.from_numpy(np.asanyarray(nib.load(str(ROOT/m.loc[cid,'ct_processed_path'])).dataobj).astype('float32'))[None]
class DS(Dataset):
 def __init__(s,k):s.k=k;s.ids=sp[k]
 def __len__(s):return len(s.ids)
 def __getitem__(s,i):
  cid=s.ids[i];return cache[cid],torch.from_numpy(ehr[s.k][i]),torch.tensor(float(m.loc[cid,'ht_label'])),cid
def predict(net,ld):
 net.eval();o=[]
 with torch.no_grad(),torch.autocast('cuda'):
  for ct,er,y,c in ld:
   z,_=net(er.to(DEVICE),ct.to(DEVICE),torch.ones((len(y),2),device=DEVICE));o.extend(zip(c,y.numpy(),torch.sigmoid(z).float().cpu().numpy()))
 return pd.DataFrame(o,columns=['case_id','y_true','probability'])
def getthreshold(y,p):return float(max(np.linspace(.05,.95,181),key=lambda t:f1_score(y,p>=t)))
def metrics(y,p,t):
 q=p>=t;tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel();return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'f1':f1_score(y,q),'balanced_accuracy':balanced_accuracy_score(y,q),'sensitivity':recall_score(y,q),'specificity':tn/(tn+fp),'brier':brier_score_loss(y,p)}
ctload(sp['A_train']+sp['A_val']);rows=[]
for seed in SEEDS:
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 tr=DataLoader(DS('A_train'),batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=True);va=DataLoader(DS('A_val'),batch_size=BATCH,num_workers=2,pin_memory=True)
 net=M3Care3D(ehr['A_train'].shape[1]).to(DEVICE);opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda');best=(-1,None);wait=0
 for ep in range(EPOCHS):
  net.train();opt.zero_grad(set_to_none=True)
  for i,(ct,er,y,c) in enumerate(tr):
   mask=torch.ones((len(y),2),device=DEVICE);d=torch.rand(len(y),device=DEVICE)<MISSING_AUG;w=torch.randint(0,2,(len(y),),device=DEVICE);mask[d,w[d]]=0
   with torch.autocast('cuda'):
    z,comp=net(er.to(DEVICE),ct.to(DEVICE),mask);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,y.to(DEVICE))+COMP_WEIGHT*comp)/ACCUM
   scaler.scale(loss).backward()
   if (i+1)%ACCUM==0 or i+1==len(tr):scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
  d=predict(net,va);a=roc_auc_score(d.y_true,d.probability)
  if a>best[0]:best=(a,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()});wait=0
  else:wait+=1
  if wait>=PATIENCE:break
 net.load_state_dict(best[1]);torch.save(net.state_dict(),OUT/'checkpoints/m3care'/f'seed{seed}.pt');va_pred=predict(net,va);t=getthreshold(va_pred.y_true,va_pred.probability)
 for k in ['A_val','B_test','C_test']:
  if k!='A_val':ctload(sp[k])
  d=va_pred if k=='A_val' else predict(net,DataLoader(DS(k),batch_size=BATCH,num_workers=2,pin_memory=True));d['threshold']=t;d.to_csv(OUT/'predictions'/f'm3care_seed{seed}_{k}.csv',index=False)
  for n,v in metrics(d.y_true.to_numpy(),d.probability.to_numpy(),t).items():rows.append(dict(model='M3Care',seed=seed,split=k,metric=n,value=v,threshold=t,best_val_auroc=best[0],epochs=ep+1))
 print('done',seed,'epoch',ep+1,'val',best[0],flush=True)
pd.DataFrame(rows).to_csv(OUT/'per_seed_results/m3care.csv',index=False)
json.dump({'source_only':True,'seeds':SEEDS,'hidden':128,'heads_first':4,'heads_second':1,'epochs_max':100,'patience':10,'batch':BATCH,'accumulation':ACCUM,'missing_modality_augmentation':MISSING_AUG,'compensation_weight':COMP_WEIGHT,'core':['modality-aware similarity','graph aggregation','latent compensation','two multimodal transformer aggregators']},open(OUT/'configs/m3care.json','w'),indent=2)
