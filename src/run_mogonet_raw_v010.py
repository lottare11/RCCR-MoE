from pathlib import Path
import sys, json, csv, random, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from raw_rebuild import load_raw_modalities, fit_fold_selection, materialize_selected, fit_fold_scalers, apply_fold_scalers
from data import make_split, AlignedModalities
from train import binary_metrics, select_threshold

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'src/modern_baselines/_official_repos/MOGONET'
sys.path.insert(0,str(REPO))
if not hasattr(np,'asscalar'): np.asscalar=lambda a:a.item()
from models import init_model_dict, init_optim
from utils import cal_sample_weight, cal_adj_mat_parameter, gen_adj_mat_tensor, gen_test_adj_mat_tensor
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT=ROOT/'results/baseline_rebuild_v010_20260922/multiomics/mogonet'
OUT.mkdir(parents=True,exist_ok=True)

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def tensor_list(ds):
 return [torch.tensor(ds.sers,dtype=torch.float32,device=DEVICE),torch.tensor(ds.tran,dtype=torch.float32,device=DEVICE),torch.tensor(ds.meta,dtype=torch.float32,device=DEVICE)]

def train_epoch(md,opt,data,adj,y,sw,vcdn):
 crit=torch.nn.CrossEntropyLoss(reduction='none')
 for m in md.values(): m.train()
 for i in range(3):
  opt[f'C{i+1}'].zero_grad()
  o=md[f'C{i+1}'](md[f'E{i+1}'](data[i],adj[i]))
  l=torch.mean(crit(o,y)*sw); l.backward(); opt[f'C{i+1}'].step()
 if vcdn:
  opt['C'].zero_grad()
  os=[md[f'C{i+1}'](md[f'E{i+1}'](data[i],adj[i])) for i in range(3)]
  o=md['C'](os); l=torch.mean(crit(o,y)*sw); l.backward(); opt['C'].step()

def pred_one(md,tr_list,x_list,params):
 probs=[]
 for xi in zip(*[x.detach().cpu().numpy() for x in x_list]):
  all_lists=[]; adjs=[]
  for tr, xv, par in zip(tr_list,xi,params):
   xv=torch.tensor(xv,dtype=torch.float32,device=DEVICE).view(1,-1)
   both=torch.cat([tr,xv],0); idx={'tr':list(range(tr.shape[0])),'te':[tr.shape[0]]}
   adj=gen_test_adj_mat_tensor(both,idx,par,'cosine')
   all_lists.append(both); adjs.append(adj)
  for m in md.values(): m.eval()
  with torch.no_grad():
   os=[md[f'C{i+1}'](md[f'E{i+1}'](all_lists[i],adjs[i])) for i in range(3)]
   p=F.softmax(md['C'](os)[-1:],1)[0,1].item()
  probs.append(p)
 return np.array(probs)

raw=load_raw_modalities(ROOT/'data/raw_rebuild_20260916')
dummy=AlignedModalities(raw.sers,raw.tran_fpkm,raw.meta,raw.labels,raw.ids,raw.sers_names,raw.tran_names,raw.meta_names)
folds=[]; oof=[]; metas=[]
for fold in range(5):
 seed_all(1002+fold); tri,vi,tei=make_split(dummy,1002,fold,5)
 sel=fit_fold_selection(raw,tri,tran_k=512,meta_k=128)
 tr=materialize_selected(raw,tri,sel); va=materialize_selected(raw,vi,sel); te=materialize_selected(raw,tei,sel)
 for ds in (tr,va,te):
  mu=ds.sers.mean(1,keepdims=True); sd=ds.sers.std(1,keepdims=True); ds.sers=((ds.sers-mu)/np.maximum(sd,1e-8)).astype(np.float32)
 sc=fit_fold_scalers(tr); tr=apply_fold_scalers(tr,sc); va=apply_fold_scalers(va,sc); te=apply_fold_scalers(te,sc)
 trl=tensor_list(tr); val=tensor_list(va); tel=tensor_list(te)
 params=[cal_adj_mat_parameter(2,x,'cosine') for x in trl]
 adj=[gen_adj_mat_tensor(x,p,'cosine') for x,p in zip(trl,params)]
 md=init_model_dict(3,2,[1796,512,128],[200,200,100],8,gcn_dopout=.5)
 for m in md.values(): m.to(DEVICE)
 y=torch.tensor(tr.labels,dtype=torch.long,device=DEVICE); sw=torch.tensor(cal_sample_weight(tr.labels,2),dtype=torch.float32,device=DEVICE)
 opt=init_optim(3,md,lr_e=1e-3,lr_c=1e-3)
 for _ in range(50): train_epoch(md,opt,trl,adj,y,sw,False)
 opt=init_optim(3,md,lr_e=5e-4,lr_c=1e-3)
 best=None; best_auc=-1; bad=0; best_ep=-1
 for ep in range(200):
  train_epoch(md,opt,trl,adj,y,sw,True)
  if ep%5: continue
  vp=pred_one(md,trl,val,params); a=roc_auc_score(va.labels,vp)
  if a>best_auc+1e-10:
   best_auc=float(a); bad=0; best_ep=ep; best={k:{kk:vv.detach().cpu().clone() for kk,vv in v.state_dict().items()} for k,v in md.items()}
  else:
   bad+=1
   if bad>=5: break
 for k,st in best.items(): md[k].load_state_dict(st)
 vp=pred_one(md,trl,val,params); thr=select_threshold(va.labels,vp,'balanced_acc')
 tp=pred_one(md,trl,tel,params); met=binary_metrics(te.labels,tp,thr); met.update({'fold':fold,'best_val_auc':best_auc,'best_epoch':best_ep})
 folds.append(met); yp=(tp>=thr).astype(int)
 for sid,yv,pv,pr in zip(te.ids,te.labels,tp,yp): oof.append({'method':'MOGONET','subject_id':int(sid),'fold':fold,'true_label':int(yv),'predicted_probability':float(pv),'predicted_label':int(pr),'threshold':float(thr)})
 metas.append({'fold':fold,'train_size':len(tr.ids),'val_size':len(va.ids),'test_size':len(te.ids),'sers_dim':1796,'tran_dim':512,'meta_dim':128,'mapping':'each validation/test sample mapped individually to fixed training graph','best_epoch':best_ep})
 print('mogonet',fold,met,flush=True)
pd.DataFrame(folds).to_csv(OUT/'fold_metrics.csv',index=False); pd.DataFrame(oof).sort_values(['fold','subject_id']).to_csv(OUT/'oof_predictions.csv',index=False); pd.DataFrame(metas).to_csv(OUT/'fold_metadata.csv',index=False)
summ={}
for k in ['auc','auprc','acc','f1','specificity','brier']:
 a=np.array([r[k] for r in folds]); summ[k+'_mean']=float(a.mean()); summ[k+'_std']=float(a.std(ddof=1))
(OUT/'summary.json').write_text(json.dumps({'summary':summ,'official_repo':'https://github.com/txWang/MOGONET','official_commit':'32d9066b7fe289b3f7cdb496a5668834079c274c','adaptation':'official model/graph utilities; current raw-rebuild preprocessing; inductive one-sample-at-a-time eval graph to prevent test-test information flow'},indent=2))
