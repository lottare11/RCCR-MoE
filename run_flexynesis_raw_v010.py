from pathlib import Path
import sys,json,random
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
REPO=ROOT/'src/modern_baselines/_official_repos/flexynesis'
sys.path.insert(0,str(REPO))
from flexynesis.models.direct_pred import DirectPred
from flexynesis.data import MultiOmicDataset
sys.path.insert(0,str(ROOT/'src'))
from raw_rebuild import load_raw_modalities,fit_fold_selection,materialize_selected,fit_fold_scalers,apply_fold_scalers
from data import make_split,AlignedModalities
from train import binary_metrics,select_threshold

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT=ROOT/'results/baseline_rebuild_v010_20260922/multiomics/flexynesis'
OUT.mkdir(parents=True,exist_ok=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def make_ds(ds):
    dat={
      'SERS':torch.tensor(ds.sers,dtype=torch.float32),
      'Transcriptome':torch.tensor(ds.tran,dtype=torch.float32),
      'Metabolome':torch.tensor(ds.meta,dtype=torch.float32),
    }
    ann={'label':torch.tensor(ds.labels,dtype=torch.long)}
    feats={'SERS':[str(x) for x in ds.sers_names],
           'Transcriptome':[str(x) for x in ds.tran_names],
           'Metabolome':[str(x) for x in ds.meta_names]}
    return MultiOmicDataset(dat=dat,ann=ann,variable_types={'label':'categorical'},features=feats,
                            samples=[str(int(x)) for x in ds.ids],label_mappings={'label':{0:'HC',1:'RA'}})

def loaders(ds,batch=16,shuffle=False,seed=0):
    return DataLoader(TensorDataset(torch.tensor(ds.sers,dtype=torch.float32),
                                    torch.tensor(ds.tran,dtype=torch.float32),
                                    torch.tensor(ds.meta,dtype=torch.float32),
                                    torch.tensor(ds.labels,dtype=torch.long)),
                      batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed) if shuffle else None)

def predict(model,ds):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
      for s,t,m,y in loaders(ds,64,False):
        out=model.forward([s.to(DEVICE),t.to(DEVICE),m.to(DEVICE)])['label']
        ps.append(torch.softmax(out,1)[:,1].cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys),np.concatenate(ps)

raw=load_raw_modalities(ROOT/'data/raw_rebuild_20260916')
dummy=AlignedModalities(raw.sers,raw.tran_fpkm,raw.meta,raw.labels,raw.ids,raw.sers_names,raw.tran_names,raw.meta_names)
# compact grid inside official documented ranges
grid=[]
for latent in [32,64]:
  for hfac in [0.25,0.5]:
    for lr in [3e-4,1e-3]:
      grid.append({'latent_dim':latent,'hidden_dim_factor':hfac,'lr':lr,'supervisor_hidden_dim':16})

folds=[];oof=[];metas=[]
for fold in range(5):
    seed_all(1002+fold)
    tri,vi,tei=make_split(dummy,1002,fold,5)
    sel=fit_fold_selection(raw,tri,tran_k=512,meta_k=128)
    tr=materialize_selected(raw,tri,sel); va=materialize_selected(raw,vi,sel); te=materialize_selected(raw,tei,sel)
    for ds in (tr,va,te):
      mu=ds.sers.mean(1,keepdims=True); sd=ds.sers.std(1,keepdims=True); ds.sers=((ds.sers-mu)/np.maximum(sd,1e-8)).astype(np.float32)
    sc=fit_fold_scalers(tr); tr=apply_fold_scalers(tr,sc); va=apply_fold_scalers(va,sc); te=apply_fold_scalers(te,sc)
    proto=make_ds(tr)
    best_cfg=None;best_state=None;best_auc=-1;best_ep=-1
    for ci,cfg in enumerate(grid):
      seed_all(1002+fold*100+ci)
      model=DirectPred(config=cfg,dataset=proto,target_variables=['label'],use_loss_weighting=False,device_type='gpu').to(DEVICE)
      opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
      bad=0;local_best=-1;local_state=None;local_ep=-1
      for ep in range(200):
        model.train()
        for s,t,m,y in loaders(tr,16,True,1002+fold*1000+ci):
          opt.zero_grad()
          logits=model.forward([s.to(DEVICE),t.to(DEVICE),m.to(DEVICE)])['label']
          loss=F.cross_entropy(logits,y.to(DEVICE)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        vy,vp=predict(model,va); a=roc_auc_score(vy,vp)
        if a>local_best+1e-10:
          local_best=float(a); local_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};local_ep=ep;bad=0
        else:
          bad+=1
          if bad>=25: break
      if local_best>best_auc:
        best_auc=local_best;best_cfg=cfg.copy();best_state=local_state;best_ep=local_ep
      del model; torch.cuda.empty_cache()
    seed_all(1002+fold)
    model=DirectPred(config=best_cfg,dataset=proto,target_variables=['label'],use_loss_weighting=False,device_type='gpu').to(DEVICE)
    model.load_state_dict(best_state)
    vy,vp=predict(model,va);thr=select_threshold(vy,vp,'balanced_acc')
    ty,tp=predict(model,te);met=binary_metrics(ty,tp,thr);met.update({'fold':fold,'best_val_auc':best_auc,'best_epoch':best_ep,'selected_params':json.dumps(best_cfg,sort_keys=True)})
    folds.append(met);yp=(tp>=thr).astype(int)
    for sid,y,p,pr in zip(te.ids,ty,tp,yp):
      oof.append({'method':'Flexynesis DirectPred','subject_id':int(sid),'fold':fold,'true_label':int(y),'predicted_probability':float(p),'predicted_label':int(pr),'threshold':float(thr)})
    metas.append({'fold':fold,'train_size':len(tr.ids),'val_size':len(va.ids),'test_size':len(te.ids),'sers_dim':1796,'tran_dim':512,'meta_dim':128,'selected_params':json.dumps(best_cfg,sort_keys=True),'best_epoch':best_ep})
    print('flexynesis',fold,met,flush=True)
    del model;torch.cuda.empty_cache()
pd.DataFrame(folds).to_csv(OUT/'fold_metrics.csv',index=False);pd.DataFrame(oof).sort_values(['fold','subject_id']).to_csv(OUT/'oof_predictions.csv',index=False);pd.DataFrame(metas).to_csv(OUT/'fold_metadata.csv',index=False)
summ={}
for k in ['auc','auprc','acc','f1','specificity','brier']:
  a=np.array([r[k] for r in folds]);summ[k+'_mean']=float(a.mean());summ[k+'_std']=float(a.std(ddof=1))
(OUT/'summary.json').write_text(json.dumps({'summary':summ,'official_repo':'https://github.com/BIMSBbioinfo/flexynesis','commit':'f1305e798a2867a9defff50356604269fe5581e4','release':'1.1.14','model':'DirectPred','adaptation':'official DirectPred class; externally materialized current fold-selected 1796/512/128 features; compact validation-only grid; project validation thresholding'},indent=2))
