from __future__ import annotations
import argparse,json,copy
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold,train_test_split
from torch.utils.data import DataLoader
from data import TripleTensorDataset,build_feature_group_spec
from raw_rebuild import load_raw_modalities,fit_fold_selection,materialize_selected,fit_fold_scalers,apply_fold_scalers
from train_raw_rebuild_sersnorm_20260916 import apply_sers_sample_norm,predict_probs,fixed_mask,select_threshold,binary_metrics
from model_rccr_v023_20260916 import build_rccr_moe

def quality_loss(clean, corrupt, idx, target_bad):
    rc=clean["reliability"][:,idx]; rp=corrupt["reliability"][:,idx]
    target_clean=torch.ones_like(rc)*0.9; tb=torch.ones_like(rp)*float(target_bad)
    margin=max(0.05,0.9-float(target_bad)-0.05)
    rank=F.relu(margin-(rc-rp)).mean()
    return F.mse_loss(rc,target_clean)+F.mse_loss(rp,tb)+rank

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--s0-run",required=True); ap.add_argument("--out-dir",required=True); ap.add_argument("--raw-data-dir",default="data/raw_rebuild_20260916"); ap.add_argument("--seed",type=int,default=1002); a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=load_raw_modalities(a.raw_data_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=a.seed); rows=[]; folds=[]
    for fold,(tv,teidx) in enumerate(skf.split(np.zeros_like(data.labels),data.labels)):
      tridx,vidx=train_test_split(tv,test_size=.2,random_state=a.seed+fold,stratify=data.labels[tv])
      ck0=torch.load(Path(a.s0_run)/f"fold_{fold}_best.pt",map_location="cpu"); cfg=ck0["args"]
      sel=fit_fold_selection(data,tridx,tran_k=cfg["tran_k"],meta_k=cfg["meta_k"],tran_prevar_k=cfg["tran_prevar_k"],expressed_fpkm=cfg["expressed_fpkm"],expressed_fraction=cfg["expressed_fraction"])
      tr=materialize_selected(data,tridx,sel); va=materialize_selected(data,vidx,sel); te=materialize_selected(data,teidx,sel)
      for d in (tr,va,te): apply_sers_sample_norm(d,cfg["sers_sample_norm"])
      gs=build_feature_group_spec(tr,seed=a.seed+fold,sers_groups=cfg["sers_groups"],tran_groups=cfg["tran_groups"],meta_groups=cfg["meta_groups"]); sc=fit_fold_scalers(tr); tr=apply_fold_scalers(tr,sc); va=apply_fold_scalers(va,sc); te=apply_fold_scalers(te,sc)
      model=build_rccr_moe(gs,d_model=cfg["d_model"],dropout=cfg["dropout"],use_consensus_token=True,use_conflict_token=True,use_shared_private=False,use_moe_routing=True,use_refinement_head=True).to(dev)
      # initialize shared S0-compatible parameters
      cur=model.state_dict(); src=ck0["state_dict"]; matched={k:v for k,v in src.items() if k in cur and cur[k].shape==v.shape}; cur.update(matched); model.load_state_dict(cur)
      for n,p in model.named_parameters(): p.requires_grad=("reliability_" in n or "rel_" in n)
      opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg["lr"],weight_decay=cfg["weight_decay"])
      dl=DataLoader(TripleTensorDataset(tr.sers,tr.tran,tr.meta,tr.labels),batch_size=cfg["batch_size"],shuffle=True)
      best=None; bestloss=1e9; patience=0
      for ep in range(cfg["epochs"]):
        model.train(); ls=[]
        for s,t,m,y in dl:
          s,t,m=s.to(dev),t.to(dev),m.to(dev); idx=int(torch.randint(0,3,(1,)).item()); clean=model(s,t,m)
          cs,ct,cm=s.clone(),t.clone(),m.clone()
          level=int(torch.randint(0,4,(1,)).item())
          if level==0: sigma,target_bad=0.05,0.75
          elif level==1: sigma,target_bad=0.10,0.60
          elif level==2: sigma,target_bad=0.20,0.40
          else: sigma,target_bad=None,0.10
          if idx==0:
            cs.zero_() if sigma is None else cs.add_(sigma*torch.randn_like(cs))
          elif idx==1:
            ct.zero_() if sigma is None else ct.add_(sigma*torch.randn_like(ct))
          else:
            cm.zero_() if sigma is None else cm.add_(sigma*torch.randn_like(cm))
          bad=model(cs,ct,cm); loss=quality_loss(clean,bad,idx,target_bad); opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item())
        ml=float(np.mean(ls))
        if ml<bestloss-1e-5: bestloss=ml; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience+=1
        if patience>=cfg["patience"]: break
      model.load_state_dict(best); torch.save({"state_dict":best,"args":cfg,"fold":fold,"group_spec":gs.__dict__},out/f"fold_{fold}_best.pt")
      vdl=DataLoader(TripleTensorDataset(va.sers,va.tran,va.meta,va.labels),batch_size=len(va.labels)); tdl=DataLoader(TripleTensorDataset(te.sers,te.tran,te.meta,te.labels),batch_size=len(te.labels))
      vy,vp=predict_probs(model,vdl,dev,fixed_mask((1,1,1))); thr=select_threshold(vy,vp,"balanced_acc"); ty,tp=predict_probs(model,tdl,dev,fixed_mask((1,1,1))); met=binary_metrics(ty,tp,thr); folds.append(met); rows.append({"fold":fold,**met}); print(fold,met)
    summ={k+"_mean":float(np.mean([x[k] for x in folds])) for k in folds[0]}; summ.update({k+"_std":float(np.std([x[k] for x in folds],ddof=1)) for k in folds[0]}); (out/"summary.json").write_text(json.dumps(summ,indent=2)); print(json.dumps(summ,indent=2))
if __name__=="__main__": main()
