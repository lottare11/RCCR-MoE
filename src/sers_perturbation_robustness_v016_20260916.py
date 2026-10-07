from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from torch.utils.data import DataLoader
from data import AlignedModalities,TripleTensorDataset,build_feature_group_spec
from raw_rebuild import load_raw_modalities,fit_fold_selection,materialize_selected,fit_fold_scalers,apply_fold_scalers
from model import build_tricor_moe
from train_raw_rebuild_sersnorm_20260916 import apply_sers_sample_norm,predict_probs,fixed_mask

def clone(d,x): return AlignedModalities(x,d.tran,d.meta,d.labels,d.ids,d.sers_names,d.tran_names,d.meta_names)
def shifted(x,k):
    y=np.empty_like(x)
    if k>0: y[:,:k]=x[:,:1]; y[:,k:]=x[:,:-k]
    else: q=-k; y[:,-q:]=x[:,-1:]; y[:,:-q]=x[:,q:]
    return y

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--noise-reps',type=int,default=20); a=ap.parse_args()
    run=Path(a.run_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    args=torch.load(run/'fold_0_best.pt',map_location='cpu')['args']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=load_raw_modalities(args['raw_data_dir']); splits=list(StratifiedKFold(args['n_splits'],shuffle=True,random_state=args['seed']).split(np.zeros_like(data.labels),data.labels)); rows=[]
    for fold,(train_val_idx,test_idx) in enumerate(splits):
        train_idx,_=train_test_split(train_val_idx,test_size=.2,random_state=args['seed']+fold,stratify=data.labels[train_val_idx])
        sel=fit_fold_selection(data,train_idx,tran_k=args['tran_k'],meta_k=args['meta_k'],tran_prevar_k=args['tran_prevar_k'],expressed_fpkm=args['expressed_fpkm'],expressed_fraction=args['expressed_fraction'])
        tr=materialize_selected(data,train_idx,sel); te=materialize_selected(data,test_idx,sel)
        tr=apply_sers_sample_norm(tr,args['sers_sample_norm']); te=apply_sers_sample_norm(te,args['sers_sample_norm'])
        gs=build_feature_group_spec(tr,seed=args['seed']+fold,sers_groups=args['sers_groups'],tran_groups=args['tran_groups'],meta_groups=args['meta_groups'])
        scal=fit_fold_scalers(tr); base=apply_fold_scalers(clone(te,te.sers.copy()),scal)
        ck=torch.load(run/f'fold_{fold}_best.pt',map_location='cpu'); model=build_tricor_moe(gs,d_model=args['d_model'],dropout=args['dropout'],use_consensus_token=not args['no_consensus_token'],use_conflict_token=not args['no_conflict_token'],use_shared_private=not args['no_shared_private'],use_moe_routing=not args['no_moe_routing'],use_refinement_head=not args['no_refinement_head']).to(device)
        model.load_state_dict(ck['state_dict']); model.eval(); mask=fixed_mask((1,1,1))
        dl=DataLoader(TripleTensorDataset(base.sers,base.tran,base.meta,base.labels),batch_size=len(base.labels),shuffle=False); y,p0=predict_probs(model,dl,device,mask); bauc=roc_auc_score(y,p0)
        rows.append({'fold':fold,'condition':'baseline','rep':0,'auc':bauc,'auc_drop':0.0,'mean_abs_prob_shift':0.0})
        for k in [-4,-2,-1,1,2,4]:
            pert=apply_fold_scalers(clone(te,shifted(te.sers,k)),scal); dl=DataLoader(TripleTensorDataset(pert.sers,pert.tran,pert.meta,pert.labels),batch_size=len(pert.labels),shuffle=False); yy,pp=predict_probs(model,dl,device,mask)
            rows.append({'fold':fold,'condition':f'shift_{k:+d}_index','rep':0,'auc':roc_auc_score(yy,pp),'auc_drop':bauc-roc_auc_score(yy,pp),'mean_abs_prob_shift':float(np.mean(np.abs(p0-pp)))})
        scale=np.maximum(te.sers.std(axis=1,keepdims=True),1e-6)
        for frac in [0.01,0.03,0.05]:
            for rep in range(a.noise_reps):
                rng=np.random.default_rng(args['seed']+fold*10000+int(frac*1000)*100+rep)
                xp=te.sers + rng.normal(0.0,frac,size=te.sers.shape).astype(np.float32)*scale
                pert=apply_fold_scalers(clone(te,xp.astype(np.float32)),scal); dl=DataLoader(TripleTensorDataset(pert.sers,pert.tran,pert.meta,pert.labels),batch_size=len(pert.labels),shuffle=False); yy,pp=predict_probs(model,dl,device,mask); auc=roc_auc_score(yy,pp)
                rows.append({'fold':fold,'condition':f'noise_{frac:.2f}sd','rep':rep,'auc':auc,'auc_drop':bauc-auc,'mean_abs_prob_shift':float(np.mean(np.abs(p0-pp)))})
    d=pd.DataFrame(rows); d.to_csv(out/'perturbation_by_fold_rep.csv',index=False)
    agg=(d.groupby('condition',as_index=False).agg(auc_mean=('auc','mean'),auc_std=('auc','std'),auc_drop_mean=('auc_drop','mean'),auc_drop_std=('auc_drop','std'),prob_shift_mean=('mean_abs_prob_shift','mean'),n=('auc','size')).sort_values('auc_drop_mean'))
    agg.to_csv(out/'perturbation_summary.csv',index=False)
    summary={'run_dir':str(run),'sers_sample_norm':args['sers_sample_norm'],'seed':args['seed'],'noise_reps':a.noise_reps,'summary':agg.to_dict('records'),'guardrail':'Inference-only perturbation on held-out folds. Training, feature selection, splits and hyperparameters unchanged.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
