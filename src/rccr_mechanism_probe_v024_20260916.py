from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold,train_test_split
from raw_rebuild import load_raw_modalities,fit_fold_selection,materialize_selected,fit_fold_scalers,apply_fold_scalers
from data import build_feature_group_spec
from train_raw_rebuild_sersnorm_20260916 import apply_sers_sample_norm
from model_rccr_v023_20260916 import build_rccr_moe

def fwd(model,s,t,m,device):
    with torch.no_grad():
        o=model(torch.tensor(s,dtype=torch.float32,device=device),torch.tensor(t,dtype=torch.float32,device=device),torch.tensor(m,dtype=torch.float32,device=device))
    return {k:o[k].detach().cpu().numpy() for k in ['reliability','gate','pairwise_js']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--out-dir',required=True);a=ap.parse_args()
    run=Path(a.run_dir);outd=Path(a.out_dir);outd.mkdir(parents=True,exist_ok=True)
    cfg=torch.load(run/'fold_0_best.pt',map_location='cpu')['args'];seed=int(cfg['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=load_raw_modalities(cfg['raw_data_dir']);rows=[]
    skf=StratifiedKFold(n_splits=cfg['n_splits'],shuffle=True,random_state=seed)
    for fold,(train_val_idx,test_idx) in enumerate(skf.split(np.zeros_like(data.labels),data.labels)):
        train_idx,val_idx=train_test_split(train_val_idx,test_size=.2,random_state=seed+fold,stratify=data.labels[train_val_idx])
        sel=fit_fold_selection(data,train_idx,tran_k=cfg['tran_k'],meta_k=cfg['meta_k'],tran_prevar_k=cfg['tran_prevar_k'],expressed_fpkm=cfg['expressed_fpkm'],expressed_fraction=cfg['expressed_fraction'])
        tr=materialize_selected(data,train_idx,sel);te=materialize_selected(data,test_idx,sel)
        tr=apply_sers_sample_norm(tr,cfg['sers_sample_norm']);te=apply_sers_sample_norm(te,cfg['sers_sample_norm'])
        gs=build_feature_group_spec(tr,seed=seed+fold,sers_groups=cfg['sers_groups'],tran_groups=cfg['tran_groups'],meta_groups=cfg['meta_groups'])
        scal=fit_fold_scalers(tr);te=apply_fold_scalers(te,scal)
        ck=torch.load(run/f'fold_{fold}_best.pt',map_location='cpu')
        model=build_rccr_moe(gs,d_model=cfg['d_model'],dropout=cfg['dropout'],use_consensus_token=not cfg['no_consensus_token'],use_conflict_token=not cfg['no_conflict_token'],use_shared_private=not cfg['no_shared_private'],use_moe_routing=not cfg['no_moe_routing'],use_refinement_head=not cfg['no_refinement_head']).to(device)
        model.load_state_dict(ck['state_dict']);model.eval();base=fwd(model,te.sers,te.tran,te.meta,device)
        for mod_i,mod in enumerate(['sers','transcriptome','metabolome']):
            for kind,scale in [('noise05',.05),('noise10',.10),('noise20',.20),('zero',0.0)]:
                rng=np.random.default_rng(seed+fold*1000+mod_i*100+int(scale*1000))
                s,t,m=te.sers.copy(),te.tran.copy(),te.meta.copy()
                arr=[s,t,m][mod_i]
                if kind=='zero': arr[:]=0.0
                else: arr += rng.normal(0.0,scale,size=arr.shape).astype(np.float32)
                pert=fwd(model,s,t,m,device)
                for i,sid in enumerate(te.ids):
                    rows.append({'subject_id':int(sid),'fold':fold,'modality':mod,'condition':kind,'d_rel':float(pert['reliability'][i,mod_i]-base['reliability'][i,mod_i]),'d_gate':float(pert['gate'][i,mod_i]-base['gate'][i,mod_i]),'d_conflict_gate':float(pert['gate'][i,4]-base['gate'][i,4]),'d_js':float(pert['pairwise_js'][i].mean()-base['pairwise_js'][i].mean())})
    df=pd.DataFrame(rows);df.to_csv(outd/'dose_response_oof.csv',index=False)
    summary=df.groupby(['modality','condition'])[['d_rel','d_gate','d_conflict_gate','d_js']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join([str(x) for x in c if x!='']) if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(outd/'summary.csv',index=False);print(summary.to_string(index=False))
if __name__=='__main__':main()
